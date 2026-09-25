// Backward of the projection: chain each visible gaussian's screen-space
// gradient (rasterize_backward: du, dv, dA, dB, dC, dopacity, dr, dg, db) to
// its parameters, written in the same layouts as the forward inputs:
//   grad_transforms[gid * 10 + ..] = d(mean xyz, quat wxyz (un-normalised),
//                                      log-scale xyz)
//   grad_opacity[gid]             = d(opacity logit)
//   grad_sh[gid * 3 * cpc2 + ..]  = d(SH block: 3 DC, channel-major rest)
// Gaussians not visible this frame are not touched (the host clears them).
//
// Forward being differentiated (compute_projected / project_visible):
//   p = W m + t;  Rg = R(q / |q|);  M3 = Rg diag(exp(ls));  Sigma = M3 M3^T
//   Sc = W Sigma W^T;  J rows j0 = (fx/z, 0, -fx tx/z^2), j1 = (0, fy/z, -fy ty/z^2)
//   with tx = clamp(x/z, +-1.3 tan(fovx/2)) z (likewise ty);
//   [a b; b c] = J Sc J^T + 0.3 I;  conic Q = [a b; b c]^-1 = [A B; B C]
//   (u, v) = (fx x/z + cx, fy y/z + cy);  opacity = sigmoid(logit)
//   colour = max(0, SH(dir) + 0.5),  dir = (m - camera centre) / |.|
//
// Matrices are WGSL mat3x3 (column-major: m[c][r]); everything stays in
// registers -- no runtime-indexed local arrays, which spill to local memory.

@group(0) @binding(0) var<uniform> u: ProjectUniforms;
@group(0) @binding(1) var<storage, read> transforms: array<f32>;
@group(0) @binding(2) var<storage, read> opacity_in: array<f32>;
@group(0) @binding(3) var<storage, read> sh_in: array<f32>;
@group(0) @binding(4) var<storage, read> global_from_compact: array<u32>;
@group(0) @binding(5) var<storage, read> screen_grads: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_transforms: array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_opacity: array<f32>;
@group(0) @binding(8) var<storage, read_write> grad_sh: array<f32>;

// Outer product a b^T as a mat3x3 (column c = a * b[c]).
fn outer3(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
}

// SH rows (3 * (degree + 1)^2 <= 48 floats) of the workgroup's 64
// gaussians, staged in shared memory: each gaussian's row is contiguous in
// the scene buffers but rows are scattered, so a thread reading its own row
// float by float touches 48 strided cache lines. Instead the workgroup
// copies whole rows with consecutive lanes, and grad_sh goes back the same
// way (each thread overwrites its row with the gradient in place).
const WG: u32 = 64u;
var<workgroup> sh_rows: array<f32, 3072>; // 64 x 48
var<workgroup> row_gid: array<u32, 64>;

// SH rest coefficient k (0-based, after DC) of channel ch of this thread's
// staged row, or 0 past the scene's degree.
fn sh_rest(row: u32, rest_pc: u32, act: u32, ch: u32, k: u32) -> f32 {
    if (k >= act) {
        return 0.0;
    }
    return sh_rows[row + 3u + ch * rest_pc + k];
}

@compute @workgroup_size(64)
fn project_backward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nw: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = lid.x;
    let compact = (wid.x + wid.y * nw.x) * WG + t;
    let in_frame = compact < u.num_visible;
    let cpc = u.sh_degree + 1u;
    let cpc2 = cpc * cpc;
    let rest_pc = cpc2 - 1u;
    let row_len = 3u * cpc2;
    var gid = 0u;
    if (in_frame) {
        gid = global_from_compact[compact];
    }
    row_gid[t] = select(0xFFFFFFFFu, gid, in_frame);
    workgroupBarrier();
    for (var i = t; i < WG * row_len; i = i + WG) {
        let r = i / row_len;
        let g = row_gid[r];
        if (g != 0xFFFFFFFFu) {
            sh_rows[r * 48u + (i % row_len)] = sh_in[g * row_len + (i % row_len)];
        }
    }
    workgroupBarrier();
    if (in_frame) {
        project_backward_one(compact, gid, t * 48u, rest_pc);
    }
    workgroupBarrier();
    for (var i = t; i < WG * row_len; i = i + WG) {
        let r = i / row_len;
        let g = row_gid[r];
        if (g != 0xFFFFFFFFu) {
            grad_sh[g * row_len + (i % row_len)] = sh_rows[r * 48u + (i % row_len)];
        }
    }
}

fn project_backward_one(compact: u32, gid: u32, row: u32, rest_pc: u32) {
    // Evaluated rest coefficients per channel (progressive SH degree).
    let act = (u.sh_active_degree + 1u) * (u.sh_active_degree + 1u) - 1u;
    let sg = compact * 10u;
    let g_u = screen_grads[sg + 0u];
    let g_v = screen_grads[sg + 1u];
    let g_ca = screen_grads[sg + 2u];
    let g_cb = screen_grads[sg + 3u];
    let g_cc = screen_grads[sg + 4u];
    let g_op = screen_grads[sg + 5u];
    let g_col = vec3<f32>(screen_grads[sg + 6u], screen_grads[sg + 7u], screen_grads[sg + 8u]);

    let base = gid * 10u;
    let m = vec3<f32>(transforms[base], transforms[base + 1u], transforms[base + 2u]);
    let q = vec4<f32>(transforms[base + 3u], transforms[base + 4u], transforms[base + 5u], transforms[base + 6u]);
    let ls = vec3<f32>(transforms[base + 7u], transforms[base + 8u], transforms[base + 9u]);

    let w = view_rot(u);
    let p = w * m + u.view_t.xyz;

    // Normalised quaternion and Rg (same as rot_from_quat).
    let n2 = dot(q, q);
    var qn = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    var inv_n = 0.0;
    if (n2 > 1e-12) {
        inv_n = inverseSqrt(n2);
        qn = q * inv_n;
    }
    let rg = rot_from_quat(qn.x, qn.y, qn.z, qn.w);
    let s = exp(ls);
    let m3 = mat3x3<f32>(rg[0] * s.x, rg[1] * s.y, rg[2] * s.z);
    let sigma = m3 * transpose(m3);
    let sc = w * sigma * transpose(w);

    // EWA Jacobian with the 1.3x frustum clamp.
    let inv_z = 1.0 / p.z;
    let inv_z2 = inv_z * inv_z;
    let lim_x = 1.3 * 0.5 * f32(u.img_w) / u.fx;
    let lim_y = 1.3 * 0.5 * f32(u.img_h) / u.fy;
    let rx = p.x * inv_z;
    let ry = p.y * inv_z;
    let clamped_x = rx < -lim_x || rx > lim_x;
    let clamped_y = ry < -lim_y || ry > lim_y;
    let tx = clamp(rx, -lim_x, lim_x) * p.z;
    let ty = clamp(ry, -lim_y, lim_y) * p.z;
    let j0 = vec3<f32>(u.fx * inv_z, 0.0, -u.fx * tx * inv_z2);
    let j1 = vec3<f32>(0.0, u.fy * inv_z, -u.fy * ty * inv_z2);
    let sj0 = sc * j0;
    let sj1 = sc * j1;
    let ca = dot(j0, sj0) + 0.3;
    let cb = dot(j0, sj1);
    let cc = dot(j1, sj1) + 0.3;
    let det = ca * cc - cb * cb;
    let qa = cc / det;
    let qb = -cb / det;
    let qc = ca / det;

    // Conic -> 2D covariance: dL/dM = -Q GQ Q with GQ = [gA gB/2; gB/2 gC].
    let gq01 = 0.5 * g_cb;
    let t00 = g_ca * qa + gq01 * qb;
    let t01 = g_ca * qb + gq01 * qc;
    let t10 = gq01 * qa + g_cc * qb;
    let t11 = gq01 * qb + g_cc * qc;
    let g_a = -(qa * t00 + qb * t10);
    let g_b = 2.0 * -(qa * t01 + qb * t11);
    let g_c = -(qb * t01 + qc * t11);

    // 2D covariance -> Sc (general gradient) and J.
    let gsc = outer3(j0, j0) * g_a + outer3(j0, j1) * g_b + outer3(j1, j1) * g_c;
    let gj0 = 2.0 * g_a * sj0 + g_b * sj1;
    let gj1 = 2.0 * g_c * sj1 + g_b * sj0;

    // J and the screen mean -> camera-space point.
    var gp = vec3<f32>(0.0, 0.0, 0.0);
    gp.z = gj0.x * (-u.fx * inv_z2) + gj1.y * (-u.fy * inv_z2);
    if (clamped_x) {
        gp.z = gp.z + gj0.z * u.fx * clamp(rx, -lim_x, lim_x) * inv_z2;
    } else {
        gp.x = gp.x + gj0.z * (-u.fx * inv_z2);
        gp.z = gp.z + gj0.z * 2.0 * u.fx * p.x * inv_z2 * inv_z;
    }
    if (clamped_y) {
        gp.z = gp.z + gj1.z * u.fy * clamp(ry, -lim_y, lim_y) * inv_z2;
    } else {
        gp.y = gp.y + gj1.z * (-u.fy * inv_z2);
        gp.z = gp.z + gj1.z * 2.0 * u.fy * p.y * inv_z2 * inv_z;
    }
    gp.x = gp.x + g_u * u.fx * inv_z;
    gp.y = gp.y + g_v * u.fy * inv_z;
    gp.z = gp.z - (g_u * u.fx * p.x + g_v * u.fy * p.y) * inv_z2;

    // p = W m + t  ->  dL/dm = W^T gp.
    var gmean = transpose(w) * gp;

    // Sc = W Sigma W^T -> GSigma = W^T GSc W; Sigma = M3 M3^T -> GM3.
    let gsig = transpose(w) * gsc * w;
    let gm3 = (gsig + transpose(gsig)) * m3;
    // M3 = Rg diag(s): column k of M3 is s_k * column k of Rg.
    let gls = vec3<f32>(dot(gm3[0], rg[0]) * s.x, dot(gm3[1], rg[1]) * s.y, dot(gm3[2], rg[2]) * s.z);
    let gr = mat3x3<f32>(gm3[0] * s.x, gm3[1] * s.y, gm3[2] * s.z);
    // Rg(qn) -> normalised quaternion; R(i, k) = gr[k][i].
    let qw = qn.x;
    let qx = qn.y;
    let qy = qn.z;
    let qz = qn.w;
    let r00 = gr[0][0];
    let r01 = gr[1][0];
    let r02 = gr[2][0];
    let r10 = gr[0][1];
    let r11 = gr[1][1];
    let r12 = gr[2][1];
    let r20 = gr[0][2];
    let r21 = gr[1][2];
    let r22 = gr[2][2];
    let gw = 2.0 * (-qz * r01 + qy * r02 + qz * r10 - qx * r12 - qy * r20 + qx * r21);
    let gx = 2.0 * (qy * r01 + qz * r02 + qy * r10 - 2.0 * qx * r11 - qw * r12 + qz * r20 + qw * r21 - 2.0 * qx * r22);
    let gy = 2.0 * (-2.0 * qy * r00 + qx * r01 + qw * r02 + qx * r10 + qz * r12 - qw * r20 + qz * r21 - 2.0 * qy * r22);
    let gz = 2.0 * (-2.0 * qz * r00 - qw * r01 + qx * r02 + qw * r10 - 2.0 * qz * r11 + qy * r12 + qx * r20 + qy * r21);
    let gqn = vec4<f32>(gw, gx, gy, gz);
    // qn = q / |q|  ->  gq = (gqn - qn (qn . gqn)) / |q|.
    let gq = (gqn - qn * dot(qn, gqn)) * inv_n;

    // Opacity.
    let op = 1.0 / (1.0 + exp(-opacity_in[gid]));
    grad_opacity[gid] = g_op * op * (1.0 - op);

    // Colour -> SH coefficients, and -> mean through the view direction.
    // Basis and its direction derivative are written out term by term (no
    // runtime-indexed arrays); coefficients past the scene degree read as 0.
    let dir_raw = m - u.camera_center.xyz;
    let dn = length(dir_raw);
    var dir = vec3<f32>(0.0, 0.0, 1.0);
    if (dn > 1e-6) {
        dir = dir_raw / dn;
    }
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let c0 = 0.4886025;
    let c3 = 1.0925485;
    let c5 = 0.3153916;
    // Inria / brush real-SH normalisation (utils/sh_utils.py C2, C3).
    let c8 = 0.5900436;
    let c9 = 2.8906114;
    let c10 = 0.4570458;
    let c11 = 0.3731763;
    let c2b = 0.5462742;
    let c14 = 1.4453057;
    // Basis values b1..b15 (b0 = C0 is the DC term).
    let b1 = -c0 * y;
    let b2 = c0 * z;
    let b3 = -c0 * x;
    let b4 = c3 * x * y;
    let b5 = -c3 * y * z;
    let b6 = c5 * (2.0 * zz - xx - yy);
    let b7 = -c3 * x * z;
    let b8 = c2b * (xx - yy);
    let b9 = -c8 * y * (3.0 * xx - yy);
    let b10 = c9 * x * y * z;
    let b11 = -c10 * y * (4.0 * zz - xx - yy);
    let b12 = c11 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    let b13 = -c10 * x * (4.0 * zz - xx - yy);
    let b14 = c14 * z * (xx - yy);
    let b15 = -c8 * x * (xx - 3.0 * yy);
    var gdir = vec3<f32>(0.0, 0.0, 0.0);
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let r1 = sh_rest(row, rest_pc, act, ch, 0u);
        let r2 = sh_rest(row, rest_pc, act, ch, 1u);
        let r3 = sh_rest(row, rest_pc, act, ch, 2u);
        let r4 = sh_rest(row, rest_pc, act, ch, 3u);
        let r5 = sh_rest(row, rest_pc, act, ch, 4u);
        let r6 = sh_rest(row, rest_pc, act, ch, 5u);
        let r7 = sh_rest(row, rest_pc, act, ch, 6u);
        let r8 = sh_rest(row, rest_pc, act, ch, 7u);
        let r9 = sh_rest(row, rest_pc, act, ch, 8u);
        let r10 = sh_rest(row, rest_pc, act, ch, 9u);
        let r11 = sh_rest(row, rest_pc, act, ch, 10u);
        let r12 = sh_rest(row, rest_pc, act, ch, 11u);
        let r13 = sh_rest(row, rest_pc, act, ch, 12u);
        let r14 = sh_rest(row, rest_pc, act, ch, 13u);
        let r15 = sh_rest(row, rest_pc, act, ch, 14u);
        let raw = 0.2820948 * sh_rows[row + ch]
            + b1 * r1 + b2 * r2 + b3 * r3 + b4 * r4 + b5 * r5 + b6 * r6 + b7 * r7 + b8 * r8
            + b9 * r9 + b10 * r10 + b11 * r11 + b12 * r12 + b13 * r13 + b14 * r14 + b15 * r15
            + 0.5;
        var gc = g_col[ch];
        if (raw < 0.0) {
            gc = 0.0;
        }
        // This channel's coefficients are in registers now: overwrite them
        // with their gradient (stored back to grad_sh by the workgroup).
        sh_rows[row + ch] = gc * 0.2820948;
        let rb = row + 3u + ch * rest_pc;
        if (act >= 3u) {
            sh_rows[rb + 0u] = gc * b1;
            sh_rows[rb + 1u] = gc * b2;
            sh_rows[rb + 2u] = gc * b3;
        }
        if (act >= 8u) {
            sh_rows[rb + 3u] = gc * b4;
            sh_rows[rb + 4u] = gc * b5;
            sh_rows[rb + 5u] = gc * b6;
            sh_rows[rb + 6u] = gc * b7;
            sh_rows[rb + 7u] = gc * b8;
        }
        if (act >= 15u) {
            sh_rows[rb + 8u] = gc * b9;
            sh_rows[rb + 9u] = gc * b10;
            sh_rows[rb + 10u] = gc * b11;
            sh_rows[rb + 11u] = gc * b12;
            sh_rows[rb + 12u] = gc * b13;
            sh_rows[rb + 13u] = gc * b14;
            sh_rows[rb + 14u] = gc * b15;
        }
        // Bands past the active degree get zero gradient (their staged
        // coefficients would otherwise be written back as gradients).
        for (var k = act; k < rest_pc; k = k + 1u) {
            sh_rows[rb + k] = 0.0;
        }
        // d(colour)/d(dir) = sum_k coef_k * d(b_k)/d(dir).
        var dd = vec3<f32>(0.0, -c0, 0.0) * r1
            + vec3<f32>(0.0, 0.0, c0) * r2
            + vec3<f32>(-c0, 0.0, 0.0) * r3
            + vec3<f32>(c3 * y, c3 * x, 0.0) * r4
            + vec3<f32>(0.0, -c3 * z, -c3 * y) * r5
            + vec3<f32>(-2.0 * c5 * x, -2.0 * c5 * y, 4.0 * c5 * z) * r6
            + vec3<f32>(-c3 * z, 0.0, -c3 * x) * r7
            + vec3<f32>(2.0 * c2b * x, -2.0 * c2b * y, 0.0) * r8;
        dd = dd
            + vec3<f32>(-6.0 * c8 * x * y, -c8 * (3.0 * xx - 3.0 * yy), 0.0) * r9
            + vec3<f32>(c9 * y * z, c9 * x * z, c9 * x * y) * r10
            + vec3<f32>(2.0 * c10 * x * y, -c10 * (4.0 * zz - xx - 3.0 * yy), -8.0 * c10 * y * z) * r11
            + vec3<f32>(-6.0 * c11 * x * z, -6.0 * c11 * y * z, c11 * (6.0 * zz - 3.0 * xx - 3.0 * yy)) * r12
            + vec3<f32>(-c10 * (4.0 * zz - 3.0 * xx - yy), 2.0 * c10 * x * y, -8.0 * c10 * x * z) * r13
            + vec3<f32>(2.0 * c14 * x * z, -2.0 * c14 * y * z, c14 * (xx - yy)) * r14
            + vec3<f32>(-c8 * (3.0 * xx - 3.0 * yy), 6.0 * c8 * x * y, 0.0) * r15;
        gdir = gdir + gc * dd;
    }
    if (dn > 1e-6) {
        gmean = gmean + (gdir - dir * dot(dir, gdir)) / dn;
    }

    grad_transforms[base + 0u] = gmean.x;
    grad_transforms[base + 1u] = gmean.y;
    grad_transforms[base + 2u] = gmean.z;
    grad_transforms[base + 3u] = gq.x;
    grad_transforms[base + 4u] = gq.y;
    grad_transforms[base + 5u] = gq.z;
    grad_transforms[base + 6u] = gq.w;
    grad_transforms[base + 7u] = gls.x;
    grad_transforms[base + 8u] = gls.y;
    grad_transforms[base + 9u] = gls.z;
}
