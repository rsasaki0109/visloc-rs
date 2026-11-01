// Lens models on the device: gsplat-core's lens.rs (the distorted normalized
// coordinates of a camera-space point and their 2x3 Jacobian), templated on
// float and on Dual3, a value with its gradient with respect to the
// camera-space point. Evaluated on Dual3, the Jacobian's entries carry their
// own derivatives, which project_bwd_kernel chains the gradient through.
// Included by gsplat_cuda.cu after GscCamera.

#pragma once

struct Dual3 {
    float v;
    float d[3];
};

__device__ inline Dual3 dual_var(float v, int i) {
    Dual3 r = {v, {0.f, 0.f, 0.f}};
    r.d[i] = 1.f;
    return r;
}
__device__ inline Dual3 operator+(Dual3 a, Dual3 b) {
    return {a.v + b.v, {a.d[0] + b.d[0], a.d[1] + b.d[1], a.d[2] + b.d[2]}};
}
__device__ inline Dual3 operator-(Dual3 a, Dual3 b) {
    return {a.v - b.v, {a.d[0] - b.d[0], a.d[1] - b.d[1], a.d[2] - b.d[2]}};
}
__device__ inline Dual3 operator-(Dual3 a) { return {-a.v, {-a.d[0], -a.d[1], -a.d[2]}}; }
__device__ inline Dual3 operator*(Dual3 a, Dual3 b) {
    return {a.v * b.v,
            {a.d[0] * b.v + a.v * b.d[0], a.d[1] * b.v + a.v * b.d[1], a.d[2] * b.v + a.v * b.d[2]}};
}
__device__ inline Dual3 operator/(Dual3 a, Dual3 b) {
    const float inv = 1.f / b.v;
    const float q = a.v * inv;
    return {q, {(a.d[0] - q * b.d[0]) * inv, (a.d[1] - q * b.d[1]) * inv, (a.d[2] - q * b.d[2]) * inv}};
}
__device__ inline Dual3 operator+(Dual3 a, float s) { return {a.v + s, {a.d[0], a.d[1], a.d[2]}}; }
__device__ inline Dual3 operator+(float s, Dual3 a) { return a + s; }
__device__ inline Dual3 operator-(Dual3 a, float s) { return {a.v - s, {a.d[0], a.d[1], a.d[2]}}; }
__device__ inline Dual3 operator-(float s, Dual3 a) { return {s - a.v, {-a.d[0], -a.d[1], -a.d[2]}}; }
__device__ inline Dual3 operator*(Dual3 a, float s) { return {a.v * s, {a.d[0] * s, a.d[1] * s, a.d[2] * s}}; }
__device__ inline Dual3 operator*(float s, Dual3 a) { return a * s; }
__device__ inline Dual3 operator/(float s, Dual3 a) {
    const float q = s / a.v;
    const float k = -q / a.v;
    return {q, {a.d[0] * k, a.d[1] * k, a.d[2] * k}};
}
__device__ inline Dual3 dual_sqrt(Dual3 a) {
    const float s = sqrtf(a.v);
    const float k = 0.5f / s;
    return {s, {a.d[0] * k, a.d[1] * k, a.d[2] * k}};
}
// The angle of (x, y).
__device__ inline Dual3 dual_atan2(Dual3 y, Dual3 x) {
    const float inv = 1.f / (x.v * x.v + y.v * y.v);
    return {atan2f(y.v, x.v),
            {(x.v * y.d[0] - y.v * x.d[0]) * inv, (x.v * y.d[1] - y.v * x.d[1]) * inv,
             (x.v * y.d[2] - y.v * x.d[2]) * inv}};
}

__device__ inline float lens_value(float x) { return x; }
__device__ inline float lens_value(Dual3 x) { return x.v; }
__device__ inline float lens_sqrt(float x) { return sqrtf(x); }
__device__ inline Dual3 lens_sqrt(Dual3 x) { return dual_sqrt(x); }
__device__ inline float lens_atan2(float y, float x) { return atan2f(y, x); }
__device__ inline Dual3 lens_atan2(Dual3 y, Dual3 x) { return dual_atan2(y, x); }

// GscCamera::lens_kind.
constexpr uint32_t kLensPinhole = 0;      // undistorted pinhole: the original path
constexpr uint32_t kLensDistorted = 1;    // pinhole with distortion
constexpr uint32_t kLensEquidistant = 2;  // equidistant (fisheye), with or without distortion

// Below this (r / z)^2 the equidistant scale theta / r and its lateral
// derivative come from their series in r / z (lens.rs's SERIES_S2): the
// closed forms cancel catastrophically near the axis in float.
constexpr float kLensSeriesS2 = 1e-2f;

// The distorted normalized coordinates `m` of camera-space point `p` (before
// the focal lengths) and their Jacobian `J` (rows: m_x, m_y; columns: x, y,
// z), for a camera with lens_kind != kLensPinhole and `p` within max_theta
// (lens_extended past it).
template <class T>
__device__ void lens_coords(const GscCamera& c, const T p[3], T m[2], T J[2][3]) {
    const T x = p[0], y = p[1], z = p[2];
    T n[2], nj[2][3];
    if (c.lens_kind == kLensEquidistant) {
        const T r2 = x * x + y * y;
        const T rho2 = r2 + z * z;
        T scale, lateral;
        const float zv = lens_value(z);
        if (zv > 0.f && lens_value(r2) < kLensSeriesS2 * zv * zv) {
            // atan(s) / s and (1 / (1 + s^2) - atan(s) / s) / s^2 in s = r / z.
            const T iz = 1.f / z;
            const T s2 = r2 * iz * iz;
            scale = iz * (1.f + s2 * (-1.f / 3.f + s2 * (1.f / 5.f + s2 * (-1.f / 7.f))));
            lateral = iz * iz * iz * (-2.f / 3.f + s2 * (4.f / 5.f + s2 * (-6.f / 7.f + s2 * (8.f / 9.f))));
        } else {
            const T r = lens_sqrt(r2);
            scale = lens_atan2(r, z) / r;
            lateral = (z / rho2 - scale) / r2;
        }
        const T ds_dz = -1.f / rho2;
        const T xy = x * y * lateral;
        n[0] = scale * x;
        n[1] = scale * y;
        nj[0][0] = scale + x * x * lateral;
        nj[0][1] = xy;
        nj[0][2] = x * ds_dz;
        nj[1][0] = xy;
        nj[1][1] = scale + y * y * lateral;
        nj[1][2] = y * ds_dz;
    } else {
        const T iz = 1.f / z;
        const T iz2 = iz * iz;
        n[0] = x * iz;
        n[1] = y * iz;
        nj[0][0] = iz;
        nj[0][1] = 0.f * iz;
        nj[0][2] = -(x * iz2);
        nj[1][0] = 0.f * iz;
        nj[1][1] = iz;
        nj[1][2] = -(y * iz2);
    }
    const float* k = c.lens;
    bool distorted = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) distorted |= k[i] != 0.f;
    if (!distorted) {
        m[0] = n[0];
        m[1] = n[1];
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int col = 0; col < 3; ++col) J[r][col] = nj[r][col];
        return;
    }
    // COLMAP's distortion (lens.rs's `distort`), k = [k1 k2 k3 k4 p1 p2 sx1 sy1].
    const float k1 = k[0], k2 = k[1], k3 = k[2], k4 = k[3], p1 = k[4], p2 = k[5], sx1 = k[6], sy1 = k[7];
    const T u = n[0], v = n[1];
    const T t2 = u * u + v * v;
    const T radial = t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)));
    const T g = k1 + t2 * (2.f * k2 + t2 * (3.f * k3 + t2 * (4.f * k4)));
    const T uu = u * u, vv = v * v, uv = u * v;
    m[0] = u + u * radial + 2.f * p1 * uv + p2 * (t2 + 2.f * uu) + sx1 * t2;
    m[1] = v + v * radial + 2.f * p2 * uv + p1 * (t2 + 2.f * vv) + sy1 * t2;
    const T cross = 2.f * uv * g + 2.f * p1 * u + 2.f * p2 * v;
    const T d00 = 1.f + radial + 2.f * uu * g + 2.f * p1 * v + 6.f * p2 * u + 2.f * sx1 * u;
    const T d01 = cross + 2.f * sx1 * v;
    const T d10 = cross + 2.f * sy1 * u;
    const T d11 = 1.f + radial + 2.f * vv * g + 2.f * p2 * u + 6.f * p1 * v + 2.f * sy1 * v;
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        J[0][col] = d00 * nj[0][col] + d01 * nj[1][col];
        J[1][col] = d10 * nj[0][col] + d11 * nj[1][col];
    }
}

// The angle between the optical axis and the ray to camera-space (x, y, z).
__device__ inline float lens_ray_angle(float x, float y, float z) { return atan2f(sqrtf(x * x + y * y), z); }

// lens_coords for any camera-space point p (lens.rs's extended_lens_coords):
// within max_theta the lens itself; past it, the lens's first-order
// extension at q, p turned towards the optical axis onto the max_theta cone
// (same distance and azimuth): m(q) + J(q) (p - q). Writes m, the Jacobian at
// q, and q.
template <class T>
__device__ void lens_extended(const GscCamera& c, const T p[3], T m[2], T J[2][3], T q[3]) {
    if (lens_ray_angle(lens_value(p[0]), lens_value(p[1]), lens_value(p[2])) <= c.max_theta) {
#pragma unroll
        for (int k = 0; k < 3; ++k) q[k] = p[k];
        lens_coords(c, p, m, J);
        return;
    }
    const T r2 = p[0] * p[0] + p[1] * p[1];
    const T rho = lens_sqrt(r2 + p[2] * p[2]);
    float sin_t, cos_t;
    sincosf(c.max_theta, &sin_t, &cos_t);
    const T k = rho * sin_t / lens_sqrt(r2);
    q[0] = p[0] * k;
    q[1] = p[1] * k;
    q[2] = rho * cos_t;
    T mq[2];
    lens_coords(c, q, mq, J);
    const T d[3] = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
#pragma unroll
    for (int r = 0; r < 2; ++r) m[r] = mq[r] + J[r][0] * d[0] + J[r][1] * d[1] + J[r][2] * d[2];
}
