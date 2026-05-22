use super::kitti_revisit_format::html_escape;

pub(super) struct ReportPageData<'a> {
    pub(super) segment_a_range: (u64, u64),
    pub(super) segment_b_range: (u64, u64),
    pub(super) frontend_label: &'a str,
    pub(super) report_count: usize,
    pub(super) max_features: usize,
    pub(super) min_gap: u64,
    pub(super) min_matches: usize,
    pub(super) min_inliers: usize,
    pub(super) min_ratio: f64,
    pub(super) max_sampson: f64,
    pub(super) command: &'a str,
    pub(super) cards: &'a str,
    pub(super) tables: &'a str,
}

pub(super) fn render_report_page(data: ReportPageData<'_>) -> String {
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>visloc-rs KITTI 00 revisit loop report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --accent: #0f766e;
      --accent-2: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 system-ui, -apple-system, Segoe UI, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 40px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; line-height: 1.15; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; }}
    .lede {{ margin: 0 0 18px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 10px 0 14px; }}
    .metric {{ border-left: 3px solid var(--accent); padding-left: 9px; }}
    .metric b {{ display: block; font-size: 18px; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }}
    .overlay {{ margin-top: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; border: 1px solid var(--line); background: #e2e8f0; }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #e2e8f0; font-size: 12px; color: #334155; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    pre {{ overflow-x: auto; background: #0f172a; color: #e2e8f0; border-radius: 8px; padding: 12px; }}
    .note {{ color: var(--muted); }}
    .empty {{ color: var(--accent-2); font-weight: 600; }}
  </style>
</head>
<body>
<main>
  <h1>KITTI 00 Revisit Loop Report</h1>
  <p class="lede">Real-image appearance loop scanning over start frames {a_min}..{a_max} and revisit frames {b_min}..{b_max}. Rejected pair diagnostics are summarized by thresholds here; the current scanner returns accepted, geometrically verified candidates only.</p>

  <section class="grid">
    <div class="card">
      <h2>Run</h2>
      <div class="metrics">
        <div class="metric"><b>{frontend}</b><span>frontend selection</span></div>
        <div class="metric"><b>{report_count}</b><span>frontend reports</span></div>
        <div class="metric"><b>{max_features}</b><span>feature cap per frame</span></div>
        <div class="metric"><b>{min_gap}</b><span>minimum frame-id gap</span></div>
        <div class="metric"><b>{min_matches}</b><span>minimum raw matches</span></div>
      </div>
      <p class="note">Verifier thresholds: min inliers {min_inliers}, min inlier ratio {min_ratio:.2}, max mean Sampson {max_sampson:.5}.</p>
    </div>
    <div class="card">
      <h2>Reproduce</h2>
      <pre>{command}</pre>
    </div>
  </section>

  <h2>Strongest Candidates</h2>
  <section class="grid">
    {cards}
  </section>

  <h2>Accepted Candidate Tables</h2>
  {tables}
</main>
</body>
</html>
"#,
        a_min = data.segment_a_range.0,
        a_max = data.segment_a_range.1,
        b_min = data.segment_b_range.0,
        b_max = data.segment_b_range.1,
        frontend = html_escape(data.frontend_label),
        report_count = data.report_count,
        max_features = data.max_features,
        min_gap = data.min_gap,
        min_matches = data.min_matches,
        min_inliers = data.min_inliers,
        min_ratio = data.min_ratio,
        max_sampson = data.max_sampson,
        command = html_escape(data.command),
        cards = data.cards,
        tables = data.tables,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_report_page_escapes_run_fields_but_keeps_report_fragments() {
        let html = render_report_page(ReportPageData {
            segment_a_range: (0, 49),
            segment_b_range: (4500, 4529),
            frontend_label: "deep <unsafe>",
            report_count: 1,
            max_features: 200,
            min_gap: 50,
            min_matches: 30,
            min_inliers: 12,
            min_ratio: 0.4,
            max_sampson: 0.005,
            command: "cargo <run>",
            cards: "<article>card</article>",
            tables: "<table></table>",
        });

        assert!(html.contains("deep &lt;unsafe&gt;"));
        assert!(html.contains("cargo &lt;run&gt;"));
        assert!(html.contains("<article>card</article>"));
        assert!(html.contains("start frames 0..49"));
        assert!(html.contains("revisit frames 4500..4529"));
    }
}
