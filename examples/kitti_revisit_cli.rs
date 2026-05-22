use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FrontendChoice {
    Classical,
    Deep,
    DeepMultiScale,
    Both,
}

impl FrontendChoice {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "classical" | "corner" => Ok(Self::Classical),
            "deep" | "hog" | "lightglue" => Ok(Self::Deep),
            "deep-ms" | "multiscale" | "deep-multi-scale" => Ok(Self::DeepMultiScale),
            "both" | "compare" => Ok(Self::Both),
            other => {
                Err(format!("--frontend must be classical|deep|deep-ms|both, got {other}").into())
            }
        }
    }

    pub(super) fn as_cli_label(&self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Deep => "deep",
            Self::DeepMultiScale => "deep-ms",
            Self::Both => "both",
        }
    }
}

#[derive(Debug)]
pub(super) struct CliArgs {
    pub(super) segment_a: PathBuf,
    pub(super) segment_b: PathBuf,
    pub(super) calib_a: PathBuf,
    pub(super) calib_b: PathBuf,
    pub(super) projection_label: String,
    pub(super) out_dir: Option<PathBuf>,
    pub(super) frontend: FrontendChoice,
    pub(super) max_features: usize,
    pub(super) min_matches: usize,
    pub(super) min_inliers: usize,
    pub(super) min_inlier_ratio: f64,
    pub(super) max_mean_sampson_error: f64,
}

pub(super) fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from<I, S>(args: I) -> Result<CliArgs, Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut segment_a: Option<PathBuf> = None;
    let mut segment_b: Option<PathBuf> = None;
    let mut calib_a: Option<PathBuf> = None;
    let mut calib_b: Option<PathBuf> = None;
    let mut projection_label = String::from("P0");
    let mut out_dir: Option<PathBuf> = None;
    let mut frontend = FrontendChoice::Classical;
    let mut max_features: usize = 400;
    let mut min_matches: usize = 30;
    let mut min_inliers: usize = 12;
    let mut min_inlier_ratio: f64 = 0.4;
    let mut max_mean_sampson_error: f64 = 5.0e-3;
    let mut iter = args.into_iter().map(Into::into);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--segment-a" => segment_a = iter.next().map(PathBuf::from),
            "--segment-b" => segment_b = iter.next().map(PathBuf::from),
            "--calib-a" => calib_a = iter.next().map(PathBuf::from),
            "--calib-b" => calib_b = iter.next().map(PathBuf::from),
            "--projection" => {
                projection_label = iter
                    .next()
                    .ok_or("--projection requires a label like P0/P1")?;
            }
            "--out-dir" => out_dir = iter.next().map(PathBuf::from),
            "--frontend" => {
                let value = iter
                    .next()
                    .ok_or("--frontend requires classical|deep|both")?;
                frontend = FrontendChoice::parse(&value)?;
            }
            "--max-features" => {
                let value = iter
                    .next()
                    .ok_or("--max-features requires a positive integer")?;
                max_features = value.parse::<usize>()?;
                if max_features == 0 {
                    return Err("--max-features must be > 0".into());
                }
            }
            "--min-matches" => {
                let value = iter
                    .next()
                    .ok_or("--min-matches requires a positive integer")?;
                min_matches = value.parse::<usize>()?;
                if min_matches == 0 {
                    return Err("--min-matches must be > 0".into());
                }
            }
            "--min-inliers" => {
                let value = iter
                    .next()
                    .ok_or("--min-inliers requires a positive integer")?;
                min_inliers = value.parse::<usize>()?;
                if min_inliers == 0 {
                    return Err("--min-inliers must be > 0".into());
                }
            }
            "--min-inlier-ratio" => {
                let value = iter
                    .next()
                    .ok_or("--min-inlier-ratio requires a number in [0, 1]")?;
                min_inlier_ratio = value.parse::<f64>()?;
                if !(0.0..=1.0).contains(&min_inlier_ratio) {
                    return Err("--min-inlier-ratio must be in [0, 1]".into());
                }
            }
            "--max-mean-sampson-error" => {
                let value = iter
                    .next()
                    .ok_or("--max-mean-sampson-error requires a positive number")?;
                max_mean_sampson_error = value.parse::<f64>()?;
                if max_mean_sampson_error <= 0.0 {
                    return Err("--max-mean-sampson-error must be > 0".into());
                }
            }
            other => return Err(format!("unrecognised flag {other}").into()),
        }
    }
    Ok(CliArgs {
        segment_a: segment_a.ok_or("--segment-a is required")?,
        segment_b: segment_b.ok_or("--segment-b is required")?,
        calib_a: calib_a.ok_or("--calib-a is required")?,
        calib_b: calib_b.ok_or("--calib-b is required")?,
        projection_label,
        out_dir,
        frontend,
        max_features,
        min_matches,
        min_inliers,
        min_inlier_ratio,
        max_mean_sampson_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_args() -> Vec<&'static str> {
        vec![
            "--segment-a",
            "a/image_0",
            "--segment-b",
            "b/image_0",
            "--calib-a",
            "a/calib.txt",
            "--calib-b",
            "b/calib.txt",
        ]
    }

    #[test]
    fn parses_defaults_and_required_paths() {
        let args = parse_args_from(required_args()).expect("valid args");

        assert_eq!(args.segment_a, PathBuf::from("a/image_0"));
        assert_eq!(args.segment_b, PathBuf::from("b/image_0"));
        assert_eq!(args.projection_label, "P0");
        assert_eq!(args.frontend, FrontendChoice::Classical);
        assert_eq!(args.max_features, 400);
        assert_eq!(args.min_matches, 30);
        assert_eq!(args.min_inliers, 12);
        assert_eq!(args.min_inlier_ratio, 0.4);
        assert_eq!(args.max_mean_sampson_error, 5.0e-3);
    }

    #[test]
    fn parses_frontend_aliases_and_thresholds() {
        let mut raw = required_args();
        raw.extend([
            "--frontend",
            "deep-multi-scale",
            "--projection",
            "P1",
            "--out-dir",
            "report",
            "--max-features",
            "200",
            "--min-matches",
            "31",
            "--min-inliers",
            "13",
            "--min-inlier-ratio",
            "0.45",
            "--max-mean-sampson-error",
            "0.006",
        ]);

        let args = parse_args_from(raw).expect("valid args");

        assert_eq!(args.frontend, FrontendChoice::DeepMultiScale);
        assert_eq!(args.frontend.as_cli_label(), "deep-ms");
        assert_eq!(args.projection_label, "P1");
        assert_eq!(args.out_dir, Some(PathBuf::from("report")));
        assert_eq!(args.max_features, 200);
        assert_eq!(args.min_matches, 31);
        assert_eq!(args.min_inliers, 13);
        assert_eq!(args.min_inlier_ratio, 0.45);
        assert_eq!(args.max_mean_sampson_error, 0.006);
    }

    #[test]
    fn rejects_zero_and_out_of_range_thresholds() {
        let mut zero_features = required_args();
        zero_features.extend(["--max-features", "0"]);
        assert!(parse_args_from(zero_features)
            .unwrap_err()
            .to_string()
            .contains("--max-features must be > 0"));

        let mut bad_ratio = required_args();
        bad_ratio.extend(["--min-inlier-ratio", "1.1"]);
        assert!(parse_args_from(bad_ratio)
            .unwrap_err()
            .to_string()
            .contains("--min-inlier-ratio must be in [0, 1]"));
    }

    #[test]
    fn rejects_missing_required_paths() {
        let err = parse_args_from(["--segment-a", "a/image_0"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--segment-b is required"));
    }
}
