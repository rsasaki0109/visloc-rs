use std::fs;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;

use nalgebra::Point2;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct TwoViewFeatureMatch {
    pub previous_index: usize,
    pub current_index: usize,
    pub previous_xy: Point2<f64>,
    pub current_xy: Point2<f64>,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TwoViewMatchSet {
    matches: Vec<TwoViewFeatureMatch>,
}

impl TwoViewMatchSet {
    pub fn new(matches: Vec<TwoViewFeatureMatch>) -> Self {
        Self { matches }
    }

    pub fn matches(&self) -> &[TwoViewFeatureMatch] {
        &self.matches
    }

    pub fn into_matches(self) -> Vec<TwoViewFeatureMatch> {
        self.matches
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    pub fn matched_previous_keypoints(&self) -> Vec<Point2<f64>> {
        self.matches
            .iter()
            .map(|feature_match| feature_match.previous_xy)
            .collect()
    }

    pub fn matched_current_keypoints(&self) -> Vec<Point2<f64>> {
        self.matches
            .iter()
            .map(|feature_match| feature_match.current_xy)
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum TwoViewMatchError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integer parse error: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("float parse error: {0}")]
    ParseFloat(#[from] ParseFloatError),
    #[error("invalid two-view match line: {line}")]
    InvalidLine { line: String },
}

pub fn read_two_view_matches_txt(
    path: impl AsRef<Path>,
) -> Result<TwoViewMatchSet, TwoViewMatchError> {
    parse_two_view_matches_txt(&fs::read_to_string(path)?)
}

pub fn parse_two_view_matches_txt(contents: &str) -> Result<TwoViewMatchSet, TwoViewMatchError> {
    let mut matches = Vec::new();

    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 6 && tokens.len() != 7 {
            return Err(TwoViewMatchError::InvalidLine {
                line: line.to_owned(),
            });
        }

        matches.push(TwoViewFeatureMatch {
            previous_index: tokens[0].parse()?,
            current_index: tokens[1].parse()?,
            previous_xy: Point2::new(tokens[2].parse()?, tokens[3].parse()?),
            current_xy: Point2::new(tokens[4].parse()?, tokens[5].parse()?),
            score: tokens.get(6).map(|value| value.parse()).transpose()?,
        });
    }

    Ok(TwoViewMatchSet::new(matches))
}
