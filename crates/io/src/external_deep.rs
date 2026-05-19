use std::fs;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;

use nalgebra::Point2;
use thiserror::Error;
use visloc_vision::features::{FeatureSet, FeatureSetError};
use visloc_vision::matching::DescriptorMatch;

#[derive(Debug, Clone, PartialEq)]
pub struct ExternalDeepFeature {
    pub xy: Point2<f64>,
    pub score: f32,
    pub descriptor: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalDeepFeatureSet {
    features: Vec<ExternalDeepFeature>,
}

impl ExternalDeepFeatureSet {
    pub fn new(features: Vec<ExternalDeepFeature>) -> Self {
        Self { features }
    }

    pub fn features(&self) -> &[ExternalDeepFeature] {
        &self.features
    }

    pub fn into_features(self) -> Vec<ExternalDeepFeature> {
        self.features
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn keypoints(&self) -> Vec<Point2<f64>> {
        self.features.iter().map(|feature| feature.xy).collect()
    }

    pub fn scores(&self) -> Vec<f32> {
        self.features.iter().map(|feature| feature.score).collect()
    }

    pub fn descriptors(&self) -> Vec<Vec<f32>> {
        self.features
            .iter()
            .map(|feature| feature.descriptor.clone())
            .collect()
    }

    pub fn to_feature_set(&self) -> Result<FeatureSet, FeatureSetError> {
        FeatureSet::new(self.keypoints(), self.descriptors())
    }

    pub fn into_feature_set(self) -> Result<FeatureSet, FeatureSetError> {
        let mut keypoints = Vec::with_capacity(self.features.len());
        let mut descriptors = Vec::with_capacity(self.features.len());
        for feature in self.features {
            keypoints.push(feature.xy);
            descriptors.push(feature.descriptor);
        }
        FeatureSet::new(keypoints, descriptors)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternalDeepMatch {
    pub query_index: usize,
    pub train_index: usize,
    pub confidence: f32,
    pub distance: Option<f32>,
}

impl ExternalDeepMatch {
    pub fn to_descriptor_match(&self) -> DescriptorMatch {
        DescriptorMatch {
            query_index: self.query_index,
            train_index: self.train_index,
            distance: self.distance.unwrap_or(1.0 - self.confidence),
            second_best_distance: None,
            ratio: None,
            confidence: Some(self.confidence),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalDeepMatchSet {
    matches: Vec<ExternalDeepMatch>,
}

impl ExternalDeepMatchSet {
    pub fn new(matches: Vec<ExternalDeepMatch>) -> Self {
        Self { matches }
    }

    pub fn matches(&self) -> &[ExternalDeepMatch] {
        &self.matches
    }

    pub fn into_matches(self) -> Vec<ExternalDeepMatch> {
        self.matches
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    pub fn to_descriptor_matches(&self) -> Vec<DescriptorMatch> {
        self.matches
            .iter()
            .map(ExternalDeepMatch::to_descriptor_match)
            .collect()
    }

    pub fn into_descriptor_matches(self) -> Vec<DescriptorMatch> {
        self.matches
            .into_iter()
            .map(|deep_match| deep_match.to_descriptor_match())
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum ExternalDeepError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integer parse error: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("float parse error: {0}")]
    ParseFloat(#[from] ParseFloatError),
    #[error("feature set error: {0}")]
    FeatureSet(#[from] FeatureSetError),
    #[error("invalid external deep feature line: {line}")]
    InvalidFeatureLine { line: String },
    #[error("invalid external deep match line: {line}")]
    InvalidMatchLine { line: String },
}

pub fn read_external_deep_features_txt(
    path: impl AsRef<Path>,
) -> Result<ExternalDeepFeatureSet, ExternalDeepError> {
    parse_external_deep_features_txt(&fs::read_to_string(path)?)
}

pub fn parse_external_deep_features_txt(
    contents: &str,
) -> Result<ExternalDeepFeatureSet, ExternalDeepError> {
    let mut features = Vec::new();

    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 4 {
            return Err(ExternalDeepError::InvalidFeatureLine {
                line: line.to_owned(),
            });
        }

        let xy = Point2::new(
            parse_finite_f64(tokens[0], line, true)?,
            parse_finite_f64(tokens[1], line, true)?,
        );
        let score = parse_unit_confidence(tokens[2], line, true)?;
        let descriptor = tokens[3..]
            .iter()
            .map(|value| parse_finite_f32(value, line, true))
            .collect::<Result<Vec<_>, _>>()?;

        features.push(ExternalDeepFeature {
            xy,
            score,
            descriptor,
        });
    }

    let set = ExternalDeepFeatureSet::new(features);
    set.to_feature_set()?;
    Ok(set)
}

pub fn read_external_deep_matches_txt(
    path: impl AsRef<Path>,
) -> Result<ExternalDeepMatchSet, ExternalDeepError> {
    parse_external_deep_matches_txt(&fs::read_to_string(path)?)
}

pub fn parse_external_deep_matches_txt(
    contents: &str,
) -> Result<ExternalDeepMatchSet, ExternalDeepError> {
    let mut matches = Vec::new();

    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 3 && tokens.len() != 4 {
            return Err(ExternalDeepError::InvalidMatchLine {
                line: line.to_owned(),
            });
        }

        matches.push(ExternalDeepMatch {
            query_index: tokens[0].parse()?,
            train_index: tokens[1].parse()?,
            confidence: parse_unit_confidence(tokens[2], line, false)?,
            distance: tokens
                .get(3)
                .map(|value| parse_finite_f32(value, line, false))
                .transpose()?,
        });
    }

    Ok(ExternalDeepMatchSet::new(matches))
}

fn parse_unit_confidence(
    value: &str,
    line: &str,
    feature_line: bool,
) -> Result<f32, ExternalDeepError> {
    let confidence = parse_finite_f32(value, line, feature_line)?;
    if !(0.0..=1.0).contains(&confidence) {
        return invalid_line(line, feature_line);
    }
    Ok(confidence)
}

fn parse_finite_f32(value: &str, line: &str, feature_line: bool) -> Result<f32, ExternalDeepError> {
    let parsed = value.parse::<f32>()?;
    if !parsed.is_finite() {
        return invalid_line(line, feature_line);
    }
    Ok(parsed)
}

fn parse_finite_f64(value: &str, line: &str, feature_line: bool) -> Result<f64, ExternalDeepError> {
    let parsed = value.parse::<f64>()?;
    if !parsed.is_finite() {
        return invalid_line(line, feature_line);
    }
    Ok(parsed)
}

fn invalid_line<T>(line: &str, feature_line: bool) -> Result<T, ExternalDeepError> {
    if feature_line {
        Err(ExternalDeepError::InvalidFeatureLine {
            line: line.to_owned(),
        })
    } else {
        Err(ExternalDeepError::InvalidMatchLine {
            line: line.to_owned(),
        })
    }
}
