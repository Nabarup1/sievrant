//! Conformal calibration state, coverage monitoring and query history.

pub mod store;

pub use store::{
    split_half_validation, CalibrationRecord, CorpusCalibration, FeedbackOutcome, GapEntry,
    GateDecision, HoldoutResult, LoggedCandidate, QueryRecord,
};
