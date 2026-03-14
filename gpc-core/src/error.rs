//! Error types for the GPC framework.

use thiserror::Error;

/// Result type alias using [`GpcError`].
pub type Result<T> = std::result::Result<T, GpcError>;

/// Top-level error type for the GPC framework.
#[derive(Debug, Error)]
pub enum GpcError {
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Model error: {0}")]
    Model(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}
