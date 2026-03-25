//! Model interoperability and checkpoint loading.
//!
//! Supports:
//! - Saving and loading Burn model checkpoints
//! - ONNX model inspection via Tract
//! - Configuration serialization

pub mod checkpoint;
pub mod onnx;

pub use checkpoint::{
    CheckpointArtifact, CheckpointConfigSummary, CheckpointFormat, CheckpointInspectionReport,
    CheckpointKind, CheckpointMetadata, checkpoint_path, convert_policy_checkpoint,
    convert_world_model_checkpoint, inspect_checkpoint_artifact, load_checkpoint_bytes,
    load_metadata, load_policy_checkpoint, load_world_model_checkpoint, metadata_path_for,
    save_checkpoint, save_policy_checkpoint, save_world_model_checkpoint,
    verify_checkpoint_artifact,
};
pub use onnx::OnnxInspector;
