//! Model interoperability and checkpoint loading.
//!
//! Supports:
//! - Saving and loading Burn model checkpoints
//! - ONNX model inspection via Tract
//! - Configuration serialization

pub mod checkpoint;
pub mod onnx;

pub use checkpoint::{
    CheckpointArtifact, CheckpointFormat, CheckpointKind, CheckpointMetadata, checkpoint_path,
    convert_policy_checkpoint, convert_world_model_checkpoint, load_checkpoint_bytes,
    load_metadata, load_policy_checkpoint, load_world_model_checkpoint, metadata_path_for,
    save_checkpoint, save_policy_checkpoint, save_world_model_checkpoint,
};
pub use onnx::OnnxInspector;
