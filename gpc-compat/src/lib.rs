//! Model interoperability and checkpoint loading.
//!
//! Supports:
//! - Saving and loading Burn model checkpoints
//! - ONNX model inspection via Tract
//! - Configuration serialization

pub mod checkpoint;
pub mod onnx;

pub use checkpoint::{CheckpointMetadata, load_checkpoint_bytes, save_checkpoint};
pub use onnx::OnnxInspector;
