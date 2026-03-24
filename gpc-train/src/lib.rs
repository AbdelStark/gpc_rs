//! Training orchestration for GPC models.
//!
//! Handles the two-phase training pipeline:
//! - Phase 1: Single-step world model warmup
//! - Phase 2: Multi-step world model training
//! - Diffusion policy training (independent)

pub mod data;
pub mod policy_trainer;
pub mod world_model_trainer;

pub use data::{GpcDataset, GpcDatasetConfig};
pub use policy_trainer::{PolicyTrainer, PolicyTrainingResult};
pub use world_model_trainer::{WorldModelTrainer, WorldModelTrainingResult};
