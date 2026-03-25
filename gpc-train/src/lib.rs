//! Training orchestration for GPC models.
//!
//! Handles the two-phase training pipeline:
//! - Phase 1: Single-step world model warmup
//! - Phase 2: Multi-step world model training
//! - Diffusion policy training (independent)

pub mod data;
pub mod policy_trainer;
pub mod world_model_trainer;

pub use data::{GpcDataset, GpcDatasetConfig, GpcDatasetSplit};
pub use policy_trainer::{PolicyTrainer, PolicyTrainingResult, PolicyValidationSummary};
pub use world_model_trainer::{
    WorldModelTrainer, WorldModelTrainingResult, WorldModelValidationSummary,
};

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Mutex, MutexGuard};

    static TRAINING_TEST_LOCK: Mutex<()> = Mutex::new(());

    pub(crate) fn training_test_guard() -> MutexGuard<'static, ()> {
        TRAINING_TEST_LOCK
            .lock()
            .expect("training test mutex should not be poisoned")
    }
}
