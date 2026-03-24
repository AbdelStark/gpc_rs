//! World model training orchestration (Phase 1 + Phase 2).

use burn::optim::{AdamWConfig, Optimizer};
use burn::prelude::*;

use gpc_core::config::{TrainingConfig, WorldModelConfig};
use gpc_core::tensor_utils;

use crate::data::{GpcDataset, WorldModelBatch, WorldModelBatcher};

/// Result of a completed world model training run.
pub struct WorldModelTrainingResult<B: burn::tensor::backend::Backend> {
    /// Trained world model.
    pub model: gpc_world::StateWorldModel<B>,
    /// Final epoch that ran, if any.
    pub final_epoch: Option<usize>,
    /// Final averaged loss for the last epoch, if any.
    pub final_loss: Option<f32>,
    /// Average loss for each completed epoch.
    pub epoch_losses: Vec<f32>,
}

/// World model trainer handling both training phases.
pub struct WorldModelTrainer {
    training_config: TrainingConfig,
    world_model_config: WorldModelConfig,
}

impl WorldModelTrainer {
    /// Create a new trainer.
    pub fn new(training_config: TrainingConfig, world_model_config: WorldModelConfig) -> Self {
        Self {
            training_config,
            world_model_config,
        }
    }

    /// Train Phase 1: Single-step prediction warmup.
    ///
    /// Trains the world model to predict single-step state transitions.
    pub fn train_phase1<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        device: &B::Device,
    ) -> gpc_world::StateWorldModel<B> {
        self.train_phase1_with_summary::<B>(dataset, device).model
    }

    /// Train Phase 1: Single-step prediction warmup and return final metadata.
    ///
    /// Trains the world model to predict single-step state transitions.
    pub fn train_phase1_with_summary<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        device: &B::Device,
    ) -> WorldModelTrainingResult<B> {
        use gpc_world::world_model::StateWorldModelConfig;

        tracing::info!("Starting Phase 1 training: single-step prediction");

        let model_config = StateWorldModelConfig {
            state_dim: self.world_model_config.state_dim,
            action_dim: self.world_model_config.action_dim,
            hidden_dim: self.world_model_config.hidden_dim,
            num_layers: self.world_model_config.num_layers,
        };

        let mut model = model_config.init::<B>(device);
        let mut final_epoch = None;
        let mut final_loss = None;
        let mut epoch_losses = Vec::with_capacity(self.training_config.num_epochs);
        let optimizer_config =
            AdamWConfig::new().with_weight_decay(self.training_config.weight_decay as f32);

        let mut optim = optimizer_config.init();

        let samples = dataset.world_model_samples();
        if samples.is_empty() {
            tracing::warn!("No world model training samples available for Phase 1");
            return WorldModelTrainingResult {
                model,
                final_epoch,
                final_loss,
                epoch_losses,
            };
        }

        let batcher = WorldModelBatcher::<B>::new(device.clone());
        let batch_size = self.training_config.batch_size.min(samples.len());

        for epoch in 0..self.training_config.num_epochs {
            let mut epoch_loss = 0.0_f32;
            let mut num_batches = 0;

            // Simple batching
            for chunk in samples.chunks(batch_size) {
                let batch: WorldModelBatch<B> = burn::data::dataloader::batcher::Batcher::batch(
                    &batcher,
                    chunk.to_vec(),
                    device,
                );

                // Forward pass
                let predicted_delta = model.predict_delta(&batch.states, &batch.actions);
                let target_delta = batch.next_states.clone() - batch.states.clone();

                // MSE loss
                let loss = tensor_utils::mse_loss(&predicted_delta, &target_delta);
                let loss_val: f32 = loss.clone().into_scalar().elem();

                // Backward pass
                let grads = loss.backward();
                let grads = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(self.training_config.learning_rate, model, grads);

                epoch_loss += loss_val;
                num_batches += 1;
            }

            let avg_loss = epoch_loss / num_batches as f32;
            final_epoch = Some(epoch + 1);
            final_loss = Some(avg_loss);
            epoch_losses.push(avg_loss);

            if epoch % self.training_config.log_every == 0 {
                tracing::info!(
                    "Phase 1 | Epoch {}/{} | Loss: {:.6}",
                    epoch + 1,
                    self.training_config.num_epochs,
                    avg_loss
                );
            }
        }

        tracing::info!("Phase 1 training complete");
        WorldModelTrainingResult {
            model,
            final_epoch,
            final_loss,
            epoch_losses,
        }
    }

    /// Train Phase 2: Multi-step rollout training.
    ///
    /// Fine-tunes the world model using multi-step trajectory prediction
    /// with joint supervision at each timestep.
    pub fn train_phase2<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        model: gpc_world::StateWorldModel<B>,
        horizon: usize,
        device: &B::Device,
    ) -> gpc_world::StateWorldModel<B> {
        self.train_phase2_with_summary::<B>(dataset, model, horizon, device)
            .model
    }

    /// Train Phase 2: Multi-step rollout training and return final metadata.
    ///
    /// Fine-tunes the world model using multi-step trajectory prediction
    /// with joint supervision at each timestep.
    pub fn train_phase2_with_summary<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        mut model: gpc_world::StateWorldModel<B>,
        horizon: usize,
        device: &B::Device,
    ) -> WorldModelTrainingResult<B> {
        tracing::info!(
            "Starting Phase 2 training: multi-step rollout (horizon={})",
            horizon
        );

        let mut final_epoch = None;
        let mut final_loss = None;
        let mut epoch_losses = Vec::with_capacity(self.training_config.num_epochs);
        let optimizer_config =
            AdamWConfig::new().with_weight_decay(self.training_config.weight_decay as f32);
        let mut optim = optimizer_config.init();

        let sequences = dataset.world_model_sequences(horizon);
        if sequences.is_empty() {
            tracing::warn!("No sequences of sufficient length for Phase 2");
            return WorldModelTrainingResult {
                model,
                final_epoch,
                final_loss,
                epoch_losses,
            };
        }

        let batch_size = self.training_config.batch_size.min(sequences.len());
        let state_dim = self.world_model_config.state_dim;
        let action_dim = self.world_model_config.action_dim;

        for epoch in 0..self.training_config.num_epochs {
            let mut epoch_loss = 0.0_f32;
            let mut num_batches = 0;

            for chunk in sequences.chunks(batch_size) {
                let bs = chunk.len();

                // Build tensors from sequences
                let init_flat: Vec<f32> = chunk
                    .iter()
                    .flat_map(|(init, _, _)| init.iter().copied())
                    .collect();
                let initial_states =
                    Tensor::<B, 2>::from_data(TensorData::new(init_flat, [bs, state_dim]), device);

                let actions_flat: Vec<f32> = chunk
                    .iter()
                    .flat_map(|(_, acts, _)| acts.iter().flat_map(|a| a.iter().copied()))
                    .collect();
                let actions = Tensor::<B, 3>::from_data(
                    TensorData::new(actions_flat, [bs, horizon, action_dim]),
                    device,
                );

                let targets_flat: Vec<f32> = chunk
                    .iter()
                    .flat_map(|(_, _, tgts)| tgts.iter().flat_map(|t| t.iter().copied()))
                    .collect();
                let target_states = Tensor::<B, 3>::from_data(
                    TensorData::new(targets_flat, [bs, horizon, state_dim]),
                    device,
                );

                // Multi-step rollout
                let mut current_state = initial_states;
                let mut total_loss = Tensor::<B, 1>::zeros([1], device);

                for t in 0..horizon {
                    let action_t = actions
                        .clone()
                        .slice([0..bs, t..t + 1, 0..action_dim])
                        .reshape([bs, action_dim]);
                    let target_t = target_states
                        .clone()
                        .slice([0..bs, t..t + 1, 0..state_dim])
                        .reshape([bs, state_dim]);

                    let next_state = model.predict_step(&current_state, &action_t);
                    let step_loss = tensor_utils::mse_loss(&next_state, &target_t);
                    total_loss = total_loss + step_loss;

                    current_state = next_state;
                }

                let avg_loss = total_loss / (horizon as f32);
                let loss_val: f32 = avg_loss.clone().into_scalar().elem();

                let grads = avg_loss.backward();
                let grads = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(self.training_config.learning_rate, model, grads);

                epoch_loss += loss_val;
                num_batches += 1;
            }

            let avg_loss = epoch_loss / num_batches as f32;
            final_epoch = Some(epoch + 1);
            final_loss = Some(avg_loss);
            epoch_losses.push(avg_loss);

            if epoch % self.training_config.log_every == 0 {
                tracing::info!(
                    "Phase 2 | Epoch {}/{} | Loss: {:.6}",
                    epoch + 1,
                    self.training_config.num_epochs,
                    avg_loss
                );
            }
        }

        tracing::info!("Phase 2 training complete");
        WorldModelTrainingResult {
            model,
            final_epoch,
            final_loss,
            epoch_losses,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GpcDatasetConfig;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::NdArray;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_phase1_trains_without_error() {
        let device = <TestBackend as Backend>::Device::default();

        let dataset_config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(dataset_config, 3, 20, 42);

        let training_config = TrainingConfig {
            num_epochs: 2,
            batch_size: 16,
            learning_rate: 1e-3,
            log_every: 1,
            ..Default::default()
        };

        let world_model_config = WorldModelConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_dim: 16,
            num_layers: 1,
            ..Default::default()
        };

        let trainer = WorldModelTrainer::new(training_config, world_model_config);
        let result = trainer.train_phase1_with_summary::<TestBackend>(&dataset, &device);
        assert_eq!(result.final_epoch, Some(2));
        assert_eq!(result.epoch_losses.len(), 2);
        assert_eq!(result.final_loss, result.epoch_losses.last().copied());
    }

    #[test]
    fn test_phase2_trains_without_error() {
        let device = <TestBackend as Backend>::Device::default();

        let dataset_config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(dataset_config, 3, 20, 42);

        let training_config = TrainingConfig {
            num_epochs: 2,
            batch_size: 16,
            learning_rate: 1e-3,
            log_every: 1,
            ..Default::default()
        };

        let world_model_config = WorldModelConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_dim: 16,
            num_layers: 1,
            ..Default::default()
        };

        let trainer = WorldModelTrainer::new(training_config, world_model_config);
        let phase1_result = trainer.train_phase1_with_summary::<TestBackend>(&dataset, &device);
        let result = trainer.train_phase2_with_summary::<TestBackend>(
            &dataset,
            phase1_result.model,
            2,
            &device,
        );
        assert_eq!(result.final_epoch, Some(2));
        assert_eq!(result.epoch_losses.len(), 2);
        assert_eq!(result.final_loss, result.epoch_losses.last().copied());
    }
}
