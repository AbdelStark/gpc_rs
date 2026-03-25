//! World model training orchestration (Phase 1 + Phase 2).

use burn::optim::{AdamWConfig, Optimizer};
use burn::prelude::*;

use gpc_core::config::{TrainingConfig, WorldModelConfig};
use gpc_core::tensor_utils;

use crate::data::{
    GpcDataset, WorldModelBatch, WorldModelBatcher, WorldModelSequenceSample,
    WorldModelTransitionSample,
};

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

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

/// Validation-aware summary for world model training.
pub struct WorldModelValidationSummary<B: burn::tensor::backend::Backend> {
    /// Final training result for the full run.
    pub training: WorldModelTrainingResult<B>,
    /// Best model selected by validation loss, or the final model if no validation data was used.
    pub best_model: gpc_world::StateWorldModel<B>,
    /// Epoch at which the best validation score was observed.
    pub best_epoch: Option<usize>,
    /// Validation loss after each epoch.
    pub validation_losses: Vec<f32>,
    /// Lowest validation loss observed during training.
    pub best_validation_loss: Option<f32>,
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
        self.train_phase1_with_validation_summary::<B>(dataset, None, device)
            .training
    }

    /// Train Phase 1: Single-step prediction warmup and return validation metadata.
    pub fn train_phase1_with_validation_summary<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        validation_dataset: Option<&GpcDataset>,
        device: &B::Device,
    ) -> WorldModelValidationSummary<B> {
        use gpc_world::world_model::StateWorldModelConfig;

        tracing::info!("Starting Phase 1 training: single-step prediction");

        B::seed(device, self.training_config.seed);
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
        let mut validation_losses = Vec::with_capacity(self.training_config.num_epochs);
        let mut best_validation_loss = None;
        let mut best_epoch = None;
        let mut best_model = model.clone();
        let optimizer_config =
            AdamWConfig::new().with_weight_decay(self.training_config.weight_decay as f32);

        let mut optim = optimizer_config.init();

        let samples = dataset.world_model_samples();
        if samples.is_empty() {
            tracing::warn!("No world model training samples available for Phase 1");
            return WorldModelValidationSummary {
                training: WorldModelTrainingResult {
                    model,
                    final_epoch,
                    final_loss,
                    epoch_losses,
                },
                best_model,
                best_epoch,
                validation_losses,
                best_validation_loss,
            };
        }

        let validation_samples = validation_dataset
            .map(|dataset| dataset.world_model_samples())
            .filter(|samples| !samples.is_empty());

        let batcher = WorldModelBatcher::<B>::new(device.clone());
        let batch_size = self.training_config.batch_size.min(samples.len());

        for epoch in 0..self.training_config.num_epochs {
            let mut epoch_loss = 0.0_f32;
            let mut num_batches = 0;

            let mut indices: Vec<usize> = (0..samples.len()).collect();
            let mut shuffle_rng =
                StdRng::seed_from_u64(epoch_seed(self.training_config.seed, epoch));
            indices.shuffle(&mut shuffle_rng);

            // Simple batching
            for chunk in indices.chunks(batch_size) {
                let batch_samples = chunk.iter().map(|&index| samples[index].clone()).collect();
                let batch: WorldModelBatch<B> = burn::data::dataloader::batcher::Batcher::batch(
                    &batcher,
                    batch_samples,
                    batcher.device(),
                );
                let loss = world_model_phase1_batch_loss(&model, &batch);
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

            if let Some(validation_samples) = validation_samples.as_ref() {
                if let Some(validation_loss) =
                    evaluate_phase1_loss(&model, validation_samples, &batcher)
                {
                    validation_losses.push(validation_loss);
                    let is_best = best_validation_loss
                        .map(|best| validation_loss < best)
                        .unwrap_or(true);
                    if is_best {
                        best_validation_loss = Some(validation_loss);
                        best_epoch = Some(epoch + 1);
                        best_model = model.clone();
                    }
                }
            }
        }

        tracing::info!("Phase 1 training complete");
        let training = WorldModelTrainingResult {
            model,
            final_epoch,
            final_loss,
            epoch_losses,
        };
        let best_model = if best_epoch.is_some() {
            best_model
        } else {
            training.model.clone()
        };

        WorldModelValidationSummary {
            training,
            best_model,
            best_epoch,
            validation_losses,
            best_validation_loss,
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
        self.train_phase2_with_validation_summary::<B>(dataset, model, horizon, None, device)
            .training
            .model
    }

    /// Train Phase 2: Multi-step rollout training and return validation metadata.
    pub fn train_phase2_with_validation_summary<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        model: gpc_world::StateWorldModel<B>,
        horizon: usize,
        validation_dataset: Option<&GpcDataset>,
        device: &B::Device,
    ) -> WorldModelValidationSummary<B> {
        self.train_phase2_with_validation_summary_internal::<B>(
            dataset,
            model,
            horizon,
            validation_dataset,
            device,
        )
    }

    /// Train Phase 2: Multi-step rollout training and return final metadata.
    ///
    /// Fine-tunes the world model using multi-step trajectory prediction
    /// with joint supervision at each timestep.
    pub fn train_phase2_with_summary<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        model: gpc_world::StateWorldModel<B>,
        horizon: usize,
        device: &B::Device,
    ) -> WorldModelTrainingResult<B> {
        self.train_phase2_with_validation_summary_internal::<B>(
            dataset, model, horizon, None, device,
        )
        .training
    }

    fn train_phase2_with_validation_summary_internal<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        mut model: gpc_world::StateWorldModel<B>,
        horizon: usize,
        validation_dataset: Option<&GpcDataset>,
        device: &B::Device,
    ) -> WorldModelValidationSummary<B> {
        tracing::info!(
            "Starting Phase 2 training: multi-step rollout (horizon={})",
            horizon
        );

        B::seed(device, self.training_config.seed);
        let mut final_epoch = None;
        let mut final_loss = None;
        let mut epoch_losses = Vec::with_capacity(self.training_config.num_epochs);
        let mut validation_losses = Vec::with_capacity(self.training_config.num_epochs);
        let mut best_validation_loss = None;
        let mut best_epoch = None;
        let mut best_model = model.clone();
        let optimizer_config =
            AdamWConfig::new().with_weight_decay(self.training_config.weight_decay as f32);
        let mut optim = optimizer_config.init();

        let sequences = dataset.world_model_sequences(horizon);
        if sequences.is_empty() {
            tracing::warn!("No sequences of sufficient length for Phase 2");
            return WorldModelValidationSummary {
                training: WorldModelTrainingResult {
                    model,
                    final_epoch,
                    final_loss,
                    epoch_losses,
                },
                best_model,
                best_epoch,
                validation_losses,
                best_validation_loss,
            };
        }

        let validation_sequences = validation_dataset
            .map(|dataset| dataset.world_model_sequences(horizon))
            .filter(|samples| !samples.is_empty());

        let batch_size = self.training_config.batch_size.min(sequences.len());
        let state_dim = self.world_model_config.state_dim;
        let action_dim = self.world_model_config.action_dim;

        for epoch in 0..self.training_config.num_epochs {
            let mut epoch_loss = 0.0_f32;
            let mut num_batches = 0;

            let mut indices: Vec<usize> = (0..sequences.len()).collect();
            let mut shuffle_rng =
                StdRng::seed_from_u64(epoch_seed(self.training_config.seed, epoch));
            indices.shuffle(&mut shuffle_rng);

            for chunk in indices.chunks(batch_size) {
                let batch_sequences: Vec<_> = chunk
                    .iter()
                    .map(|&index| sequences[index].clone())
                    .collect();
                let avg_loss = world_model_phase2_batch_loss(
                    &model,
                    batch_sequences.as_slice(),
                    horizon,
                    state_dim,
                    action_dim,
                    device,
                );
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

            if let Some(validation_sequences) = validation_sequences.as_ref() {
                if let Some(validation_loss) = evaluate_phase2_loss(
                    &model,
                    validation_sequences,
                    horizon,
                    state_dim,
                    action_dim,
                    device,
                ) {
                    validation_losses.push(validation_loss);
                    let is_best = best_validation_loss
                        .map(|best| validation_loss < best)
                        .unwrap_or(true);
                    if is_best {
                        best_validation_loss = Some(validation_loss);
                        best_epoch = Some(epoch + 1);
                        best_model = model.clone();
                    }
                }
            }
        }

        tracing::info!("Phase 2 training complete");
        let training = WorldModelTrainingResult {
            model,
            final_epoch,
            final_loss,
            epoch_losses,
        };
        let best_model = if best_epoch.is_some() {
            best_model
        } else {
            training.model.clone()
        };

        WorldModelValidationSummary {
            training,
            best_model,
            best_epoch,
            validation_losses,
            best_validation_loss,
        }
    }
}

impl<B: burn::tensor::backend::Backend> From<WorldModelTrainingResult<B>>
    for WorldModelValidationSummary<B>
{
    fn from(training: WorldModelTrainingResult<B>) -> Self {
        let best_model = training.model.clone();
        Self {
            best_model,
            best_epoch: None,
            validation_losses: Vec::new(),
            best_validation_loss: None,
            training,
        }
    }
}

fn world_model_phase1_batch_loss<B: burn::tensor::backend::Backend>(
    model: &gpc_world::StateWorldModel<B>,
    batch: &WorldModelBatch<B>,
) -> Tensor<B, 1> {
    let predicted_delta = model.predict_delta(&batch.states, &batch.actions);
    let target_delta = batch.next_states.clone() - batch.states.clone();
    tensor_utils::mse_loss(&predicted_delta, &target_delta)
}

fn evaluate_phase1_loss<B: burn::tensor::backend::Backend>(
    model: &gpc_world::StateWorldModel<B>,
    samples: &[WorldModelTransitionSample],
    batcher: &WorldModelBatcher<B>,
) -> Option<f32> {
    if samples.is_empty() {
        return None;
    }

    let batch_size = samples.len().clamp(1, 64);
    let mut total_loss = 0.0_f32;
    let mut num_batches = 0usize;

    for chunk in samples.chunks(batch_size) {
        let batch: WorldModelBatch<B> = burn::data::dataloader::batcher::Batcher::batch(
            batcher,
            chunk.to_vec(),
            batcher.device(),
        );
        let loss = world_model_phase1_batch_loss(model, &batch);
        let loss_val: f32 = loss.into_scalar().elem();
        total_loss += loss_val;
        num_batches += 1;
    }

    (num_batches > 0).then_some(total_loss / num_batches as f32)
}

fn world_model_phase2_batch_loss<B: burn::tensor::backend::Backend>(
    model: &gpc_world::StateWorldModel<B>,
    chunk: &[WorldModelSequenceSample],
    horizon: usize,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let bs = chunk.len();

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

    total_loss / (horizon as f32)
}

fn evaluate_phase2_loss<B: Backend>(
    model: &gpc_world::StateWorldModel<B>,
    sequences: &[WorldModelSequenceSample],
    horizon: usize,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Option<f32> {
    if sequences.is_empty() {
        return None;
    }

    let batch_size = sequences.len().clamp(1, 64);
    let mut total_loss = 0.0_f32;
    let mut num_batches = 0usize;

    for chunk in sequences.chunks(batch_size) {
        let loss =
            world_model_phase2_batch_loss(model, chunk, horizon, state_dim, action_dim, device);
        let loss_val: f32 = loss.into_scalar().elem();
        total_loss += loss_val;
        num_batches += 1;
    }

    (num_batches > 0).then_some(total_loss / num_batches as f32)
}

fn epoch_seed(seed: u64, epoch: usize) -> u64 {
    seed.wrapping_add((epoch as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GpcDatasetConfig;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::NdArray;
    use gpc_core::traits::WorldModel;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_phase1_trains_without_error() {
        let _guard = crate::test_support::training_test_guard();
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
    fn test_phase1_training_is_deterministic_for_seed() {
        let _guard = crate::test_support::training_test_guard();
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
            seed: 1337,
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
        let first = trainer.train_phase1_with_summary::<TestBackend>(&dataset, &device);
        let second = trainer.train_phase1_with_summary::<TestBackend>(&dataset, &device);

        assert_eq!(first.epoch_losses, second.epoch_losses);
        assert_eq!(first.final_loss, second.final_loss);
        assert_eq!(first.final_epoch, second.final_epoch);
    }

    #[test]
    fn test_phase1_validation_summary_tracks_best_model() {
        let _guard = crate::test_support::training_test_guard();
        let device = <TestBackend as Backend>::Device::default();

        let dataset_config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(dataset_config, 6, 20, 7);
        let split = dataset.split(0.33, 11).unwrap();

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
        let summary = trainer.train_phase1_with_validation_summary::<TestBackend>(
            &split.train,
            Some(&split.validation),
            &device,
        );

        assert_eq!(summary.training.final_epoch, Some(2));
        assert_eq!(summary.validation_losses.len(), 2);
        assert!(summary.best_epoch.is_some());
        assert!(summary.best_validation_loss.is_some());

        let state = Tensor::<TestBackend, 2>::zeros([1, 4], &device);
        let action = Tensor::<TestBackend, 2>::zeros([1, 2], &device);
        let next_state = summary
            .best_model
            .predict_next_state(&state, &action)
            .unwrap();
        assert_eq!(next_state.dims(), [1, 4]);
    }

    #[test]
    fn test_phase2_trains_without_error() {
        let _guard = crate::test_support::training_test_guard();
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

    #[test]
    fn test_phase2_training_is_deterministic_for_seed() {
        let _guard = crate::test_support::training_test_guard();
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
            seed: 1337,
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
        let phase1 = trainer.train_phase1_with_summary::<TestBackend>(&dataset, &device);
        let first = trainer.train_phase2_with_summary::<TestBackend>(
            &dataset,
            phase1.model.clone(),
            2,
            &device,
        );
        let second =
            trainer.train_phase2_with_summary::<TestBackend>(&dataset, phase1.model, 2, &device);

        assert_eq!(first.epoch_losses, second.epoch_losses);
        assert_eq!(first.final_loss, second.final_loss);
        assert_eq!(first.final_epoch, second.final_epoch);
    }

    #[test]
    fn test_phase2_validation_summary_tracks_best_model() {
        let _guard = crate::test_support::training_test_guard();
        let device = <TestBackend as Backend>::Device::default();

        let dataset_config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(dataset_config, 6, 20, 7);
        let split = dataset.split(0.33, 11).unwrap();

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
        let phase1 = trainer.train_phase1_with_validation_summary::<TestBackend>(
            &split.train,
            Some(&split.validation),
            &device,
        );
        let summary = trainer.train_phase2_with_validation_summary::<TestBackend>(
            &split.train,
            phase1.training.model,
            2,
            Some(&split.validation),
            &device,
        );

        assert_eq!(summary.training.final_epoch, Some(2));
        assert_eq!(summary.validation_losses.len(), 2);
        assert!(summary.best_epoch.is_some());
        assert!(summary.best_validation_loss.is_some());

        let state = Tensor::<TestBackend, 2>::zeros([1, 4], &device);
        let action = Tensor::<TestBackend, 2>::zeros([1, 2], &device);
        let next_state = summary
            .best_model
            .predict_next_state(&state, &action)
            .unwrap();
        assert_eq!(next_state.dims(), [1, 4]);
    }
}
