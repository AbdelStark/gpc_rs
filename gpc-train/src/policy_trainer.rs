//! Diffusion policy training.

use burn::optim::{AdamWConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;

use gpc_core::config::{PolicyConfig, TrainingConfig};
use gpc_core::noise::DdpmSchedule;
use gpc_core::tensor_utils;

use crate::data::{GpcDataset, PolicyBatch, PolicyBatcher, PolicySampleData};

/// Result of a completed policy training run.
pub struct PolicyTrainingResult<B: burn::tensor::backend::Backend> {
    /// Trained diffusion policy model.
    pub model: gpc_policy::DiffusionPolicy<B>,
    /// Final epoch that ran, if any.
    pub final_epoch: Option<usize>,
    /// Final averaged loss for the last epoch, if any.
    pub final_loss: Option<f32>,
    /// Average loss for each completed epoch.
    pub epoch_losses: Vec<f32>,
}

/// Validation-aware summary for diffusion policy training.
pub struct PolicyValidationSummary<B: burn::tensor::backend::Backend> {
    /// Final training result for the full run.
    pub training: PolicyTrainingResult<B>,
    /// Best model selected by validation loss, or the final model if no validation data was used.
    pub best_model: gpc_policy::DiffusionPolicy<B>,
    /// Epoch at which the best validation score was observed.
    pub best_epoch: Option<usize>,
    /// Validation loss after each epoch.
    pub validation_losses: Vec<f32>,
    /// Lowest validation loss observed during training.
    pub best_validation_loss: Option<f32>,
}

/// Diffusion policy trainer.
pub struct PolicyTrainer {
    training_config: TrainingConfig,
    policy_config: PolicyConfig,
}

impl PolicyTrainer {
    /// Create a new trainer.
    pub fn new(training_config: TrainingConfig, policy_config: PolicyConfig) -> Self {
        Self {
            training_config,
            policy_config,
        }
    }

    /// Train the diffusion policy.
    ///
    /// Returns only the trained model for backwards compatibility.
    pub fn train<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        device: &B::Device,
    ) -> gpc_policy::DiffusionPolicy<B> {
        self.train_with_summary::<B>(dataset, device).model
    }

    /// Train the diffusion policy and return a summary of the final epoch.
    ///
    /// Uses DDPM training: sample noise, add to actions, predict noise, minimize MSE.
    pub fn train_with_summary<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        device: &B::Device,
    ) -> PolicyTrainingResult<B> {
        self.train_with_validation_summary::<B>(dataset, None, device)
            .training
    }

    /// Train the diffusion policy and return the final model plus validation selection metadata.
    ///
    /// If a validation dataset is supplied, the best model is the epoch with the
    /// lowest validation loss. Otherwise the best model falls back to the final model.
    pub fn train_with_validation_summary<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        validation_dataset: Option<&GpcDataset>,
        device: &B::Device,
    ) -> PolicyValidationSummary<B> {
        use gpc_policy::DiffusionPolicyConfig;

        tracing::info!("Starting diffusion policy training");

        let policy_config = DiffusionPolicyConfig::from_policy_config(&self.policy_config);
        let mut model = policy_config.init::<B>(device);
        let mut final_epoch = None;
        let mut final_loss = None;
        let mut epoch_losses = Vec::with_capacity(self.training_config.num_epochs);
        let mut validation_losses = Vec::with_capacity(self.training_config.num_epochs);
        let mut best_validation_loss = None;
        let mut best_epoch = None;
        let mut best_model = model.clone();

        let schedule = DdpmSchedule::new(&self.policy_config.noise_schedule);

        let optimizer_config =
            AdamWConfig::new().with_weight_decay(self.training_config.weight_decay as f32);
        let mut optim = optimizer_config.init();

        let samples = dataset.policy_samples();
        if samples.is_empty() {
            tracing::warn!("No policy training samples available");
            return PolicyValidationSummary {
                training: PolicyTrainingResult {
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
            .map(|dataset| dataset.policy_samples())
            .filter(|samples| !samples.is_empty());

        let batcher = PolicyBatcher::<B>::new(
            device.clone(),
            self.policy_config.obs_dim,
            self.policy_config.action_dim,
            self.policy_config.obs_horizon,
            self.policy_config.pred_horizon,
        );
        let batch_size = self.training_config.batch_size.min(samples.len());

        for epoch in 0..self.training_config.num_epochs {
            let mut epoch_loss = 0.0_f32;
            let mut num_batches = 0;

            for chunk in samples.chunks(batch_size) {
                let batch: PolicyBatch<B> = burn::data::dataloader::batcher::Batcher::batch(
                    &batcher,
                    chunk.to_vec(),
                    batcher.device(),
                );
                let loss = policy_batch_loss(&model, &schedule, &batch);
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
                    "Policy | Epoch {}/{} | Loss: {:.6}",
                    epoch + 1,
                    self.training_config.num_epochs,
                    avg_loss
                );
            }

            if let Some(validation_samples) = validation_samples.as_ref() {
                if let Some(validation_loss) =
                    evaluate_policy_loss(&model, &schedule, validation_samples, &batcher)
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

        tracing::info!("Policy training complete");
        let training = PolicyTrainingResult {
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

        PolicyValidationSummary {
            training,
            best_model,
            best_epoch,
            validation_losses,
            best_validation_loss,
        }
    }
}

fn policy_batch_loss<B: burn::tensor::backend::Backend>(
    model: &gpc_policy::DiffusionPolicy<B>,
    schedule: &DdpmSchedule,
    batch: &PolicyBatch<B>,
) -> Tensor<B, 1> {
    let [bs, pred_h, act_dim] = batch.actions.dims();
    let device = batch.actions.device();

    let actions_flat = tensor_utils::flatten_last_two(batch.actions.clone());
    let obs_flat = tensor_utils::flatten_last_two(batch.observations.clone());

    let timestep_indices: Vec<usize> = (0..bs)
        .map(|_| {
            (rand::random::<f32>() * schedule.num_timesteps as f32)
                .floor()
                .min((schedule.num_timesteps - 1) as f32) as usize
        })
        .collect();
    let timesteps = Tensor::<B, 1>::from_floats(
        timestep_indices
            .iter()
            .map(|&t| t as f32)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );

    let noise = Tensor::<B, 2>::random(
        [bs, pred_h * act_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let noisy_actions =
        schedule.add_noise_batch(&actions_flat, &noise, timestep_indices.as_slice());
    let noise_pred = model.predict_noise(noisy_actions, obs_flat, timesteps);

    tensor_utils::mse_loss(&noise_pred, &noise)
}

fn evaluate_policy_loss<B: burn::tensor::backend::Backend>(
    model: &gpc_policy::DiffusionPolicy<B>,
    schedule: &DdpmSchedule,
    samples: &[PolicySampleData],
    batcher: &PolicyBatcher<B>,
) -> Option<f32> {
    if samples.is_empty() {
        return None;
    }

    let batch_size = samples.len().clamp(1, 64);
    let mut total_loss = 0.0_f32;
    let mut num_batches = 0usize;

    for chunk in samples.chunks(batch_size) {
        let batch: PolicyBatch<B> = burn::data::dataloader::batcher::Batcher::batch(
            batcher,
            chunk.to_vec(),
            batcher.device(),
        );
        let loss = policy_batch_loss(model, schedule, &batch);
        let loss_val: f32 = loss.into_scalar().elem();
        total_loss += loss_val;
        num_batches += 1;
    }

    (num_batches > 0).then_some(total_loss / num_batches as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GpcDatasetConfig;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::NdArray;
    use gpc_core::traits::Policy;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_policy_training_runs() {
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
            batch_size: 8,
            learning_rate: 1e-3,
            log_every: 1,
            ..Default::default()
        };

        let policy_config = PolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 4,
            hidden_dim: 16,
            num_res_blocks: 1,
            ..Default::default()
        };

        let trainer = PolicyTrainer::new(training_config, policy_config);
        let result = trainer.train_with_summary::<TestBackend>(&dataset, &device);
        assert_eq!(result.final_epoch, Some(2));
        assert_eq!(result.epoch_losses.len(), 2);
        assert_eq!(result.final_loss, result.epoch_losses.last().copied());
    }

    #[test]
    fn test_policy_validation_summary_tracks_best_model() {
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
            batch_size: 8,
            learning_rate: 1e-3,
            log_every: 1,
            ..Default::default()
        };

        let policy_config = PolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 4,
            hidden_dim: 16,
            num_res_blocks: 1,
            ..Default::default()
        };

        let trainer = PolicyTrainer::new(training_config, policy_config);
        let summary = trainer.train_with_validation_summary::<TestBackend>(
            &split.train,
            Some(&split.validation),
            &device,
        );

        assert_eq!(summary.training.final_epoch, Some(2));
        assert_eq!(summary.validation_losses.len(), 2);
        assert!(summary.best_epoch.is_some());
        assert!(summary.best_validation_loss.is_some());

        let obs = Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device);
        let best_actions = summary.best_model.sample(&obs, &device).unwrap();
        assert_eq!(best_actions.dims(), [1, 4, 2]);
    }
}
