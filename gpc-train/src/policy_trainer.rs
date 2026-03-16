//! Diffusion policy training.

use burn::optim::{AdamWConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;

use gpc_core::config::{PolicyConfig, TrainingConfig};
use gpc_core::noise::DdpmSchedule;
use gpc_core::tensor_utils;

use crate::data::{GpcDataset, PolicyBatch, PolicyBatcher};

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
    /// Uses DDPM training: sample noise, add to actions, predict noise, minimize MSE.
    pub fn train<B: burn::tensor::backend::AutodiffBackend>(
        &self,
        dataset: &GpcDataset,
        device: &B::Device,
    ) -> gpc_policy::DiffusionPolicy<B> {
        use gpc_policy::DiffusionPolicyConfig;

        tracing::info!("Starting diffusion policy training");

        let policy_config = DiffusionPolicyConfig::from_policy_config(&self.policy_config);
        let mut model = policy_config.init::<B>(device);

        let schedule = DdpmSchedule::new(&self.policy_config.noise_schedule);

        let optimizer_config =
            AdamWConfig::new().with_weight_decay(self.training_config.weight_decay as f32);
        let mut optim = optimizer_config.init();

        let samples = dataset.policy_samples();
        if samples.is_empty() {
            tracing::warn!("No policy training samples available");
            return model;
        }

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
                    device,
                );

                let [bs, pred_h, act_dim] = batch.actions.dims();

                // Flatten actions for diffusion
                let actions_flat = tensor_utils::flatten_last_two(batch.actions.clone());
                let obs_flat = tensor_utils::flatten_last_two(batch.observations.clone());

                // Sample random timesteps
                let timesteps_vec: Vec<f32> = (0..bs)
                    .map(|_| {
                        (rand::random::<f32>() * schedule.num_timesteps as f32)
                            .floor()
                            .min((schedule.num_timesteps - 1) as f32)
                    })
                    .collect();
                let timesteps = Tensor::<B, 1>::from_floats(timesteps_vec.as_slice(), device);

                // Sample noise
                let noise = Tensor::<B, 2>::random(
                    [bs, pred_h * act_dim],
                    Distribution::Normal(0.0, 1.0),
                    device,
                );

                // Add noise to actions (per-sample timestep)
                // For simplicity, use a uniform timestep per batch
                let t_idx = timesteps_vec[0] as usize;
                let noisy_actions = schedule.add_noise(&actions_flat, &noise, t_idx);

                // Predict noise
                let noise_pred = model.predict_noise(noisy_actions, obs_flat, timesteps);

                // MSE loss between predicted and actual noise
                let loss = tensor_utils::mse_loss(&noise_pred, &noise);
                let loss_val: f32 = loss.clone().into_scalar().elem();

                // Backward pass
                let grads = loss.backward();
                let grads = burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(self.training_config.learning_rate, model, grads);

                epoch_loss += loss_val;
                num_batches += 1;
            }

            if epoch % self.training_config.log_every == 0 {
                let avg_loss = epoch_loss / num_batches as f32;
                tracing::info!(
                    "Policy | Epoch {}/{} | Loss: {:.6}",
                    epoch + 1,
                    self.training_config.num_epochs,
                    avg_loss
                );
            }
        }

        tracing::info!("Policy training complete");
        model
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
        let _model = trainer.train::<TestBackend>(&dataset, &device);
    }
}
