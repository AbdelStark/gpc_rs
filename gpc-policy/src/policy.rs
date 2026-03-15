//! Diffusion policy implementing the DDPM sampling loop.

use burn::prelude::*;
use burn::tensor::Distribution;

use gpc_core::config::PolicyConfig;
use gpc_core::noise::DdpmSchedule;
use gpc_core::tensor_utils;
use gpc_core::traits::Policy;

use crate::network::{DenoisingNetwork, DenoisingNetworkConfig};

/// Diffusion-based action policy.
///
/// Generates action sequences by iteratively denoising Gaussian noise,
/// conditioned on observation history, using the DDPM reverse process.
#[derive(Module, Debug)]
pub struct DiffusionPolicy<B: Backend> {
    /// The denoising network that predicts noise at each step.
    network: DenoisingNetwork<B>,
    /// Flattened action dimension (pred_horizon * action_dim).
    #[module(skip)]
    flat_action_dim: usize,
    /// Observation conditioning dimension (obs_horizon * obs_dim).
    #[module(skip)]
    cond_dim: usize,
    /// Timestep embedding dimension.
    #[module(skip)]
    time_embed_dim: usize,
    /// Prediction horizon.
    #[module(skip)]
    pred_horizon: usize,
    /// Action dimensionality.
    #[module(skip)]
    action_dim: usize,
}

/// Configuration for the diffusion policy.
#[derive(Config, Debug)]
pub struct DiffusionPolicyConfig {
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Action dimensionality.
    pub action_dim: usize,
    /// Number of observation history frames.
    #[config(default = 2)]
    pub obs_horizon: usize,
    /// Number of future action steps to predict.
    #[config(default = 16)]
    pub pred_horizon: usize,
    /// Hidden layer size.
    #[config(default = 256)]
    pub hidden_dim: usize,
    /// Timestep embedding dimension.
    #[config(default = 128)]
    pub time_embed_dim: usize,
    /// Number of residual blocks.
    #[config(default = 3)]
    pub num_res_blocks: usize,
}

impl DiffusionPolicyConfig {
    /// Create from a [`PolicyConfig`].
    pub fn from_policy_config(config: &PolicyConfig) -> Self {
        Self {
            obs_dim: config.obs_dim,
            action_dim: config.action_dim,
            obs_horizon: config.obs_horizon,
            pred_horizon: config.pred_horizon,
            hidden_dim: config.hidden_dim,
            time_embed_dim: 128,
            num_res_blocks: config.num_res_blocks,
        }
    }

    /// Initialize the diffusion policy.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DiffusionPolicy<B> {
        let flat_action_dim = self.pred_horizon * self.action_dim;
        let cond_dim = self.obs_horizon * self.obs_dim;

        let network_config = DenoisingNetworkConfig {
            input_dim: flat_action_dim,
            cond_dim,
            hidden_dim: self.hidden_dim,
            time_embed_dim: self.time_embed_dim,
            num_blocks: self.num_res_blocks,
        };

        DiffusionPolicy {
            network: network_config.init(device),
            flat_action_dim,
            cond_dim,
            time_embed_dim: self.time_embed_dim,
            pred_horizon: self.pred_horizon,
            action_dim: self.action_dim,
        }
    }
}

impl<B: Backend> DiffusionPolicy<B> {
    /// Predict noise for the training loss.
    ///
    /// # Arguments
    /// * `noisy_actions` - Noisy flattened actions `[batch, pred_horizon * action_dim]`
    /// * `obs_cond` - Flattened observation history `[batch, obs_horizon * obs_dim]`
    /// * `timesteps` - Diffusion timesteps `[batch]`
    pub fn predict_noise(
        &self,
        noisy_actions: Tensor<B, 2>,
        obs_cond: Tensor<B, 2>,
        timesteps: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let device = noisy_actions.device();
        let time_emb = tensor_utils::timestep_embedding(&timesteps, self.time_embed_dim, &device);
        self.network.forward(noisy_actions, obs_cond, time_emb)
    }

    /// Run the full DDPM reverse sampling loop.
    ///
    /// Starts from pure Gaussian noise and iteratively denoises to produce
    /// an action sequence.
    ///
    /// # Arguments
    /// * `obs_cond` - Flattened observation conditioning `[batch, obs_horizon * obs_dim]`
    /// * `schedule` - DDPM noise schedule
    /// * `device` - Device to run on
    fn ddpm_sample(
        &self,
        obs_cond: Tensor<B, 2>,
        schedule: &DdpmSchedule,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch_size, _] = obs_cond.dims();

        // Start from pure noise
        let mut x_t = Tensor::<B, 2>::random(
            [batch_size, self.flat_action_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );

        // Reverse diffusion: t = T-1, T-2, ..., 0
        for t in (0..schedule.num_timesteps).rev() {
            let timesteps =
                Tensor::<B, 1>::from_floats(vec![t as f32; batch_size].as_slice(), device);

            let predicted_noise = self.predict_noise(x_t.clone(), obs_cond.clone(), timesteps);

            // Apply reverse step
            x_t = schedule.remove_noise(&x_t, &predicted_noise, t);

            // Add noise for all steps except t=0
            if t > 0 {
                let noise = Tensor::<B, 2>::random(
                    [batch_size, self.flat_action_dim],
                    Distribution::Normal(0.0, 1.0),
                    device,
                );
                let sigma = (schedule.posterior_variance[t] as f32).sqrt();
                x_t = x_t + noise * sigma;
            }
        }

        // Reshape to [batch, pred_horizon, action_dim]
        x_t.reshape([batch_size, self.pred_horizon, self.action_dim])
    }
}

impl<B: Backend> Policy<B> for DiffusionPolicy<B> {
    fn sample(
        &self,
        obs_history: &Tensor<B, 3>,
        device: &B::Device,
    ) -> gpc_core::Result<Tensor<B, 3>> {
        let schedule = DdpmSchedule::new(&gpc_core::config::NoiseScheduleConfig::default());
        let obs_cond = tensor_utils::flatten_last_two(obs_history.clone());
        Ok(self.ddpm_sample(obs_cond, &schedule, device))
    }

    fn sample_k(
        &self,
        obs_history: &Tensor<B, 3>,
        num_candidates: usize,
        device: &B::Device,
    ) -> gpc_core::Result<Tensor<B, 3>> {
        let schedule = DdpmSchedule::new(&gpc_core::config::NoiseScheduleConfig::default());
        // Repeat observations K times along batch dimension
        let obs_repeated = tensor_utils::repeat_batch(obs_history, num_candidates);
        let obs_cond = tensor_utils::flatten_last_two(obs_repeated);

        Ok(self.ddpm_sample(obs_cond, &schedule, device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use gpc_core::traits::Policy;

    type TestBackend = NdArray;

    #[test]
    fn test_diffusion_policy_sample_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = DiffusionPolicyConfig {
            obs_dim: 10,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 8,
            hidden_dim: 32,
            time_embed_dim: 16,
            num_res_blocks: 1,
        };

        let policy = config.init::<TestBackend>(&device);

        let obs = Tensor::<TestBackend, 3>::zeros([1, 2, 10], &device);
        let actions = policy.sample(&obs, &device).unwrap();
        assert_eq!(actions.dims(), [1, 8, 2]);
    }

    #[test]
    fn test_diffusion_policy_sample_k_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = DiffusionPolicyConfig {
            obs_dim: 10,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 8,
            hidden_dim: 32,
            time_embed_dim: 16,
            num_res_blocks: 1,
        };

        let policy = config.init::<TestBackend>(&device);

        let obs = Tensor::<TestBackend, 3>::zeros([1, 2, 10], &device);
        let actions = policy.sample_k(&obs, 5, &device).unwrap();
        assert_eq!(actions.dims(), [5, 8, 2]);
    }

    #[test]
    fn test_predict_noise_shape() {
        let device = <TestBackend as Backend>::Device::default();

        let config = DiffusionPolicyConfig {
            obs_dim: 10,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 8,
            hidden_dim: 32,
            time_embed_dim: 16,
            num_res_blocks: 1,
        };

        let policy = config.init::<TestBackend>(&device);

        let noisy = Tensor::<TestBackend, 2>::zeros([4, 16], &device);
        let cond = Tensor::<TestBackend, 2>::zeros([4, 20], &device);
        let t = Tensor::<TestBackend, 1>::from_floats([5.0, 10.0, 50.0, 99.0], &device);

        let noise_pred = policy.predict_noise(noisy, cond, t);
        assert_eq!(noise_pred.dims(), [4, 16]);
    }
}
