//! Configuration types for all GPC components.

/// Configuration for the diffusion noise schedule.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NoiseScheduleConfig {
    /// Number of diffusion timesteps.
    pub num_timesteps: usize,
    /// Minimum beta value (start of linear schedule).
    pub beta_start: f64,
    /// Maximum beta value (end of linear schedule).
    pub beta_end: f64,
}

impl Default for NoiseScheduleConfig {
    fn default() -> Self {
        Self {
            num_timesteps: 100,
            beta_start: 1e-4,
            beta_end: 0.02,
        }
    }
}

/// Configuration for the diffusion policy network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyConfig {
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Action dimensionality.
    pub action_dim: usize,
    /// Number of observation history frames.
    pub obs_horizon: usize,
    /// Number of future action steps to predict.
    pub pred_horizon: usize,
    /// Number of action steps to execute before replanning.
    pub action_horizon: usize,
    /// Hidden layer size for the policy network.
    pub hidden_dim: usize,
    /// Number of residual blocks in each stage.
    pub num_res_blocks: usize,
    /// Noise schedule configuration.
    pub noise_schedule: NoiseScheduleConfig,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            obs_dim: 20,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 16,
            action_horizon: 9,
            hidden_dim: 256,
            num_res_blocks: 3,
            noise_schedule: NoiseScheduleConfig::default(),
        }
    }
}

/// Configuration for the world model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldModelConfig {
    /// State dimensionality.
    pub state_dim: usize,
    /// Action dimensionality.
    pub action_dim: usize,
    /// Hidden layer size for the world model.
    pub hidden_dim: usize,
    /// Number of hidden layers in the dynamics MLP.
    pub num_layers: usize,
    /// Dropout rate during training.
    pub dropout: f64,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            state_dim: 20,
            action_dim: 2,
            hidden_dim: 256,
            num_layers: 4,
            dropout: 0.0,
        }
    }
}

/// Configuration for training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay for AdamW.
    pub weight_decay: f64,
    /// Gradient clipping max norm (0 = disabled).
    pub grad_clip_norm: f64,
    /// Warmup steps for learning rate schedule.
    pub warmup_steps: usize,
    /// How often to save checkpoints (in epochs).
    pub checkpoint_every: usize,
    /// How often to log metrics (in steps).
    pub log_every: usize,
    /// Seed for reproducibility.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 3000,
            batch_size: 256,
            learning_rate: 1e-4,
            weight_decay: 1e-6,
            grad_clip_norm: 1.0,
            warmup_steps: 500,
            checkpoint_every: 100,
            log_every: 10,
            seed: 42,
        }
    }
}

/// Configuration for GPC-RANK evaluation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpcRankConfig {
    /// Number of candidate trajectories to sample.
    pub num_candidates: usize,
    /// Prediction horizon for trajectory rollout.
    pub prediction_horizon: usize,
}

impl Default for GpcRankConfig {
    fn default() -> Self {
        Self {
            num_candidates: 100,
            prediction_horizon: 16,
        }
    }
}

/// Configuration for GPC-OPT evaluation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpcOptConfig {
    /// Number of gradient optimization steps.
    pub num_opt_steps: usize,
    /// Optimization learning rate.
    pub opt_learning_rate: f64,
    /// Prediction horizon for trajectory rollout.
    pub prediction_horizon: usize,
    /// Whether to freeze noise in the world model during optimization.
    pub freeze_noise: bool,
}

impl Default for GpcOptConfig {
    fn default() -> Self {
        Self {
            num_opt_steps: 25,
            opt_learning_rate: 1e-2,
            prediction_horizon: 16,
            freeze_noise: true,
        }
    }
}

/// Top-level configuration combining all components.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GpcConfig {
    pub policy: PolicyConfig,
    pub world_model: WorldModelConfig,
    pub training: TrainingConfig,
    pub gpc_rank: GpcRankConfig,
    pub gpc_opt: GpcOptConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_serialization_roundtrip() {
        let config = GpcConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let recovered: GpcConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.policy.action_dim, config.policy.action_dim);
        assert_eq!(recovered.training.num_epochs, config.training.num_epochs);
    }
}
