//! Configuration types for all GPC components.

use crate::{GpcError, Result};

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

impl NoiseScheduleConfig {
    /// Validate that the configuration is well-formed.
    pub fn validate(&self) -> Result<()> {
        if self.num_timesteps == 0 {
            return Err(GpcError::Config("num_timesteps must be > 0".into()));
        }
        if self.beta_start <= 0.0 || self.beta_end <= 0.0 {
            return Err(GpcError::Config(
                "beta_start and beta_end must be positive".into(),
            ));
        }
        if self.beta_start >= self.beta_end {
            return Err(GpcError::Config(
                "beta_start must be less than beta_end".into(),
            ));
        }
        Ok(())
    }
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

impl PolicyConfig {
    /// Validate that the configuration is well-formed.
    pub fn validate(&self) -> Result<()> {
        if self.obs_dim == 0 {
            return Err(GpcError::Config("obs_dim must be > 0".into()));
        }
        if self.action_dim == 0 {
            return Err(GpcError::Config("action_dim must be > 0".into()));
        }
        if self.obs_horizon == 0 {
            return Err(GpcError::Config("obs_horizon must be > 0".into()));
        }
        if self.pred_horizon == 0 {
            return Err(GpcError::Config("pred_horizon must be > 0".into()));
        }
        if self.hidden_dim == 0 {
            return Err(GpcError::Config("hidden_dim must be > 0".into()));
        }
        self.noise_schedule.validate()?;
        Ok(())
    }
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

impl WorldModelConfig {
    /// Validate that the configuration is well-formed.
    pub fn validate(&self) -> Result<()> {
        if self.state_dim == 0 {
            return Err(GpcError::Config("state_dim must be > 0".into()));
        }
        if self.action_dim == 0 {
            return Err(GpcError::Config("action_dim must be > 0".into()));
        }
        if self.hidden_dim == 0 {
            return Err(GpcError::Config("hidden_dim must be > 0".into()));
        }
        if self.num_layers == 0 {
            return Err(GpcError::Config("num_layers must be > 0".into()));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(GpcError::Config("dropout must be in [0.0, 1.0)".into()));
        }
        Ok(())
    }
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

impl TrainingConfig {
    /// Validate that the configuration is well-formed.
    pub fn validate(&self) -> Result<()> {
        if self.num_epochs == 0 {
            return Err(GpcError::Config("num_epochs must be > 0".into()));
        }
        if self.batch_size == 0 {
            return Err(GpcError::Config("batch_size must be > 0".into()));
        }
        if self.learning_rate <= 0.0 {
            return Err(GpcError::Config("learning_rate must be positive".into()));
        }
        Ok(())
    }
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

impl GpcConfig {
    /// Validate all sub-configurations.
    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.world_model.validate()?;
        self.training.validate()?;
        if self.gpc_rank.num_candidates == 0 {
            return Err(GpcError::Config(
                "gpc_rank.num_candidates must be > 0".into(),
            ));
        }
        if self.gpc_opt.num_opt_steps == 0 {
            return Err(GpcError::Config("gpc_opt.num_opt_steps must be > 0".into()));
        }
        Ok(())
    }
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

    #[test]
    fn test_default_config_validates() {
        let config = GpcConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_noise_schedule_validation_rejects_zero_timesteps() {
        let config = NoiseScheduleConfig {
            num_timesteps: 0,
            ..NoiseScheduleConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_noise_schedule_validation_rejects_bad_betas() {
        let config = NoiseScheduleConfig {
            beta_start: 0.05,
            beta_end: 0.01,
            ..NoiseScheduleConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_policy_config_validation_rejects_zero_dims() {
        let config = PolicyConfig {
            obs_dim: 0,
            ..PolicyConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_world_model_config_validation_rejects_bad_dropout() {
        let config = WorldModelConfig {
            dropout: 1.5,
            ..WorldModelConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_config_validation_rejects_zero_batch() {
        let config = TrainingConfig {
            batch_size: 0,
            ..TrainingConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
