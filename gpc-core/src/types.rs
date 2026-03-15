//! Data types for observations, actions, states, and trajectories.

use burn::tensor::{Tensor, backend::Backend};

/// A batch of observations as a 2D tensor `[batch_size, obs_dim]`.
#[derive(Debug, Clone)]
pub struct Observation<B: Backend> {
    pub data: Tensor<B, 2>,
}

/// A batch of actions as a 2D tensor `[batch_size, action_dim]`.
#[derive(Debug, Clone)]
pub struct Action<B: Backend> {
    pub data: Tensor<B, 2>,
}

/// An action sequence (trajectory) as a 3D tensor `[batch_size, horizon, action_dim]`.
#[derive(Debug, Clone)]
pub struct ActionSequence<B: Backend> {
    pub data: Tensor<B, 3>,
}

/// A batch of states as a 2D tensor `[batch_size, state_dim]`.
#[derive(Debug, Clone)]
pub struct State<B: Backend> {
    pub data: Tensor<B, 2>,
}

/// A state trajectory as a 3D tensor `[batch_size, horizon, state_dim]`.
#[derive(Debug, Clone)]
pub struct StateSequence<B: Backend> {
    pub data: Tensor<B, 3>,
}

/// An observation history as a 3D tensor `[batch_size, obs_horizon, obs_dim]`.
#[derive(Debug, Clone)]
pub struct ObservationHistory<B: Backend> {
    pub data: Tensor<B, 3>,
}

/// Scalar reward for each trajectory in a batch `[batch_size]`.
#[derive(Debug, Clone)]
pub struct Reward<B: Backend> {
    pub data: Tensor<B, 1>,
}

/// A complete trajectory: action sequence + predicted state sequence + reward.
#[derive(Debug, Clone)]
pub struct Trajectory<B: Backend> {
    pub actions: ActionSequence<B>,
    pub predicted_states: StateSequence<B>,
    pub reward: Reward<B>,
}

/// A single training sample for the world model.
#[derive(Debug, Clone)]
pub struct WorldModelSample<B: Backend> {
    /// Current state `[batch_size, state_dim]`.
    pub state: Tensor<B, 2>,
    /// Action applied `[batch_size, action_dim]`.
    pub action: Tensor<B, 2>,
    /// Next state (ground truth) `[batch_size, state_dim]`.
    pub next_state: Tensor<B, 2>,
}

/// A multi-step training sample for world model phase 2.
#[derive(Debug, Clone)]
pub struct WorldModelSequenceSample<B: Backend> {
    /// Initial state `[batch_size, state_dim]`.
    pub initial_state: Tensor<B, 2>,
    /// Action sequence `[batch_size, horizon, action_dim]`.
    pub actions: Tensor<B, 3>,
    /// Ground truth state sequence `[batch_size, horizon, state_dim]`.
    pub target_states: Tensor<B, 3>,
}

/// A training sample for the diffusion policy.
#[derive(Debug, Clone)]
pub struct PolicySample<B: Backend> {
    /// Observation history `[batch_size, obs_horizon, obs_dim]`.
    pub observations: Tensor<B, 3>,
    /// Ground truth action sequence `[batch_size, pred_horizon, action_dim]`.
    pub actions: Tensor<B, 3>,
}

/// Normalization statistics for data preprocessing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormalizationStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl NormalizationStats {
    /// Normalize a value using stored statistics.
    pub fn normalize(&self, values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(v, (m, s))| if *s > 1e-8 { (v - m) / s } else { v - m })
            .collect()
    }

    /// Denormalize a value using stored statistics.
    ///
    /// Uses the same zero-std guard as [`normalize`](Self::normalize):
    /// when std <= 1e-8 the value is treated as a pure offset (v + mean).
    pub fn denormalize(&self, values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(v, (m, s))| if *s > 1e-8 { v * s + m } else { v + m })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalization_roundtrip() {
        let stats = NormalizationStats {
            mean: vec![1.0, 2.0, 3.0],
            std: vec![0.5, 1.0, 2.0],
        };
        let original = vec![1.5, 3.0, 7.0];
        let normalized = stats.normalize(&original);
        let recovered = stats.denormalize(&normalized);

        for (o, r) in original.iter().zip(recovered.iter()) {
            assert_relative_eq!(o, r, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_normalization_zero_std() {
        let stats = NormalizationStats {
            mean: vec![5.0],
            std: vec![0.0],
        };
        let normalized = stats.normalize(&[5.0]);
        assert_relative_eq!(normalized[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_denormalize_zero_std_roundtrip() {
        let stats = NormalizationStats {
            mean: vec![5.0],
            std: vec![0.0],
        };
        let normalized = stats.normalize(&[5.0]);
        let recovered = stats.denormalize(&normalized);
        assert_relative_eq!(recovered[0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalization_empty_input() {
        let stats = NormalizationStats {
            mean: vec![],
            std: vec![],
        };
        let normalized = stats.normalize(&[]);
        assert!(normalized.is_empty());
        let denormalized = stats.denormalize(&[]);
        assert!(denormalized.is_empty());
    }
}
