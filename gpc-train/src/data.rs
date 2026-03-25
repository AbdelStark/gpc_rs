//! Dataset and data loading for GPC training.

use rand::SeedableRng;
use rand::seq::SliceRandom;

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use serde::{Deserialize, Serialize};

/// A world model sequence sample: (initial_state, action_sequence, target_states).
pub type WorldModelTransitionSample = (Vec<f32>, Vec<f32>, Vec<f32>);

/// A world model sequence sample: (initial_state, action_sequence, target_states).
pub type WorldModelSequenceSample = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>);

/// A policy sample: (observation_history, action_sequence).
pub type PolicySampleData = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Configuration for the GPC dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpcDatasetConfig {
    /// Path to the data directory.
    pub data_dir: String,
    /// State dimensionality.
    pub state_dim: usize,
    /// Action dimensionality.
    pub action_dim: usize,
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Observation horizon for policy.
    pub obs_horizon: usize,
    /// Prediction horizon.
    pub pred_horizon: usize,
}

impl Default for GpcDatasetConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".to_string(),
            state_dim: 20,
            action_dim: 2,
            obs_dim: 20,
            obs_horizon: 2,
            pred_horizon: 16,
        }
    }
}

/// A deterministic split of a GPC dataset into train and validation subsets.
#[derive(Debug, Clone)]
pub struct GpcDatasetSplit {
    /// Training subset.
    pub train: GpcDataset,
    /// Validation subset.
    pub validation: GpcDataset,
}

/// A single demonstration episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Sequence of states `[timesteps, state_dim]`.
    pub states: Vec<Vec<f32>>,
    /// Sequence of actions `[timesteps - 1, action_dim]`.
    pub actions: Vec<Vec<f32>>,
    /// Sequence of observations `[timesteps, obs_dim]`.
    pub observations: Vec<Vec<f32>>,
}

/// GPC dataset holding demonstration episodes.
#[derive(Debug, Clone)]
pub struct GpcDataset {
    episodes: Vec<Episode>,
    config: GpcDatasetConfig,
}

impl GpcDataset {
    /// Create a new dataset from episodes.
    pub fn new(episodes: Vec<Episode>, config: GpcDatasetConfig) -> Self {
        Self { episodes, config }
    }

    /// Load dataset from a JSON file.
    pub fn from_json(path: &str, config: GpcDatasetConfig) -> gpc_core::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let episodes: Vec<Episode> = serde_json::from_str(&data)?;
        Ok(Self::new(episodes, config))
    }

    /// Generate a synthetic dataset for testing and development.
    pub fn generate_synthetic(
        config: GpcDatasetConfig,
        num_episodes: usize,
        episode_length: usize,
        seed: u64,
    ) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let mut episodes = Vec::with_capacity(num_episodes);

        for _ in 0..num_episodes {
            let mut states = Vec::with_capacity(episode_length);
            let mut actions = Vec::with_capacity(episode_length - 1);
            let mut observations = Vec::with_capacity(episode_length);

            // Random initial state
            let mut state: Vec<f32> = (0..config.state_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            for t in 0..episode_length {
                states.push(state.clone());
                observations.push(state[..config.obs_dim.min(config.state_dim)].to_vec());

                if t < episode_length - 1 {
                    // Random action
                    let action: Vec<f32> = (0..config.action_dim)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect();

                    // Simple linear dynamics with noise
                    state = state
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| {
                            let a = if i < config.action_dim {
                                action[i]
                            } else {
                                0.0
                            };
                            s + 0.1 * a + rng.gen_range(-0.01..0.01)
                        })
                        .collect();

                    actions.push(action);
                }
            }

            episodes.push(Episode {
                states,
                actions,
                observations,
            });
        }

        Self::new(episodes, config)
    }

    /// Get total number of transition samples available.
    pub fn num_transitions(&self) -> usize {
        self.episodes.iter().map(|e| e.actions.len()).sum()
    }

    /// Get number of episodes.
    pub fn num_episodes(&self) -> usize {
        self.episodes.len()
    }

    /// Deterministically split the dataset into train and validation subsets.
    ///
    /// The split operates at the episode level so the resulting subsets remain
    /// internally consistent. The shuffle is seeded so repeated calls with the
    /// same seed produce identical splits.
    pub fn split(&self, validation_fraction: f32, seed: u64) -> gpc_core::Result<GpcDatasetSplit> {
        if !(0.0..1.0).contains(&validation_fraction) {
            return Err(gpc_core::GpcError::Config(
                "validation_fraction must be in [0.0, 1.0)".into(),
            ));
        }

        if self.episodes.is_empty() {
            return Ok(GpcDatasetSplit {
                train: Self::new(Vec::new(), self.config.clone()),
                validation: Self::new(Vec::new(), self.config.clone()),
            });
        }

        if validation_fraction > 0.0 && self.episodes.len() < 2 {
            return Err(gpc_core::GpcError::Config(
                "cannot create a validation split from a single episode".into(),
            ));
        }

        let mut indices: Vec<usize> = (0..self.episodes.len()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let mut validation_count =
            (self.episodes.len() as f32 * validation_fraction).round() as usize;
        if validation_fraction > 0.0 {
            validation_count = validation_count.max(1);
        }
        validation_count = validation_count.min(self.episodes.len().saturating_sub(1));

        let validation_indices = &indices[..validation_count];
        let train_indices = &indices[validation_count..];

        Ok(GpcDatasetSplit {
            train: Self::new(
                train_indices
                    .iter()
                    .map(|&index| self.episodes[index].clone())
                    .collect(),
                self.config.clone(),
            ),
            validation: Self::new(
                validation_indices
                    .iter()
                    .map(|&index| self.episodes[index].clone())
                    .collect(),
                self.config.clone(),
            ),
        })
    }

    /// Extract world model training samples (state, action, next_state).
    pub fn world_model_samples(&self) -> Vec<WorldModelTransitionSample> {
        let mut samples = Vec::new();
        for ep in &self.episodes {
            for t in 0..ep.actions.len() {
                samples.push((
                    ep.states[t].clone(),
                    ep.actions[t].clone(),
                    ep.states[t + 1].clone(),
                ));
            }
        }
        samples
    }

    /// Extract multi-step world model training sequences.
    pub fn world_model_sequences(&self, horizon: usize) -> Vec<WorldModelSequenceSample> {
        let mut samples = Vec::new();
        for ep in &self.episodes {
            if ep.actions.len() < horizon {
                continue;
            }
            for start in 0..=(ep.actions.len() - horizon) {
                let initial_state = ep.states[start].clone();
                let actions: Vec<Vec<f32>> = ep.actions[start..start + horizon].to_vec();
                let target_states: Vec<Vec<f32>> =
                    ep.states[start + 1..start + 1 + horizon].to_vec();
                samples.push((initial_state, actions, target_states));
            }
        }
        samples
    }

    /// Extract policy training samples (obs_history, action_sequence).
    pub fn policy_samples(&self) -> Vec<PolicySampleData> {
        let mut samples = Vec::new();
        let obs_h = self.config.obs_horizon;
        let pred_h = self.config.pred_horizon;

        for ep in &self.episodes {
            if ep.observations.len() < obs_h || ep.actions.len() < pred_h {
                continue;
            }

            for start in 0..=(ep.actions.len() - pred_h) {
                if start + obs_h > ep.observations.len() {
                    continue;
                }

                let obs_start = if start >= obs_h { start - obs_h + 1 } else { 0 };
                let mut obs_history: Vec<Vec<f32>> =
                    ep.observations[obs_start..obs_start + obs_h].to_vec();

                // Pad if needed
                while obs_history.len() < obs_h {
                    obs_history.insert(0, obs_history[0].clone());
                }

                let action_seq: Vec<Vec<f32>> = ep.actions[start..start + pred_h].to_vec();
                samples.push((obs_history, action_seq));
            }
        }
        samples
    }
}

/// Batcher for world model single-step training.
pub struct WorldModelBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> WorldModelBatcher<B> {
    /// Create a new batcher.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub(crate) fn device(&self) -> &B::Device {
        &self.device
    }
}

/// A batch of world model training data.
#[derive(Debug, Clone)]
pub struct WorldModelBatch<B: Backend> {
    /// Current states `[batch, state_dim]`.
    pub states: Tensor<B, 2>,
    /// Actions `[batch, action_dim]`.
    pub actions: Tensor<B, 2>,
    /// Next states `[batch, state_dim]`.
    pub next_states: Tensor<B, 2>,
}

impl<B: Backend> Batcher<B, (Vec<f32>, Vec<f32>, Vec<f32>), WorldModelBatch<B>>
    for WorldModelBatcher<B>
{
    fn batch(
        &self,
        items: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
        _device: &B::Device,
    ) -> WorldModelBatch<B> {
        let batch_size = items.len();
        let state_dim = items[0].0.len();
        let action_dim = items[0].1.len();

        let states_flat: Vec<f32> = items
            .iter()
            .flat_map(|(s, _, _)| s.iter().copied())
            .collect();
        let actions_flat: Vec<f32> = items
            .iter()
            .flat_map(|(_, a, _)| a.iter().copied())
            .collect();
        let next_flat: Vec<f32> = items
            .iter()
            .flat_map(|(_, _, n)| n.iter().copied())
            .collect();

        WorldModelBatch {
            states: Tensor::<B, 1>::from_floats(states_flat.as_slice(), &self.device)
                .reshape([batch_size, state_dim]),
            actions: Tensor::<B, 1>::from_floats(actions_flat.as_slice(), &self.device)
                .reshape([batch_size, action_dim]),
            next_states: Tensor::<B, 1>::from_floats(next_flat.as_slice(), &self.device)
                .reshape([batch_size, state_dim]),
        }
    }
}

/// Batcher for policy training.
pub struct PolicyBatcher<B: Backend> {
    device: B::Device,
    obs_dim: usize,
    action_dim: usize,
    obs_horizon: usize,
    pred_horizon: usize,
}

impl<B: Backend> PolicyBatcher<B> {
    /// Create a new policy batcher.
    pub fn new(
        device: B::Device,
        obs_dim: usize,
        action_dim: usize,
        obs_horizon: usize,
        pred_horizon: usize,
    ) -> Self {
        Self {
            device,
            obs_dim,
            action_dim,
            obs_horizon,
            pred_horizon,
        }
    }

    pub(crate) fn device(&self) -> &B::Device {
        &self.device
    }
}

/// A batch of policy training data.
#[derive(Debug, Clone)]
pub struct PolicyBatch<B: Backend> {
    /// Observation histories `[batch, obs_horizon, obs_dim]`.
    pub observations: Tensor<B, 3>,
    /// Action sequences `[batch, pred_horizon, action_dim]`.
    pub actions: Tensor<B, 3>,
}

impl<B: Backend> Batcher<B, (Vec<Vec<f32>>, Vec<Vec<f32>>), PolicyBatch<B>> for PolicyBatcher<B> {
    fn batch(
        &self,
        items: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
        _device: &B::Device,
    ) -> PolicyBatch<B> {
        let batch_size = items.len();

        let obs_flat: Vec<f32> = items
            .iter()
            .flat_map(|(obs, _)| obs.iter().flat_map(|o| o.iter().copied()))
            .collect();

        let act_flat: Vec<f32> = items
            .iter()
            .flat_map(|(_, acts)| acts.iter().flat_map(|a| a.iter().copied()))
            .collect();

        PolicyBatch {
            observations: Tensor::<B, 1>::from_floats(obs_flat.as_slice(), &self.device).reshape([
                batch_size,
                self.obs_horizon,
                self.obs_dim,
            ]),
            actions: Tensor::<B, 1>::from_floats(act_flat.as_slice(), &self.device).reshape([
                batch_size,
                self.pred_horizon,
                self.action_dim,
            ]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset_generation() {
        let config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 10,
            action_dim: 2,
            obs_dim: 10,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(config, 5, 20, 42);
        assert_eq!(dataset.num_episodes(), 5);
        assert!(dataset.num_transitions() > 0);
    }

    #[test]
    fn test_world_model_samples() {
        let config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(config, 2, 10, 42);
        let samples = dataset.world_model_samples();

        assert_eq!(samples.len(), 18); // 2 episodes * 9 transitions each
        assert_eq!(samples[0].0.len(), 4); // state_dim
        assert_eq!(samples[0].1.len(), 2); // action_dim
        assert_eq!(samples[0].2.len(), 4); // next_state_dim
    }

    #[test]
    fn test_policy_samples() {
        let config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(config, 2, 10, 42);
        let samples = dataset.policy_samples();

        assert!(!samples.is_empty());
        assert_eq!(samples[0].0.len(), 2); // obs_horizon
        assert_eq!(samples[0].1.len(), 4); // pred_horizon
    }

    #[test]
    fn test_world_model_sequences() {
        let config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(config, 2, 10, 42);
        let seqs = dataset.world_model_sequences(4);

        assert!(!seqs.is_empty());
        assert_eq!(seqs[0].1.len(), 4); // horizon actions
        assert_eq!(seqs[0].2.len(), 4); // horizon target states
    }

    #[test]
    fn test_split_is_deterministic() {
        let config = GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };

        let dataset = GpcDataset::generate_synthetic(config, 6, 12, 7);
        let first = dataset.split(0.25, 99).unwrap();
        let second = dataset.split(0.25, 99).unwrap();

        assert_eq!(first.train.num_episodes(), second.train.num_episodes());
        assert_eq!(
            first.validation.num_episodes(),
            second.validation.num_episodes()
        );
        assert_eq!(
            first.train.world_model_samples(),
            second.train.world_model_samples()
        );
        assert_eq!(
            first.validation.world_model_samples(),
            second.validation.world_model_samples()
        );
        assert_eq!(
            first.train.num_episodes() + first.validation.num_episodes(),
            dataset.num_episodes()
        );
    }
}
