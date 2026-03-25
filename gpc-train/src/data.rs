//! Dataset and data loading for GPC training.

use std::path::{Path, PathBuf};

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use gpc_core::config::{PolicyConfig, WorldModelConfig};

/// A world model sequence sample: `(initial_state, action_sequence, target_states)`.
pub type WorldModelTransitionSample = (Vec<f32>, Vec<f32>, Vec<f32>);

/// A world model sequence sample: `(initial_state, action_sequence, target_states)`.
pub type WorldModelSequenceSample = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>);

/// A policy sample: `(observation_history, action_sequence)`.
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    /// Sequence of states `[timesteps, state_dim]`.
    pub states: Vec<Vec<f32>>,
    /// Sequence of actions `[timesteps - 1, action_dim]`.
    pub actions: Vec<Vec<f32>>,
    /// Sequence of observations `[timesteps, obs_dim]`.
    pub observations: Vec<Vec<f32>>,
}

/// Summary of dataset validation against policy and world-model configs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetValidationReport {
    /// Number of episodes in the dataset.
    pub episode_count: usize,
    /// Total number of transitions across all episodes.
    pub transition_count: usize,
    /// Number of usable open-loop windows for policy/world-model evaluation.
    pub open_loop_window_count: usize,
    /// Number of episodes structurally compatible with closed-loop evaluation.
    pub closed_loop_compatible_episode_count: usize,
}

impl DatasetValidationReport {
    /// Returns `true` when at least one open-loop evaluation window is available.
    pub fn has_usable_open_loop_windows(&self) -> bool {
        self.open_loop_window_count > 0
    }

    /// Returns `true` when every episode is closed-loop compatible.
    pub fn is_closed_loop_compatible(&self) -> bool {
        self.episode_count > 0 && self.closed_loop_compatible_episode_count == self.episode_count
    }
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

    /// Load dataset from a directory containing `episodes.json` or from a direct JSON file.
    pub fn from_path(path: impl AsRef<Path>, config: GpcDatasetConfig) -> gpc_core::Result<Self> {
        let episodes = load_episodes_from_path(path)?;
        Ok(Self::new(episodes, config))
    }

    /// Load dataset from a JSON file or dataset directory.
    pub fn from_json(path: impl AsRef<Path>, config: GpcDatasetConfig) -> gpc_core::Result<Self> {
        Self::from_path(path, config)
    }

    /// Validate the dataset against policy and world model configs.
    pub fn validate(
        &self,
        policy_config: &PolicyConfig,
        world_model_config: &WorldModelConfig,
    ) -> gpc_core::Result<DatasetValidationReport> {
        validate_episodes(&self.episodes, policy_config, world_model_config)
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
            let mut actions = Vec::with_capacity(episode_length.saturating_sub(1));
            let mut observations = Vec::with_capacity(episode_length);

            let mut state: Vec<f32> = (0..config.state_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            for timestep in 0..episode_length {
                states.push(state.clone());
                observations.push(state[..config.obs_dim.min(config.state_dim)].to_vec());

                if timestep < episode_length.saturating_sub(1) {
                    let action: Vec<f32> = (0..config.action_dim)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect();

                    state = state
                        .iter()
                        .enumerate()
                        .map(|(i, &value)| {
                            let action_term = if i < config.action_dim {
                                action[i]
                            } else {
                                0.0
                            };
                            value + 0.1 * action_term + rng.gen_range(-0.01..0.01)
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

/// Resolve a dataset directory or file path to an `episodes.json` file.
pub fn resolve_dataset_path(path: impl AsRef<Path>) -> gpc_core::Result<PathBuf> {
    let path = path.as_ref();

    if path.is_dir() {
        let episodes_path = path.join("episodes.json");
        if episodes_path.exists() {
            return Ok(episodes_path);
        }

        return Err(gpc_core::GpcError::DataLoading(format!(
            "dataset directory {} does not contain episodes.json",
            path.display()
        )));
    }

    if path.exists() {
        return Ok(path.to_path_buf());
    }

    Err(gpc_core::GpcError::DataLoading(format!(
        "dataset path {} does not exist",
        path.display()
    )))
}

/// Load a dataset's episodes from a directory or direct JSON file.
pub fn load_episodes_from_path(path: impl AsRef<Path>) -> gpc_core::Result<Vec<Episode>> {
    let path = resolve_dataset_path(path)?;
    let data = std::fs::read_to_string(&path).map_err(|error| {
        gpc_core::GpcError::DataLoading(format!(
            "failed to read dataset from {}: {error}",
            path.display()
        ))
    })?;

    serde_json::from_str(&data).map_err(|error| {
        gpc_core::GpcError::DataLoading(format!(
            "failed to parse episodes from {}: {error}",
            path.display()
        ))
    })
}

/// Validate loaded episodes against the policy and world-model configs.
pub fn validate_episodes(
    episodes: &[Episode],
    policy_config: &PolicyConfig,
    world_model_config: &WorldModelConfig,
) -> gpc_core::Result<DatasetValidationReport> {
    let mut report = DatasetValidationReport {
        episode_count: episodes.len(),
        transition_count: 0,
        open_loop_window_count: 0,
        closed_loop_compatible_episode_count: 0,
    };

    for (episode_index, episode) in episodes.iter().enumerate() {
        validate_episode(episode, episode_index, policy_config, world_model_config)?;
        report.transition_count += episode.actions.len();
        report.open_loop_window_count += episode
            .states
            .len()
            .saturating_sub(policy_config.pred_horizon);
        report.closed_loop_compatible_episode_count += 1;
    }

    Ok(report)
}

fn validate_episode(
    episode: &Episode,
    episode_index: usize,
    policy_config: &PolicyConfig,
    world_model_config: &WorldModelConfig,
) -> gpc_core::Result<()> {
    if episode.states.is_empty() {
        return Err(gpc_core::GpcError::DataLoading(format!(
            "episode {episode_index} is missing state samples"
        )));
    }
    if episode.actions.is_empty() {
        return Err(gpc_core::GpcError::DataLoading(format!(
            "episode {episode_index} is missing action samples"
        )));
    }
    if episode.observations.is_empty() {
        return Err(gpc_core::GpcError::DataLoading(format!(
            "episode {episode_index} is missing observation samples"
        )));
    }

    if episode.states.len() != episode.observations.len() {
        return Err(gpc_core::GpcError::DataLoading(format!(
            "episode {episode_index} must have the same number of states and observations"
        )));
    }
    if episode.states.len() != episode.actions.len() + 1 {
        return Err(gpc_core::GpcError::DataLoading(format!(
            "episode {episode_index} must contain exactly one more state than action"
        )));
    }

    let action_dim = episode
        .actions
        .first()
        .map(|action| action.len())
        .expect("episode actions checked to be non-empty");
    if action_dim != policy_config.action_dim || action_dim != world_model_config.action_dim {
        return Err(gpc_core::GpcError::DataLoading(format!(
            "episode {episode_index} action dimension mismatch: dataset={} policy={} world_model={}",
            action_dim, policy_config.action_dim, world_model_config.action_dim
        )));
    }

    validate_sequence_shapes(
        &episode.states,
        world_model_config.state_dim,
        "state",
        episode_index,
        "world model",
    )?;
    validate_sequence_shapes(
        &episode.actions,
        policy_config.action_dim,
        "action",
        episode_index,
        "policy/world model",
    )?;
    validate_sequence_shapes(
        &episode.observations,
        policy_config.obs_dim,
        "observation",
        episode_index,
        "policy",
    )?;

    Ok(())
}

fn validate_sequence_shapes(
    rows: &[Vec<f32>],
    expected_dim: usize,
    label: &str,
    episode_index: usize,
    context: &str,
) -> gpc_core::Result<()> {
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != expected_dim {
            return Err(gpc_core::GpcError::DimensionMismatch {
                context: format!("episode {episode_index} {label} {row_index} ({context})"),
                expected: expected_dim,
                got: row.len(),
            });
        }
    }

    Ok(())
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dataset_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gpc_train_{name}_{}_{}",
            std::process::id(),
            test_suffix()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_suffix() -> u64 {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => duration.as_secs() ^ u64::from(duration.subsec_nanos()),
            Err(_) => 42,
        }
    }

    fn write_episode_file(dir: &Path, episodes: &[Episode]) -> PathBuf {
        let path = dir.join("episodes.json");
        std::fs::write(&path, serde_json::to_string_pretty(episodes).unwrap()).unwrap();
        path
    }

    fn valid_policy_config() -> GpcDatasetConfig {
        GpcDatasetConfig {
            data_dir: "test".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 3,
        }
    }

    fn valid_policy_world_configs() -> (PolicyConfig, WorldModelConfig) {
        let policy = PolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 3,
            action_horizon: 1,
            hidden_dim: 16,
            num_res_blocks: 1,
            noise_schedule: gpc_core::config::NoiseScheduleConfig {
                num_timesteps: 8,
                beta_start: 1e-4,
                beta_end: 0.02,
            },
        };
        let world = WorldModelConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_dim: 16,
            num_layers: 1,
            dropout: 0.0,
        };
        (policy, world)
    }

    fn valid_episode() -> Episode {
        Episode {
            states: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![0.2, 0.0, 0.0, 0.0],
                vec![0.3, 0.0, 0.0, 0.0],
            ],
            actions: vec![vec![0.1, 0.0], vec![0.1, 0.0], vec![0.1, 0.0]],
            observations: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![0.2, 0.0, 0.0, 0.0],
                vec![0.3, 0.0, 0.0, 0.0],
            ],
        }
    }

    #[test]
    fn test_resolve_dataset_path_supports_direct_file_and_directory() {
        let dir = temp_dataset_dir("resolve");
        let episodes_path = write_episode_file(&dir, &[valid_episode()]);

        assert_eq!(resolve_dataset_path(&dir).unwrap(), episodes_path,);
        assert_eq!(resolve_dataset_path(&episodes_path).unwrap(), episodes_path,);
    }

    #[test]
    fn test_load_episodes_from_path_supports_direct_file_and_directory() {
        let dir = temp_dataset_dir("load");
        let episodes = vec![valid_episode()];
        let episodes_path = write_episode_file(&dir, &episodes);

        let loaded_from_dir = load_episodes_from_path(&dir).unwrap();
        let loaded_from_file = load_episodes_from_path(&episodes_path).unwrap();

        assert_eq!(loaded_from_dir, episodes);
        assert_eq!(loaded_from_file, episodes);
    }

    #[test]
    fn test_dataset_from_path_supports_direct_file_and_directory() {
        let dir = temp_dataset_dir("dataset");
        let config = valid_policy_config();
        write_episode_file(&dir, &[valid_episode()]);

        let dataset_from_dir = GpcDataset::from_path(&dir, config.clone()).unwrap();
        let dataset_from_file = GpcDataset::from_json(dir.join("episodes.json"), config).unwrap();

        assert_eq!(dataset_from_dir.num_episodes(), 1);
        assert_eq!(dataset_from_file.num_episodes(), 1);
    }

    #[test]
    fn test_validate_episodes_reports_open_loop_and_closed_loop_compatibility() {
        let (policy, world) = valid_policy_world_configs();
        let episodes = vec![valid_episode(), valid_episode()];

        let report = validate_episodes(&episodes, &policy, &world).unwrap();

        assert_eq!(report.episode_count, 2);
        assert_eq!(report.transition_count, 6);
        assert_eq!(report.closed_loop_compatible_episode_count, 2);
        assert!(report.has_usable_open_loop_windows());
        assert!(report.is_closed_loop_compatible());
        assert_eq!(report.open_loop_window_count, 2);
    }

    #[test]
    fn test_validate_episodes_reports_unusable_open_loop_windows() {
        let (mut policy, world) = valid_policy_world_configs();
        policy.pred_horizon = 8;
        let episodes = vec![valid_episode()];

        let report = validate_episodes(&episodes, &policy, &world).unwrap();

        assert_eq!(report.open_loop_window_count, 0);
        assert!(!report.has_usable_open_loop_windows());
        assert!(report.is_closed_loop_compatible());
    }

    #[test]
    fn test_validate_episodes_rejects_missing_state_samples() {
        let (policy, world) = valid_policy_world_configs();
        let mut episode = valid_episode();
        episode.states.clear();

        let err = validate_episodes(&[episode], &policy, &world).unwrap_err();
        assert!(err.to_string().contains("missing state samples"));
    }

    #[test]
    fn test_validate_episodes_rejects_missing_action_samples() {
        let (policy, world) = valid_policy_world_configs();
        let mut episode = valid_episode();
        episode.actions.clear();
        episode.states.truncate(1);
        episode.observations.truncate(1);

        let err = validate_episodes(&[episode], &policy, &world).unwrap_err();
        assert!(err.to_string().contains("missing action samples"));
    }

    #[test]
    fn test_validate_episodes_rejects_missing_observation_samples() {
        let (policy, world) = valid_policy_world_configs();
        let mut episode = valid_episode();
        episode.observations.clear();

        let err = validate_episodes(&[episode], &policy, &world).unwrap_err();
        assert!(err.to_string().contains("missing observation samples"));
    }

    #[test]
    fn test_validate_episodes_rejects_dimension_mismatch() {
        let (policy, world) = valid_policy_world_configs();
        let mut episode = valid_episode();
        episode.states[0] = vec![0.0, 0.0];

        let err = validate_episodes(&[episode], &policy, &world).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("Dimension mismatch"));
        assert!(message.contains("episode 0 state 0"));
    }

    #[test]
    fn test_validate_episodes_rejects_world_model_action_dimension_mismatch() {
        let (policy, mut world) = valid_policy_world_configs();
        world.action_dim = 3;
        let episodes = vec![valid_episode()];

        let err = validate_episodes(&episodes, &policy, &world).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("action dimension mismatch"));
        assert!(message.contains("policy=2"));
        assert!(message.contains("world_model=3"));
    }
}
