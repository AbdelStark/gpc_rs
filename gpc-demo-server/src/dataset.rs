use serde::{Deserialize, Serialize};

pub type WorldModelSequenceSample = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>);
pub type PolicySampleData = (Vec<Vec<f32>>, Vec<Vec<f32>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub obs_dim: usize,
    pub obs_horizon: usize,
    pub pred_horizon: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode {
    pub states: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub observations: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct DemoDataset {
    episodes: Vec<Episode>,
    config: DatasetConfig,
}

impl DemoDataset {
    pub fn new(episodes: Vec<Episode>, config: DatasetConfig) -> Self {
        Self { episodes, config }
    }

    pub fn num_transitions(&self) -> usize {
        self.episodes
            .iter()
            .map(|episode| episode.actions.len())
            .sum()
    }

    pub fn world_model_samples(&self) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut samples = Vec::new();

        for episode in &self.episodes {
            for timestep in 0..episode.actions.len() {
                samples.push((
                    episode.states[timestep].clone(),
                    episode.actions[timestep].clone(),
                    episode.states[timestep + 1].clone(),
                ));
            }
        }

        samples
    }

    pub fn world_model_sequences(&self, horizon: usize) -> Vec<WorldModelSequenceSample> {
        let mut samples = Vec::new();

        for episode in &self.episodes {
            if episode.actions.len() < horizon {
                continue;
            }

            for start in 0..=(episode.actions.len() - horizon) {
                samples.push((
                    episode.states[start].clone(),
                    episode.actions[start..start + horizon].to_vec(),
                    episode.states[start + 1..start + 1 + horizon].to_vec(),
                ));
            }
        }

        samples
    }

    pub fn policy_samples(&self) -> Vec<PolicySampleData> {
        let mut samples = Vec::new();
        let obs_horizon = self.config.obs_horizon;
        let pred_horizon = self.config.pred_horizon;

        for episode in &self.episodes {
            if episode.observations.len() < obs_horizon || episode.actions.len() < pred_horizon {
                continue;
            }

            for start in 0..=(episode.actions.len() - pred_horizon) {
                if start + obs_horizon > episode.observations.len() {
                    continue;
                }

                let obs_start = if start >= obs_horizon {
                    start - obs_horizon + 1
                } else {
                    0
                };
                let mut obs_history =
                    episode.observations[obs_start..obs_start + obs_horizon].to_vec();

                while obs_history.len() < obs_horizon {
                    obs_history.insert(0, obs_history[0].clone());
                }

                samples.push((
                    obs_history,
                    episode.actions[start..start + pred_horizon].to_vec(),
                ));
            }
        }

        samples
    }
}
