//! Eval command implementation.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use clap::Args;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use gpc_compat::{
    CheckpointKind, load_metadata, load_policy_checkpoint, load_world_model_checkpoint,
};
use gpc_core::config::{GpcConfig, PolicyConfig, WorldModelConfig};
use gpc_core::traits::{Evaluator, RewardFunction, WorldModel};
use gpc_eval::{GpcOptBuilder, GpcRankBuilder};
use gpc_policy::DiffusionPolicyConfig;
use gpc_train::data::Episode;
use gpc_world::reward::L2RewardFunctionConfig;
use gpc_world::world_model::StateWorldModelConfig;

type EvalBackend = Autodiff<NdArray>;

/// Arguments for the eval command.
#[derive(Args, Debug, Clone)]
pub struct EvalArgs {
    /// Evaluation strategy: "rank" or "opt".
    #[arg(long, default_value = "rank")]
    strategy: String,

    /// Path to configuration file (JSON).
    #[arg(short, long)]
    config: Option<String>,

    /// Path to the checkpoint directory.
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,

    /// Explicit path to a policy checkpoint.
    #[arg(long)]
    policy_checkpoint: Option<String>,

    /// Explicit path to a world model checkpoint.
    #[arg(long)]
    world_model_checkpoint: Option<String>,

    /// Path to training data directory or episodes.json.
    #[arg(long)]
    data: Option<String>,

    /// Force synthetic evaluation data even when `--data` is set.
    #[arg(long)]
    synthetic: bool,

    /// Number of synthetic episodes to generate.
    #[arg(long, default_value = "12")]
    episodes: usize,

    /// Number of timesteps per synthetic episode.
    #[arg(long, default_value = "36")]
    episode_length: usize,

    /// Number of candidate trajectories (GPC-RANK).
    #[arg(long)]
    num_candidates: Option<usize>,

    /// Number of optimization steps (GPC-OPT).
    #[arg(long)]
    opt_steps: Option<usize>,

    /// Run a demo evaluation with random models.
    #[arg(long)]
    demo: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvaluationStrategy {
    Rank,
    Opt,
}

impl EvaluationStrategy {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "rank" => Ok(Self::Rank),
            "opt" => Ok(Self::Opt),
            other => anyhow::bail!("Unknown strategy: {other}. Use 'rank' or 'opt'."),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Rank => "rank",
            Self::Opt => "opt",
        }
    }
}

#[derive(Debug, Clone)]
struct LoadedModels<B: Backend> {
    policy: gpc_policy::DiffusionPolicy<B>,
    world_model: gpc_world::StateWorldModel<B>,
    policy_config: PolicyConfig,
    world_model_config: WorldModelConfig,
}

#[derive(Debug, Clone)]
struct EvaluationWindow {
    obs_history: Vec<Vec<f32>>,
    current_state: Vec<f32>,
    expert_actions: Vec<Vec<f32>>,
    target_states: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct EvaluationReport {
    strategy: EvaluationStrategy,
    windows_evaluated: usize,
    mean_rollout_mse: f32,
    mean_terminal_distance: f32,
    mean_action_mse: f32,
    mean_reward: f32,
    success_rate: f32,
    num_candidates: usize,
    opt_steps: usize,
}

/// Run the eval command.
pub fn run_eval(args: EvalArgs) -> Result<()> {
    let config = if let Some(config_path) = &args.config {
        let data = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&data)?
    } else {
        GpcConfig::default()
    };

    if args.demo {
        return run_eval_demo(&config, &args);
    }

    let report = run_checkpoint_eval(&args, &config)?;
    log_eval_report(&report);
    Ok(())
}

fn run_checkpoint_eval(args: &EvalArgs, config: &GpcConfig) -> Result<EvaluationReport> {
    let strategy = EvaluationStrategy::parse(&args.strategy)?;
    let device = <EvalBackend as Backend>::Device::default();
    let models = load_models(args, &device)?;
    let windows = load_evaluation_windows(args, &models)?;

    let num_candidates = args
        .num_candidates
        .unwrap_or(config.gpc_rank.num_candidates)
        .max(1);
    let opt_steps = args
        .opt_steps
        .unwrap_or(config.gpc_opt.num_opt_steps)
        .max(1);

    tracing::info!(
        strategy = strategy.label(),
        windows = windows.len(),
        policy_checkpoint = %resolved_policy_checkpoint_path(args).display(),
        world_model_checkpoint = %resolved_world_model_checkpoint_path(args).display(),
        "Starting checkpoint-backed evaluation"
    );

    let report = evaluate_windows(
        &models,
        &windows,
        strategy,
        num_candidates,
        opt_steps,
        &device,
    )?;

    Ok(report)
}

fn load_models(
    args: &EvalArgs,
    device: &<EvalBackend as Backend>::Device,
) -> Result<LoadedModels<EvalBackend>> {
    let policy_path = resolved_policy_checkpoint_path(args);
    let world_model_path = resolved_world_model_checkpoint_path(args);

    let policy_metadata = load_metadata(&policy_path)?;
    let world_model_metadata = load_metadata(&world_model_path)?;

    let policy_kind = CheckpointKind::from_metadata(&policy_metadata)?;
    let world_model_kind = CheckpointKind::from_metadata(&world_model_metadata)?;

    if policy_kind != CheckpointKind::Policy {
        anyhow::bail!(
            "policy checkpoint must contain a policy model, found {}",
            policy_metadata.model_type
        );
    }
    if world_model_kind != CheckpointKind::WorldModel {
        anyhow::bail!(
            "world model checkpoint must contain a world model, found {}",
            world_model_metadata.model_type
        );
    }

    let policy_config: PolicyConfig = serde_json::from_str(&policy_metadata.config_json)?;
    let world_model_config: WorldModelConfig =
        serde_json::from_str(&world_model_metadata.config_json)?;

    policy_config.validate()?;
    world_model_config.validate()?;

    if policy_config.action_dim != world_model_config.action_dim {
        anyhow::bail!(
            "checkpoint action dimensions do not match: policy={} world_model={}",
            policy_config.action_dim,
            world_model_config.action_dim
        );
    }

    let policy = load_policy_checkpoint::<EvalBackend>(&policy_path, device)?;
    let world_model = load_world_model_checkpoint::<EvalBackend>(&world_model_path, device)?;

    Ok(LoadedModels {
        policy,
        world_model,
        policy_config,
        world_model_config,
    })
}

fn load_evaluation_windows(
    args: &EvalArgs,
    models: &LoadedModels<EvalBackend>,
) -> Result<Vec<EvaluationWindow>> {
    let episodes = if args.synthetic || args.data.is_none() {
        let episode_length = args
            .episode_length
            .max(models.policy_config.pred_horizon + 1);
        generate_synthetic_episodes(
            models.world_model_config.state_dim,
            models.world_model_config.action_dim,
            models.policy_config.obs_dim,
            episode_length,
            args.episodes.max(1),
            current_seed(),
        )
    } else {
        load_episodes_from_path(args.data.as_deref().context("missing data path")?)?
    };

    let windows = build_windows(
        &episodes,
        models.policy_config.obs_horizon,
        models.policy_config.pred_horizon,
    );

    if windows.is_empty() {
        anyhow::bail!("no evaluation windows available for the selected dataset");
    }

    validate_episode_dimensions(&episodes, &models.policy_config, &models.world_model_config)?;

    Ok(windows)
}

fn evaluate_windows(
    models: &LoadedModels<EvalBackend>,
    windows: &[EvaluationWindow],
    strategy: EvaluationStrategy,
    num_candidates: usize,
    opt_steps: usize,
    device: &<EvalBackend as Backend>::Device,
) -> Result<EvaluationReport> {
    let mut windows_evaluated = 0usize;
    let mut rollout_mse_sum = 0.0_f32;
    let mut terminal_distance_sum = 0.0_f32;
    let mut action_mse_sum = 0.0_f32;
    let mut reward_sum = 0.0_f32;
    let mut success_count = 0usize;

    for window in windows {
        let obs_history = tensor_3d::<EvalBackend>(&window.obs_history, device);
        let current_state = tensor_2d::<EvalBackend>(&window.current_state, device);
        let expert_actions = tensor_3d::<EvalBackend>(&window.expert_actions, device);
        let target_states = tensor_3d::<EvalBackend>(&window.target_states, device);
        let goal_state = window
            .target_states
            .last()
            .context("missing target state")?
            .clone();
        let reward_fn_eval = L2RewardFunctionConfig {
            state_dim: models.world_model_config.state_dim,
        }
        .init::<EvalBackend>(device)
        .with_goal(tensor_1d::<EvalBackend>(&goal_state, device));
        let reward_fn_metric = L2RewardFunctionConfig {
            state_dim: models.world_model_config.state_dim,
        }
        .init::<EvalBackend>(device)
        .with_goal(tensor_1d::<EvalBackend>(&goal_state, device));

        let selected_actions = match strategy {
            EvaluationStrategy::Rank => GpcRankBuilder::new(
                models.policy.clone(),
                models.world_model.clone(),
                reward_fn_eval,
            )
            .num_candidates(num_candidates)
            .build()
            .select_action(&obs_history, &current_state, device)?,
            EvaluationStrategy::Opt => GpcOptBuilder::new(
                models.policy.clone(),
                models.world_model.clone(),
                reward_fn_eval,
            )
            .num_opt_steps(opt_steps)
            .build()
            .select_action(&obs_history, &current_state, device)?,
        };

        let predicted_states = models
            .world_model
            .rollout(&current_state, &selected_actions)?;
        let reward = reward_fn_metric.compute_reward(&predicted_states)?;

        let predicted_vec = tensor_to_vec(predicted_states)?;
        let target_vec = tensor_to_vec(target_states.clone())?;
        let selected_vec = tensor_to_vec(selected_actions)?;
        let expert_vec = tensor_to_vec(expert_actions)?;

        let rollout_mse = mean_squared_error(&predicted_vec, &target_vec);
        let terminal_distance = terminal_l2_distance(
            &predicted_vec,
            &target_vec,
            models.world_model_config.state_dim,
            models.policy_config.pred_horizon,
        );
        let action_mse = mean_squared_error(&selected_vec, &expert_vec);
        let reward_value: f32 = reward.into_scalar().elem();

        rollout_mse_sum += rollout_mse;
        terminal_distance_sum += terminal_distance;
        action_mse_sum += action_mse;
        reward_sum += reward_value;
        windows_evaluated += 1;
        if terminal_distance < 0.1 {
            success_count += 1;
        }
    }

    let denom = windows_evaluated as f32;
    Ok(EvaluationReport {
        strategy,
        windows_evaluated,
        mean_rollout_mse: rollout_mse_sum / denom,
        mean_terminal_distance: terminal_distance_sum / denom,
        mean_action_mse: action_mse_sum / denom,
        mean_reward: reward_sum / denom,
        success_rate: success_count as f32 / denom,
        num_candidates,
        opt_steps,
    })
}

fn run_eval_demo(config: &GpcConfig, args: &EvalArgs) -> Result<()> {
    let device = <EvalBackend as Backend>::Device::default();

    tracing::info!("Running evaluation demo with random models");

    let policy_config = DiffusionPolicyConfig {
        obs_dim: config.policy.obs_dim,
        action_dim: config.policy.action_dim,
        obs_horizon: config.policy.obs_horizon,
        pred_horizon: config.policy.pred_horizon,
        hidden_dim: 32,
        time_embed_dim: 16,
        num_res_blocks: 1,
        diffusion_steps: 24,
        beta_start: 1e-4,
        beta_end: 0.02,
    };
    let policy = policy_config.init::<EvalBackend>(&device);

    let world_config = StateWorldModelConfig {
        state_dim: config.world_model.state_dim,
        action_dim: config.world_model.action_dim,
        hidden_dim: 32,
        num_layers: 1,
    };
    let world_model = world_config.init::<EvalBackend>(&device);

    let reward_config = L2RewardFunctionConfig {
        state_dim: config.world_model.state_dim,
    };
    let reward_fn = reward_config.init::<EvalBackend>(&device);

    let obs = Tensor::<EvalBackend, 3>::zeros(
        [1, config.policy.obs_horizon, config.policy.obs_dim],
        &device,
    );
    let state = Tensor::<EvalBackend, 2>::zeros([1, config.world_model.state_dim], &device);

    match args.strategy.as_str() {
        "rank" => {
            let k = args
                .num_candidates
                .unwrap_or(config.gpc_rank.num_candidates);
            tracing::info!("GPC-RANK with K={k} candidates");

            let evaluator = GpcRankBuilder::new(policy, world_model, reward_fn)
                .num_candidates(k)
                .build();

            let best = evaluator.select_action(&obs, &state, &device)?;
            tracing::info!("Best action shape: {:?}", best.dims());
        }
        "opt" => {
            let steps = args.opt_steps.unwrap_or(config.gpc_opt.num_opt_steps);
            tracing::info!("GPC-OPT with {steps} optimization steps");

            let evaluator = GpcOptBuilder::new(policy, world_model, reward_fn)
                .num_opt_steps(steps)
                .build();

            let best = evaluator.select_action(&obs, &state, &device)?;
            tracing::info!("Optimized action shape: {:?}", best.dims());
        }
        other => {
            anyhow::bail!("Unknown strategy: {other}. Use 'rank' or 'opt'.");
        }
    }

    tracing::info!("Evaluation demo complete");
    Ok(())
}

fn load_episodes_from_path(path: &str) -> Result<Vec<Episode>> {
    let path = Path::new(path);
    let json_path = if path.is_file() {
        path.to_path_buf()
    } else {
        path.join("episodes.json")
    };

    let data = std::fs::read_to_string(&json_path)
        .with_context(|| format!("failed to read dataset from {}", json_path.display()))?;
    let episodes = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse episodes from {}", json_path.display()))?;
    Ok(episodes)
}

fn generate_synthetic_episodes(
    state_dim: usize,
    action_dim: usize,
    obs_dim: usize,
    episode_length: usize,
    num_episodes: usize,
    seed: u64,
) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut episodes = Vec::with_capacity(num_episodes);

    for _ in 0..num_episodes {
        let mut states = Vec::with_capacity(episode_length);
        let mut actions = Vec::with_capacity(episode_length.saturating_sub(1));
        let mut observations = Vec::with_capacity(episode_length);
        let mut state: Vec<f32> = (0..state_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        for timestep in 0..episode_length {
            states.push(state.clone());
            observations.push(state[..obs_dim.min(state_dim)].to_vec());

            if timestep + 1 < episode_length {
                let action: Vec<f32> = (0..action_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                state = state
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| {
                        let action_term = if index < action_dim {
                            action[index]
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

    episodes
}

fn build_windows(
    episodes: &[Episode],
    obs_horizon: usize,
    pred_horizon: usize,
) -> Vec<EvaluationWindow> {
    let mut windows = Vec::new();

    for episode in episodes {
        if episode.actions.len() < pred_horizon || episode.observations.len() < obs_horizon {
            continue;
        }

        for start in 0..=(episode.actions.len() - pred_horizon) {
            let obs_start = if start >= obs_horizon {
                start + 1 - obs_horizon
            } else {
                0
            };
            if start + 1 + pred_horizon > episode.states.len() {
                continue;
            }
            if obs_start + obs_horizon > episode.observations.len() {
                continue;
            }
            let mut obs_history = episode.observations[obs_start..obs_start + obs_horizon].to_vec();
            while obs_history.len() < obs_horizon {
                obs_history.insert(0, obs_history[0].clone());
            }

            windows.push(EvaluationWindow {
                obs_history,
                current_state: episode.states[start].clone(),
                expert_actions: episode.actions[start..start + pred_horizon].to_vec(),
                target_states: episode.states[start + 1..start + 1 + pred_horizon].to_vec(),
            });
        }
    }

    windows
}

fn validate_episode_dimensions(
    episodes: &[Episode],
    policy_config: &PolicyConfig,
    world_model_config: &WorldModelConfig,
) -> Result<()> {
    let Some(episode) = episodes.first() else {
        anyhow::bail!("dataset does not contain any episodes");
    };

    let state_dim = episode
        .states
        .first()
        .map(|state| state.len())
        .context("episode is missing state samples")?;
    let action_dim = episode
        .actions
        .first()
        .map(|action| action.len())
        .context("episode is missing action samples")?;
    let obs_dim = episode
        .observations
        .first()
        .map(|obs| obs.len())
        .context("episode is missing observation samples")?;

    if state_dim != world_model_config.state_dim {
        anyhow::bail!(
            "state dimension mismatch: dataset={} checkpoint={}",
            state_dim,
            world_model_config.state_dim
        );
    }
    if action_dim != policy_config.action_dim || action_dim != world_model_config.action_dim {
        anyhow::bail!(
            "action dimension mismatch: dataset={} policy={} world_model={}",
            action_dim,
            policy_config.action_dim,
            world_model_config.action_dim
        );
    }
    if obs_dim != policy_config.obs_dim {
        anyhow::bail!(
            "observation dimension mismatch: dataset={} checkpoint={}",
            obs_dim,
            policy_config.obs_dim
        );
    }

    Ok(())
}

fn resolved_policy_checkpoint_path(args: &EvalArgs) -> PathBuf {
    args.policy_checkpoint
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_policy_checkpoint_path(args))
}

fn resolved_world_model_checkpoint_path(args: &EvalArgs) -> PathBuf {
    args.world_model_checkpoint
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_world_model_checkpoint_path(args))
}

fn default_policy_checkpoint_path(args: &EvalArgs) -> PathBuf {
    PathBuf::from(&args.checkpoint_dir).join("policy_final.bin")
}

fn default_world_model_checkpoint_path(args: &EvalArgs) -> PathBuf {
    PathBuf::from(&args.checkpoint_dir).join("world_model_final.bin")
}

fn log_eval_report(report: &EvaluationReport) {
    tracing::info!("Checkpoint evaluation complete");
    tracing::info!("  strategy: {}", report.strategy.label());
    tracing::info!("  windows: {}", report.windows_evaluated);
    tracing::info!("  mean rollout MSE: {:.6}", report.mean_rollout_mse);
    tracing::info!(
        "  mean terminal distance: {:.6}",
        report.mean_terminal_distance
    );
    tracing::info!("  mean action MSE: {:.6}", report.mean_action_mse);
    tracing::info!("  mean reward: {:.6}", report.mean_reward);
    tracing::info!("  success rate: {:.2}%", report.success_rate * 100.0);
    tracing::info!("  num candidates: {}", report.num_candidates);
    tracing::info!("  opt steps: {}", report.opt_steps);
}

fn tensor_1d<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats(values, device)
}

fn tensor_2d<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(values, device).reshape([1, values.len()])
}

fn tensor_3d<B: Backend>(values: &[Vec<f32>], device: &B::Device) -> Tensor<B, 3> {
    let outer = values.len();
    let inner = values
        .first()
        .map(|row| row.len())
        .expect("evaluation windows are non-empty");
    let flat = values
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([1, outer, inner])
}

fn tensor_to_vec<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>> {
    Ok(tensor
        .into_data()
        .to_vec()
        .expect("tensor data should always be contiguous"))
}

fn mean_squared_error(prediction: &[f32], target: &[f32]) -> f32 {
    assert_eq!(prediction.len(), target.len());
    prediction
        .iter()
        .zip(target.iter())
        .map(|(pred, tgt)| {
            let diff = pred - tgt;
            diff * diff
        })
        .sum::<f32>()
        / prediction.len() as f32
}

fn terminal_l2_distance(
    prediction: &[f32],
    target: &[f32],
    state_dim: usize,
    horizon: usize,
) -> f32 {
    let start = (horizon - 1) * state_dim;
    prediction[start..start + state_dim]
        .iter()
        .zip(&target[start..start + state_dim])
        .map(|(pred, tgt)| {
            let diff = pred - tgt;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn current_seed() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs() ^ u64::from(duration.subsec_nanos()),
        Err(_) => 42,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpc_compat::{
        CheckpointFormat, CheckpointMetadata, save_policy_checkpoint, save_world_model_checkpoint,
    };
    use gpc_core::config::TrainingConfig;
    use gpc_train::{PolicyTrainer, WorldModelTrainer};

    fn temp_eval_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gpc_eval_{name}_{}_{}",
            std::process::id(),
            current_seed()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn current_timestamp() -> String {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => format!("{}Z", duration.as_secs()),
            Err(_) => "0Z".to_string(),
        }
    }

    fn checkpoint_metadata(
        model_type: CheckpointKind,
        epoch: Option<usize>,
        loss: Option<f32>,
        config_json: String,
    ) -> CheckpointMetadata {
        CheckpointMetadata {
            model_type: model_type.as_str().to_string(),
            epoch: epoch.unwrap_or_default(),
            loss: loss.map(f64::from).unwrap_or_default(),
            timestamp: current_timestamp(),
            config_json,
        }
    }

    fn save_tiny_checkpoints(
        dir: &Path,
    ) -> Result<(PathBuf, PathBuf, PolicyConfig, WorldModelConfig)> {
        let device = <EvalBackend as Backend>::Device::default();
        let dataset = generate_synthetic_episodes(4, 2, 4, 10, 4, 42);
        let dataset_config = gpc_train::data::GpcDatasetConfig {
            data_dir: "synthetic".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };
        let gpc_dataset = gpc_train::data::GpcDataset::new(dataset, dataset_config);

        let training_config = TrainingConfig {
            num_epochs: 1,
            batch_size: 4,
            learning_rate: 1e-3,
            log_every: 1,
            ..Default::default()
        };

        let policy_config = PolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 4,
            action_horizon: 1,
            hidden_dim: 16,
            num_res_blocks: 1,
            noise_schedule: gpc_core::config::NoiseScheduleConfig {
                num_timesteps: 8,
                beta_start: 1e-4,
                beta_end: 0.02,
            },
        };
        let world_model_config = WorldModelConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_dim: 16,
            num_layers: 1,
            dropout: 0.0,
        };

        let policy = PolicyTrainer::new(training_config.clone(), policy_config.clone())
            .train_with_summary::<EvalBackend>(&gpc_dataset, &device)
            .model;
        let world_model = WorldModelTrainer::new(training_config, world_model_config.clone())
            .train_phase1_with_summary::<EvalBackend>(&gpc_dataset, &device)
            .model;

        let policy_path = save_policy_checkpoint(
            policy,
            &checkpoint_metadata(
                CheckpointKind::Policy,
                Some(1),
                Some(0.0),
                serde_json::to_string(&policy_config)?,
            ),
            dir.join("policy_final"),
            CheckpointFormat::Bin,
        )?
        .checkpoint_path;

        let world_model_path = save_world_model_checkpoint(
            world_model,
            &checkpoint_metadata(
                CheckpointKind::WorldModel,
                Some(1),
                Some(0.0),
                serde_json::to_string(&world_model_config)?,
            ),
            dir.join("world_model_final"),
            CheckpointFormat::Bin,
        )?
        .checkpoint_path;

        Ok((
            policy_path,
            world_model_path,
            policy_config,
            world_model_config,
        ))
    }

    #[test]
    fn checkpoint_eval_runs_end_to_end() {
        let dir = temp_eval_dir("end_to_end");
        let (policy_path, world_model_path, _policy_config, _world_model_config) =
            save_tiny_checkpoints(&dir).unwrap();

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: true,
            episodes: 4,
            episode_length: 10,
            num_candidates: Some(4),
            opt_steps: Some(2),
            demo: false,
        };
        let config = GpcConfig::default();

        let report = run_checkpoint_eval(&args, &config).unwrap();

        assert_eq!(report.strategy, EvaluationStrategy::Rank);
        assert!(report.windows_evaluated > 0);
        assert!(report.mean_rollout_mse.is_finite());
        assert!(report.mean_terminal_distance.is_finite());
        assert!(report.mean_action_mse.is_finite());
        assert!(report.mean_reward.is_finite());
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        assert_eq!(report.num_candidates, 4);
        assert_eq!(report.opt_steps, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_windows_match_checkpoint_dims() {
        let episodes = generate_synthetic_episodes(4, 2, 4, 8, 2, 42);
        let windows = build_windows(&episodes, 2, 4);

        assert!(!windows.is_empty());
        assert_eq!(windows[0].obs_history.len(), 2);
        assert_eq!(windows[0].current_state.len(), 4);
        assert_eq!(windows[0].expert_actions.len(), 4);
        assert_eq!(windows[0].target_states.len(), 4);
    }
}
