//! Eval command implementation.

use std::path::{Path, PathBuf};

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
use gpc_core::traits::{Evaluator, Policy, RewardFunction, WorldModel};
use gpc_eval::{AutodiffGpcOptBuilder, GpcRankBuilder};
use gpc_policy::DiffusionPolicyConfig;
use gpc_train::data::Episode;
use gpc_world::reward::L2RewardFunctionConfig;
use gpc_world::world_model::StateWorldModelConfig;

use super::reporting::write_json_report;

pub(crate) type EvalBackend = Autodiff<NdArray>;

/// Arguments for the eval command.
#[derive(Args, Debug, Clone)]
pub struct EvalArgs {
    /// Evaluation strategy: "policy", "rank", or "opt".
    #[arg(long, default_value = "rank")]
    pub(crate) strategy: String,

    /// Path to configuration file (JSON).
    #[arg(short, long)]
    pub(crate) config: Option<String>,

    /// Path to the checkpoint directory.
    #[arg(long, default_value = "checkpoints")]
    pub(crate) checkpoint_dir: String,

    /// Explicit path to a policy checkpoint.
    #[arg(long)]
    pub(crate) policy_checkpoint: Option<String>,

    /// Explicit path to a world model checkpoint.
    #[arg(long)]
    pub(crate) world_model_checkpoint: Option<String>,

    /// Path to training data directory or episodes.json.
    #[arg(long)]
    pub(crate) data: Option<String>,

    /// Force synthetic evaluation data even when `--data` is set.
    #[arg(long)]
    pub(crate) synthetic: bool,

    /// Number of synthetic episodes to generate.
    #[arg(long, default_value = "12")]
    pub(crate) episodes: usize,

    /// Number of timesteps per synthetic episode.
    #[arg(long, default_value = "36")]
    pub(crate) episode_length: usize,

    /// Number of candidate trajectories (GPC-RANK).
    #[arg(long)]
    pub(crate) num_candidates: Option<usize>,

    /// Number of optimization steps (GPC-OPT).
    #[arg(long)]
    pub(crate) opt_steps: Option<usize>,

    /// Override the GPC-OPT learning rate.
    #[arg(long)]
    pub(crate) opt_learning_rate: Option<f32>,

    /// Seed used for synthetic evaluation data.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// Optional output path for a machine-readable JSON summary.
    #[arg(short, long)]
    pub(crate) output: Option<String>,

    /// Optional output path for detailed per-window or per-episode telemetry.
    #[arg(long)]
    pub(crate) details_output: Option<String>,

    /// Run a demo evaluation with random models.
    #[arg(long)]
    pub(crate) demo: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EvaluationStrategy {
    Policy,
    Rank,
    Opt,
}

impl EvaluationStrategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "policy" => Ok(Self::Policy),
            "rank" => Ok(Self::Rank),
            "opt" => Ok(Self::Opt),
            other => anyhow::bail!("Unknown strategy: {other}. Use 'policy', 'rank', or 'opt'."),
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Policy => "policy",
            Self::Rank => "rank",
            Self::Opt => "opt",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedModels<B: Backend> {
    pub(crate) policy: gpc_policy::DiffusionPolicy<B>,
    pub(crate) world_model: gpc_world::StateWorldModel<B>,
    pub(crate) policy_config: PolicyConfig,
    pub(crate) world_model_config: WorldModelConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct EvaluationWindow {
    episode_index: usize,
    window_index: usize,
    obs_history: Vec<Vec<f32>>,
    current_state: Vec<f32>,
    expert_actions: Vec<Vec<f32>>,
    target_states: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct EvaluationReport {
    pub(crate) strategy: EvaluationStrategy,
    pub(crate) mode: EvaluationMode,
    pub(crate) windows_evaluated: usize,
    pub(crate) mean_rollout_mse: f32,
    pub(crate) mean_terminal_distance: f32,
    pub(crate) mean_action_mse: f32,
    pub(crate) mean_reward: f32,
    pub(crate) success_rate: f32,
    pub(crate) num_candidates: usize,
    pub(crate) opt_steps: usize,
    pub(crate) opt_learning_rate: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum EvaluationMode {
    DatasetOpenLoop,
    SyntheticClosedLoop,
}

impl EvaluationMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::DatasetOpenLoop => "dataset-open-loop",
            Self::SyntheticClosedLoop => "synthetic-closed-loop",
        }
    }

    fn unit_label(self) -> &'static str {
        match self {
            Self::DatasetOpenLoop => "windows",
            Self::SyntheticClosedLoop => "episodes",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ClosedLoopEpisodeMetrics {
    pub(crate) rollout_mse: f32,
    pub(crate) terminal_distance: f32,
    pub(crate) action_mse: f32,
    pub(crate) reward: f32,
    pub(crate) success: bool,
    pub(crate) replans: usize,
    pub(crate) executed_steps: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CheckpointEvalSettings {
    pub(crate) strategy: EvaluationStrategy,
    pub(crate) num_candidates: usize,
    pub(crate) opt_steps: usize,
    pub(crate) opt_learning_rate: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct EvaluationSummaryArtifact {
    pub(crate) strategy: EvaluationStrategy,
    pub(crate) mode: EvaluationMode,
    pub(crate) checkpoint_dir: String,
    pub(crate) policy_checkpoint: String,
    pub(crate) world_model_checkpoint: String,
    pub(crate) data: Option<String>,
    pub(crate) synthetic: bool,
    pub(crate) seed: Option<u64>,
    pub(crate) report: EvaluationReport,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct EvaluationDetailsArtifact {
    pub(crate) summary: EvaluationSummaryArtifact,
    pub(crate) windows: Vec<DatasetWindowDetail>,
    pub(crate) episodes: Vec<SyntheticEpisodeDetail>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct DatasetWindowDetail {
    pub(crate) episode_index: usize,
    pub(crate) window_index: usize,
    pub(crate) rollout_mse: f32,
    pub(crate) terminal_distance: f32,
    pub(crate) action_mse: f32,
    pub(crate) reward: f32,
    pub(crate) success: bool,
    pub(crate) selected_actions: Vec<Vec<f32>>,
    pub(crate) predicted_states: Vec<Vec<f32>>,
    pub(crate) target_states: Vec<Vec<f32>>,
    pub(crate) strategy_trace: StrategyTraceDetail,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct SyntheticEpisodeDetail {
    pub(crate) episode_index: usize,
    pub(crate) goal_state: Vec<f32>,
    pub(crate) rollout_mse: f32,
    pub(crate) terminal_distance: f32,
    pub(crate) action_mse: f32,
    pub(crate) reward: f32,
    pub(crate) success: bool,
    pub(crate) replans: usize,
    pub(crate) executed_steps: usize,
    pub(crate) replans_trace: Vec<SyntheticReplanDetail>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct SyntheticReplanDetail {
    pub(crate) replan_index: usize,
    pub(crate) start_action_index: usize,
    pub(crate) executed_steps: usize,
    pub(crate) selected_actions: Vec<Vec<f32>>,
    pub(crate) predicted_states: Vec<Vec<f32>>,
    pub(crate) executed_actions: Vec<Vec<f32>>,
    pub(crate) executed_states: Vec<Vec<f32>>,
    pub(crate) strategy_trace: StrategyTraceDetail,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "strategy", rename_all = "lowercase")]
pub(crate) enum StrategyTraceDetail {
    Policy {},
    Rank {
        best_index: usize,
        candidate_rewards: Vec<f32>,
        num_candidates: usize,
    },
    Opt {
        initial_actions: Vec<Vec<f32>>,
        step_rewards: Vec<f32>,
        epsilon: f32,
        learning_rate: f32,
        num_opt_steps: usize,
    },
}

#[derive(Debug, Clone)]
struct SelectedTrajectory<B: Backend> {
    actions: Tensor<B, 3>,
    strategy_trace: StrategyTraceDetail,
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

    let details = if args.details_output.is_some() {
        Some(run_checkpoint_eval_with_details(&args, &config)?)
    } else {
        None
    };
    let report = details
        .as_ref()
        .map(|artifact| artifact.summary.report.clone())
        .unwrap_or(run_checkpoint_eval(&args, &config)?);
    let summary = build_summary_artifact(&args, &report, &config);

    log_eval_report(&report);
    if let Some(output_path) = &args.output {
        write_json_report(output_path, &summary)?;
        tracing::info!("Evaluation JSON written to {}", output_path);
    }
    if let (Some(output_path), Some(details)) = (&args.details_output, &details) {
        write_json_report(output_path, details)?;
        tracing::info!("Evaluation details JSON written to {}", output_path);
    }
    Ok(())
}

pub(crate) fn run_checkpoint_eval(args: &EvalArgs, config: &GpcConfig) -> Result<EvaluationReport> {
    let strategy = EvaluationStrategy::parse(&args.strategy)?;
    let device = <EvalBackend as Backend>::Device::default();
    let models = load_models(args, &device)?;
    let mode = if args.synthetic || args.data.is_none() {
        EvaluationMode::SyntheticClosedLoop
    } else {
        EvaluationMode::DatasetOpenLoop
    };

    let num_candidates = args
        .num_candidates
        .unwrap_or(config.gpc_rank.num_candidates)
        .max(1);
    let opt_steps = args
        .opt_steps
        .unwrap_or(config.gpc_opt.num_opt_steps)
        .max(1);
    let opt_learning_rate = args
        .opt_learning_rate
        .unwrap_or(config.gpc_opt.opt_learning_rate as f32)
        .max(f32::EPSILON);
    let settings = CheckpointEvalSettings {
        strategy,
        num_candidates,
        opt_steps,
        opt_learning_rate,
    };

    tracing::info!(
        strategy = settings.strategy.label(),
        mode = mode.label(),
        policy_checkpoint = %resolved_policy_checkpoint_path(args).display(),
        world_model_checkpoint = %resolved_world_model_checkpoint_path(args).display(),
        opt_learning_rate = settings.opt_learning_rate,
        "Starting checkpoint-backed evaluation"
    );

    let report = match mode {
        EvaluationMode::DatasetOpenLoop => {
            let windows = load_evaluation_windows(args, &models)?;
            evaluate_windows(
                &models,
                &windows,
                settings.strategy,
                settings.num_candidates,
                settings.opt_steps,
                settings.opt_learning_rate,
                &device,
            )?
        }
        EvaluationMode::SyntheticClosedLoop => {
            let episodes = load_synthetic_episodes(args, &models, config.training.seed)?;
            evaluate_synthetic_closed_loop(
                &models,
                &episodes,
                settings,
                args.seed.unwrap_or(config.training.seed),
                &device,
            )?
        }
    };

    Ok(report)
}

pub(crate) fn run_checkpoint_eval_with_details(
    args: &EvalArgs,
    config: &GpcConfig,
) -> Result<EvaluationDetailsArtifact> {
    let strategy = EvaluationStrategy::parse(&args.strategy)?;
    let device = <EvalBackend as Backend>::Device::default();
    let models = load_models(args, &device)?;
    let mode = if args.synthetic || args.data.is_none() {
        EvaluationMode::SyntheticClosedLoop
    } else {
        EvaluationMode::DatasetOpenLoop
    };

    let num_candidates = args
        .num_candidates
        .unwrap_or(config.gpc_rank.num_candidates)
        .max(1);
    let opt_steps = args
        .opt_steps
        .unwrap_or(config.gpc_opt.num_opt_steps)
        .max(1);
    let opt_learning_rate = args
        .opt_learning_rate
        .unwrap_or(config.gpc_opt.opt_learning_rate as f32)
        .max(f32::EPSILON);
    let settings = CheckpointEvalSettings {
        strategy,
        num_candidates,
        opt_steps,
        opt_learning_rate,
    };

    let (report, windows, episodes) = match mode {
        EvaluationMode::DatasetOpenLoop => {
            let windows = load_evaluation_windows(args, &models)?;
            let (report, details) = evaluate_windows_with_details(
                &models,
                &windows,
                settings.strategy,
                settings.num_candidates,
                settings.opt_steps,
                settings.opt_learning_rate,
                &device,
            )?;
            (report, details, Vec::new())
        }
        EvaluationMode::SyntheticClosedLoop => {
            let episodes = load_synthetic_episodes(args, &models, config.training.seed)?;
            let (report, details) = evaluate_synthetic_closed_loop_with_details(
                &models,
                &episodes,
                settings,
                args.seed.unwrap_or(config.training.seed),
                &device,
            )?;
            (report, Vec::new(), details)
        }
    };

    Ok(EvaluationDetailsArtifact {
        summary: build_summary_artifact(args, &report, config),
        windows,
        episodes,
    })
}

pub(crate) fn build_summary_artifact(
    args: &EvalArgs,
    report: &EvaluationReport,
    config: &GpcConfig,
) -> EvaluationSummaryArtifact {
    let effective_seed = if report.mode == EvaluationMode::SyntheticClosedLoop {
        Some(args.seed.unwrap_or(config.training.seed))
    } else {
        None
    };

    EvaluationSummaryArtifact {
        strategy: report.strategy,
        mode: report.mode,
        checkpoint_dir: args.checkpoint_dir.clone(),
        policy_checkpoint: resolved_policy_checkpoint_path(args)
            .to_string_lossy()
            .to_string(),
        world_model_checkpoint: resolved_world_model_checkpoint_path(args)
            .to_string_lossy()
            .to_string(),
        data: args.data.clone(),
        synthetic: report.mode == EvaluationMode::SyntheticClosedLoop,
        seed: effective_seed,
        report: report.clone(),
    }
}

pub(crate) fn load_models(
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

pub(crate) fn load_evaluation_windows(
    args: &EvalArgs,
    models: &LoadedModels<EvalBackend>,
) -> Result<Vec<EvaluationWindow>> {
    let episodes = load_episodes_from_path(args.data.as_deref().context("missing data path")?)?;
    validate_episodes_for_evaluation(&episodes, &models.policy_config, &models.world_model_config)?;

    let windows = build_windows(
        &episodes,
        models.policy_config.obs_horizon,
        models.policy_config.pred_horizon,
    )?;

    if windows.is_empty() {
        anyhow::bail!("no evaluation windows available for the selected dataset");
    }

    Ok(windows)
}

pub(crate) fn load_synthetic_episodes(
    args: &EvalArgs,
    models: &LoadedModels<EvalBackend>,
    default_seed: u64,
) -> Result<Vec<Episode>> {
    let episode_length = args
        .episode_length
        .max(models.policy_config.pred_horizon + 1);
    let episodes = generate_synthetic_episodes(
        models.world_model_config.state_dim,
        models.world_model_config.action_dim,
        models.policy_config.obs_dim,
        episode_length,
        args.episodes.max(1),
        args.seed.unwrap_or(default_seed),
    );

    validate_episodes_for_closed_loop(
        &episodes,
        &models.policy_config,
        &models.world_model_config,
    )?;

    Ok(episodes)
}

pub(crate) fn evaluate_windows(
    models: &LoadedModels<EvalBackend>,
    windows: &[EvaluationWindow],
    strategy: EvaluationStrategy,
    num_candidates: usize,
    opt_steps: usize,
    opt_learning_rate: f32,
    device: &<EvalBackend as Backend>::Device,
) -> Result<EvaluationReport> {
    Ok(evaluate_windows_with_details(
        models,
        windows,
        strategy,
        num_candidates,
        opt_steps,
        opt_learning_rate,
        device,
    )?
    .0)
}

fn evaluate_windows_with_details(
    models: &LoadedModels<EvalBackend>,
    windows: &[EvaluationWindow],
    strategy: EvaluationStrategy,
    num_candidates: usize,
    opt_steps: usize,
    opt_learning_rate: f32,
    device: &<EvalBackend as Backend>::Device,
) -> Result<(EvaluationReport, Vec<DatasetWindowDetail>)> {
    let mut windows_evaluated = 0usize;
    let mut rollout_mse_sum = 0.0_f32;
    let mut terminal_distance_sum = 0.0_f32;
    let mut action_mse_sum = 0.0_f32;
    let mut reward_sum = 0.0_f32;
    let mut success_count = 0usize;
    let mut details = Vec::with_capacity(windows.len());

    for window in windows {
        let obs_history = tensor_3d::<EvalBackend>(&window.obs_history, device)?;
        let current_state = tensor_2d::<EvalBackend>(&window.current_state, device)?;
        let expert_actions = tensor_3d::<EvalBackend>(&window.expert_actions, device)?;
        let target_states = tensor_3d::<EvalBackend>(&window.target_states, device)?;
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

        let selected = select_action_sequence_with_trace(
            models,
            &obs_history,
            &current_state,
            reward_fn_eval,
            CheckpointEvalSettings {
                strategy,
                num_candidates,
                opt_steps,
                opt_learning_rate,
            },
            device,
        )?;

        let predicted_states = models
            .world_model
            .rollout(&current_state, &selected.actions)?;
        let reward = reward_fn_metric.compute_reward(&predicted_states)?;

        let predicted_rows = tensor_single_batch_sequence_to_rows(predicted_states.clone())?;
        let predicted_vec = flatten_rows(&predicted_rows);
        let target_vec = tensor_to_vec(target_states.clone())?;
        let selected_rows = tensor_single_batch_sequence_to_rows(selected.actions.clone())?;
        let selected_vec = flatten_rows(&selected_rows);
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
        let success = terminal_distance < 0.1;
        if success {
            success_count += 1;
        }
        details.push(DatasetWindowDetail {
            episode_index: window.episode_index,
            window_index: window.window_index,
            rollout_mse,
            terminal_distance,
            action_mse,
            reward: reward_value,
            success,
            selected_actions: selected_rows,
            predicted_states: predicted_rows,
            target_states: window.target_states.clone(),
            strategy_trace: selected.strategy_trace,
        });
    }

    let denom = windows_evaluated as f32;
    Ok((
        EvaluationReport {
            strategy,
            mode: EvaluationMode::DatasetOpenLoop,
            windows_evaluated,
            mean_rollout_mse: rollout_mse_sum / denom,
            mean_terminal_distance: terminal_distance_sum / denom,
            mean_action_mse: action_mse_sum / denom,
            mean_reward: reward_sum / denom,
            success_rate: success_count as f32 / denom,
            num_candidates,
            opt_steps,
            opt_learning_rate,
        },
        details,
    ))
}

pub(crate) fn evaluate_synthetic_closed_loop(
    models: &LoadedModels<EvalBackend>,
    episodes: &[Episode],
    settings: CheckpointEvalSettings,
    seed: u64,
    device: &<EvalBackend as Backend>::Device,
) -> Result<EvaluationReport> {
    Ok(evaluate_synthetic_closed_loop_with_details(models, episodes, settings, seed, device)?.0)
}

fn evaluate_synthetic_closed_loop_with_details(
    models: &LoadedModels<EvalBackend>,
    episodes: &[Episode],
    settings: CheckpointEvalSettings,
    seed: u64,
    device: &<EvalBackend as Backend>::Device,
) -> Result<(EvaluationReport, Vec<SyntheticEpisodeDetail>)> {
    let mut episodes_evaluated = 0usize;
    let mut rollout_mse_sum = 0.0_f32;
    let mut terminal_distance_sum = 0.0_f32;
    let mut action_mse_sum = 0.0_f32;
    let mut reward_sum = 0.0_f32;
    let mut success_count = 0usize;
    let mut total_replans = 0usize;
    let mut total_executed_steps = 0usize;
    let mut details = Vec::with_capacity(episodes.len());

    for (episode_index, episode) in episodes.iter().enumerate() {
        let (metrics, detail) = evaluate_closed_loop_episode_with_details(
            models,
            episode,
            episode_index,
            settings,
            seed,
            device,
        )?;

        rollout_mse_sum += metrics.rollout_mse;
        terminal_distance_sum += metrics.terminal_distance;
        action_mse_sum += metrics.action_mse;
        reward_sum += metrics.reward;
        total_replans += metrics.replans;
        total_executed_steps += metrics.executed_steps;
        episodes_evaluated += 1;
        if metrics.success {
            success_count += 1;
        }
        details.push(detail);
    }

    let denom = episodes_evaluated as f32;
    tracing::info!(
        episodes = episodes_evaluated,
        replans = total_replans,
        executed_steps = total_executed_steps,
        action_horizon = models.policy_config.action_horizon,
        "Synthetic closed-loop evaluation complete"
    );

    Ok((
        EvaluationReport {
            strategy: settings.strategy,
            mode: EvaluationMode::SyntheticClosedLoop,
            windows_evaluated: episodes_evaluated,
            mean_rollout_mse: rollout_mse_sum / denom,
            mean_terminal_distance: terminal_distance_sum / denom,
            mean_action_mse: action_mse_sum / denom,
            mean_reward: reward_sum / denom,
            success_rate: success_count as f32 / denom,
            num_candidates: settings.num_candidates,
            opt_steps: settings.opt_steps,
            opt_learning_rate: settings.opt_learning_rate,
        },
        details,
    ))
}

#[cfg(test)]
pub(crate) fn evaluate_closed_loop_episode(
    models: &LoadedModels<EvalBackend>,
    episode: &Episode,
    episode_index: usize,
    settings: CheckpointEvalSettings,
    seed: u64,
    device: &<EvalBackend as Backend>::Device,
) -> Result<ClosedLoopEpisodeMetrics> {
    Ok(evaluate_closed_loop_episode_with_details(
        models,
        episode,
        episode_index,
        settings,
        seed,
        device,
    )?
    .0)
}

fn evaluate_closed_loop_episode_with_details(
    models: &LoadedModels<EvalBackend>,
    episode: &Episode,
    episode_index: usize,
    settings: CheckpointEvalSettings,
    seed: u64,
    device: &<EvalBackend as Backend>::Device,
) -> Result<(ClosedLoopEpisodeMetrics, SyntheticEpisodeDetail)> {
    let goal_state = episode
        .states
        .last()
        .cloned()
        .with_context(|| format!("episode {episode_index} is missing a goal state"))?;
    let mut actual_state = episode
        .states
        .first()
        .cloned()
        .with_context(|| format!("episode {episode_index} is missing an initial state"))?;
    let action_horizon = models
        .policy_config
        .action_horizon
        .clamp(1, models.policy_config.pred_horizon);
    let mut obs_history = vec![
        observation_from_state(&actual_state, models.policy_config.obs_dim);
        models.policy_config.obs_horizon
    ];
    let mut executed_actions = Vec::with_capacity(episode.actions.len());
    let mut executed_states = Vec::with_capacity(episode.actions.len());
    let mut predicted_states = Vec::with_capacity(episode.actions.len());
    let mut replans = 0usize;
    let mut action_index = 0usize;
    let mut rng = StdRng::seed_from_u64(closed_loop_seed(seed, episode_index));
    let mut replan_details = Vec::new();

    while action_index < episode.actions.len() {
        let obs_tensor = tensor_3d::<EvalBackend>(&obs_history, device)?;
        let current_state = tensor_2d::<EvalBackend>(&actual_state, device)?;
        let reward_fn_eval = L2RewardFunctionConfig {
            state_dim: models.world_model_config.state_dim,
        }
        .init::<EvalBackend>(device)
        .with_goal(tensor_1d::<EvalBackend>(&goal_state, device));
        let selected = select_action_sequence_with_trace(
            models,
            &obs_tensor,
            &current_state,
            reward_fn_eval,
            settings,
            device,
        )?;
        let predicted_rollout = models
            .world_model
            .rollout(&current_state, &selected.actions)?;
        let planned_actions = tensor_single_batch_sequence_to_rows(selected.actions)?;
        let planned_states = tensor_single_batch_sequence_to_rows(predicted_rollout)?;
        let execute_steps = action_horizon
            .min(episode.actions.len() - action_index)
            .min(planned_actions.len())
            .min(planned_states.len());
        let mut executed_actions_this_replan = Vec::with_capacity(execute_steps);
        let mut executed_states_this_replan = Vec::with_capacity(execute_steps);
        let replan_start = action_index;

        for step_offset in 0..execute_steps {
            let action = planned_actions[step_offset].clone();
            actual_state = synthetic_transition(&mut rng, &actual_state, &action);
            predicted_states.push(planned_states[step_offset].clone());
            executed_actions.push(action.clone());
            executed_states.push(actual_state.clone());
            executed_actions_this_replan.push(action);
            executed_states_this_replan.push(actual_state.clone());
            obs_history.remove(0);
            obs_history.push(observation_from_state(
                &actual_state,
                models.policy_config.obs_dim,
            ));
        }

        replan_details.push(SyntheticReplanDetail {
            replan_index: replans,
            start_action_index: replan_start,
            executed_steps: execute_steps,
            selected_actions: planned_actions.clone(),
            predicted_states: planned_states.clone(),
            executed_actions: executed_actions_this_replan,
            executed_states: executed_states_this_replan,
            strategy_trace: selected.strategy_trace,
        });
        action_index += execute_steps;
        replans += 1;
    }

    let reward_fn_metric = L2RewardFunctionConfig {
        state_dim: models.world_model_config.state_dim,
    }
    .init::<EvalBackend>(device)
    .with_goal(tensor_1d::<EvalBackend>(&goal_state, device));
    let executed_rollout = tensor_3d::<EvalBackend>(&executed_states, device)?;
    let reward_value: f32 = reward_fn_metric
        .compute_reward(&executed_rollout)?
        .into_scalar()
        .elem();
    let rollout_mse = mean_squared_error(
        &flatten_rows(&predicted_states),
        &flatten_rows(&executed_states),
    );
    let terminal_distance = l2_distance(&actual_state, &goal_state);
    let action_mse = mean_squared_error(
        &flatten_rows(&executed_actions),
        &flatten_rows(&episode.actions),
    );

    let metrics = ClosedLoopEpisodeMetrics {
        rollout_mse,
        terminal_distance,
        action_mse,
        reward: reward_value,
        success: terminal_distance < 0.1,
        replans,
        executed_steps: executed_actions.len(),
    };
    let detail = SyntheticEpisodeDetail {
        episode_index,
        goal_state,
        rollout_mse: metrics.rollout_mse,
        terminal_distance: metrics.terminal_distance,
        action_mse: metrics.action_mse,
        reward: metrics.reward,
        success: metrics.success,
        replans: metrics.replans,
        executed_steps: metrics.executed_steps,
        replans_trace: replan_details,
    };

    Ok((metrics, detail))
}

fn select_action_sequence_with_trace(
    models: &LoadedModels<EvalBackend>,
    obs_history: &Tensor<EvalBackend, 3>,
    current_state: &Tensor<EvalBackend, 2>,
    reward_fn: gpc_world::reward::L2RewardFunction<EvalBackend>,
    settings: CheckpointEvalSettings,
    device: &<EvalBackend as Backend>::Device,
) -> Result<SelectedTrajectory<EvalBackend>> {
    match settings.strategy {
        EvaluationStrategy::Policy => Ok(SelectedTrajectory {
            actions: models.policy.sample(obs_history, device)?,
            strategy_trace: StrategyTraceDetail::Policy {},
        }),
        EvaluationStrategy::Rank => {
            let trace =
                GpcRankBuilder::new(models.policy.clone(), models.world_model.clone(), reward_fn)
                    .num_candidates(settings.num_candidates)
                    .build()
                    .select_action_with_trace(obs_history, current_state, device)?;
            Ok(SelectedTrajectory {
                actions: trace.best_action,
                strategy_trace: StrategyTraceDetail::Rank {
                    best_index: trace.best_index,
                    candidate_rewards: tensor_to_vec(trace.rewards)?,
                    num_candidates: trace.num_candidates,
                },
            })
        }
        EvaluationStrategy::Opt => {
            let trace = AutodiffGpcOptBuilder::new(
                models.policy.clone(),
                models.world_model.clone(),
                reward_fn,
            )
            .num_opt_steps(settings.opt_steps)
            .learning_rate(settings.opt_learning_rate)
            .build()
            .select_action_with_trace(obs_history, current_state, device)?;
            Ok(SelectedTrajectory {
                actions: trace.optimized_actions.clone(),
                strategy_trace: StrategyTraceDetail::Opt {
                    initial_actions: tensor_single_batch_sequence_to_rows(trace.initial_actions)?,
                    step_rewards: trace
                        .step_traces
                        .into_iter()
                        .map(|step| step.reward.into_scalar().elem())
                        .collect(),
                    epsilon: trace.epsilon,
                    learning_rate: trace.learning_rate,
                    num_opt_steps: trace.num_opt_steps,
                },
            })
        }
    }
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
        "policy" => {
            tracing::info!("Policy-only baseline rollout");
            let sampled = policy.sample(&obs, &device)?;
            tracing::info!("Sampled action shape: {:?}", sampled.dims());
        }
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

            let evaluator = AutodiffGpcOptBuilder::new(policy, world_model, reward_fn)
                .num_opt_steps(steps)
                .build();

            let best = evaluator.select_action(&obs, &state, &device)?;
            tracing::info!("Optimized action shape: {:?}", best.dims());
        }
        other => {
            anyhow::bail!("Unknown strategy: {other}. Use 'policy', 'rank', or 'opt'.");
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

pub(crate) fn generate_synthetic_episodes(
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

fn synthetic_transition(rng: &mut StdRng, state: &[f32], action: &[f32]) -> Vec<f32> {
    state
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let action_term = action.get(index).copied().unwrap_or(0.0);
            value + 0.1 * action_term + rng.gen_range(-0.01..0.01)
        })
        .collect()
}

fn observation_from_state(state: &[f32], obs_dim: usize) -> Vec<f32> {
    state[..obs_dim.min(state.len())].to_vec()
}

fn closed_loop_seed(seed: u64, episode_index: usize) -> u64 {
    seed ^ ((episode_index as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn build_windows(
    episodes: &[Episode],
    obs_horizon: usize,
    pred_horizon: usize,
) -> Result<Vec<EvaluationWindow>> {
    let mut windows = Vec::new();

    for (episode_index, episode) in episodes.iter().enumerate() {
        if episode.states.len() < pred_horizon + 1 {
            continue;
        }

        for start in 0..=(episode.states.len() - pred_horizon - 1) {
            let obs_history =
                build_obs_history(&episode.observations, start, obs_horizon, episode_index)?;
            let current_state = episode.states.get(start).cloned().with_context(|| {
                format!("episode {episode_index} current state missing for sample {start}")
            })?;
            let expert_actions =
                take_window(&episode.actions, start, pred_horizon).with_context(|| {
                    format!("episode {episode_index} action window is missing for sample {start}")
                })?;
            let target_states = take_window(&episode.states, start + 1, pred_horizon)
                .with_context(|| {
                    format!(
                        "episode {episode_index} target state window is missing for sample {start}"
                    )
                })?;

            windows.push(EvaluationWindow {
                episode_index,
                window_index: start,
                obs_history,
                current_state,
                expert_actions,
                target_states,
            });
        }
    }

    Ok(windows)
}

fn validate_episodes_for_evaluation(
    episodes: &[Episode],
    policy_config: &PolicyConfig,
    world_model_config: &WorldModelConfig,
) -> Result<()> {
    if episodes.is_empty() {
        anyhow::bail!("dataset does not contain any episodes");
    }

    let mut usable_windows = 0usize;
    for (episode_index, episode) in episodes.iter().enumerate() {
        validate_episode(episode, episode_index, policy_config, world_model_config)?;
        if episode.states.len() > policy_config.pred_horizon {
            usable_windows += episode.states.len() - policy_config.pred_horizon;
        }
    }

    if usable_windows == 0 {
        anyhow::bail!("dataset does not contain any usable evaluation windows");
    }

    Ok(())
}

pub(crate) fn validate_episodes_for_closed_loop(
    episodes: &[Episode],
    policy_config: &PolicyConfig,
    world_model_config: &WorldModelConfig,
) -> Result<()> {
    if episodes.is_empty() {
        anyhow::bail!("synthetic evaluation did not generate any episodes");
    }

    for (episode_index, episode) in episodes.iter().enumerate() {
        validate_episode(episode, episode_index, policy_config, world_model_config)?;
    }

    Ok(())
}

fn validate_episode(
    episode: &Episode,
    episode_index: usize,
    policy_config: &PolicyConfig,
    world_model_config: &WorldModelConfig,
) -> Result<()> {
    let state_dim = episode
        .states
        .first()
        .map(|state| state.len())
        .with_context(|| format!("episode {episode_index} is missing state samples"))?;
    let action_dim = episode
        .actions
        .first()
        .map(|action| action.len())
        .with_context(|| format!("episode {episode_index} is missing action samples"))?;
    let obs_dim = episode
        .observations
        .first()
        .map(|obs| obs.len())
        .with_context(|| format!("episode {episode_index} is missing observation samples"))?;

    if state_dim != world_model_config.state_dim {
        anyhow::bail!(
            "episode {episode_index} state dimension mismatch: dataset={} checkpoint={}",
            state_dim,
            world_model_config.state_dim
        );
    }
    if action_dim != policy_config.action_dim || action_dim != world_model_config.action_dim {
        anyhow::bail!(
            "episode {episode_index} action dimension mismatch: dataset={} policy={} world_model={}",
            action_dim,
            policy_config.action_dim,
            world_model_config.action_dim
        );
    }
    if obs_dim != policy_config.obs_dim {
        anyhow::bail!(
            "episode {episode_index} observation dimension mismatch: dataset={} checkpoint={}",
            obs_dim,
            policy_config.obs_dim
        );
    }
    if episode.states.len() != episode.observations.len() {
        anyhow::bail!(
            "episode {episode_index} must have the same number of states and observations"
        );
    }
    if episode.states.len() != episode.actions.len() + 1 {
        anyhow::bail!("episode {episode_index} must contain exactly one more state than action");
    }

    validate_sequence_shapes(&episode.states, state_dim, "state", episode_index)?;
    validate_sequence_shapes(&episode.observations, obs_dim, "observation", episode_index)?;
    validate_sequence_shapes(&episode.actions, action_dim, "action", episode_index)?;

    Ok(())
}

fn validate_sequence_shapes(
    rows: &[Vec<f32>],
    expected_dim: usize,
    label: &str,
    episode_index: usize,
) -> Result<()> {
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != expected_dim {
            anyhow::bail!(
                "episode {episode_index} {label} {row_index} has dimension {} but expected {}",
                row.len(),
                expected_dim
            );
        }
    }

    Ok(())
}

fn build_obs_history(
    observations: &[Vec<f32>],
    start: usize,
    obs_horizon: usize,
    episode_index: usize,
) -> Result<Vec<Vec<f32>>> {
    if observations.is_empty() {
        anyhow::bail!("episode {episode_index} is missing observations");
    }

    let obs_start = if start >= obs_horizon {
        start + 1 - obs_horizon
    } else {
        0
    };
    let obs_end = obs_start
        .checked_add(obs_horizon)
        .context("observation window overflowed")?;

    let mut obs_history = observations
        .get(obs_start..obs_end.min(observations.len()))
        .map(|window| window.to_vec())
        .with_context(|| {
            format!("episode {episode_index} observation window is missing for sample {start}")
        })?;

    while obs_history.len() < obs_horizon {
        obs_history.insert(0, obs_history[0].clone());
    }

    Ok(obs_history)
}

fn take_window(rows: &[Vec<f32>], start: usize, len: usize) -> Result<Vec<Vec<f32>>> {
    let end = start
        .checked_add(len)
        .context("evaluation window overflowed")?;
    rows.get(start..end)
        .map(|window| window.to_vec())
        .context("evaluation window is out of bounds")
}

pub(crate) fn resolved_policy_checkpoint_path(args: &EvalArgs) -> PathBuf {
    args.policy_checkpoint
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| preferred_policy_checkpoint_path(args))
}

pub(crate) fn resolved_world_model_checkpoint_path(args: &EvalArgs) -> PathBuf {
    args.world_model_checkpoint
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| preferred_world_model_checkpoint_path(args))
}

fn preferred_policy_checkpoint_path(args: &EvalArgs) -> PathBuf {
    preferred_checkpoint_path(&args.checkpoint_dir, "policy_best.bin", "policy_final.bin")
}

fn preferred_world_model_checkpoint_path(args: &EvalArgs) -> PathBuf {
    preferred_checkpoint_path(
        &args.checkpoint_dir,
        "world_model_best.bin",
        "world_model_final.bin",
    )
}

fn preferred_checkpoint_path(checkpoint_dir: &str, preferred: &str, fallback: &str) -> PathBuf {
    let preferred_path = PathBuf::from(checkpoint_dir).join(preferred);
    if preferred_path.exists() {
        return preferred_path;
    }

    PathBuf::from(checkpoint_dir).join(fallback)
}

fn log_eval_report(report: &EvaluationReport) {
    tracing::info!("Checkpoint evaluation complete");
    tracing::info!("  strategy: {}", report.strategy.label());
    tracing::info!("  mode: {}", report.mode.label());
    tracing::info!(
        "  {}: {}",
        report.mode.unit_label(),
        report.windows_evaluated
    );
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
    tracing::info!("  opt learning rate: {:.6}", report.opt_learning_rate);
}

fn tensor_1d<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats(values, device)
}

fn tensor_2d<B: Backend>(values: &[f32], device: &B::Device) -> Result<Tensor<B, 2>> {
    if values.is_empty() {
        anyhow::bail!("cannot build tensor from an empty row");
    }

    Ok(Tensor::<B, 1>::from_floats(values, device).reshape([1, values.len()]))
}

fn tensor_3d<B: Backend>(values: &[Vec<f32>], device: &B::Device) -> Result<Tensor<B, 3>> {
    if values.is_empty() {
        anyhow::bail!("cannot build tensor from an empty sequence");
    }

    let outer = values.len();
    let inner = values
        .first()
        .map(|row| row.len())
        .context("cannot build tensor from an empty row")?;
    if inner == 0 {
        anyhow::bail!("cannot build tensor from zero-width rows");
    }
    for (row_index, row) in values.iter().enumerate() {
        if row.len() != inner {
            anyhow::bail!(
                "tensor row {row_index} has dimension {} but expected {}",
                row.len(),
                inner
            );
        }
    }
    let flat = values
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    Ok(Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([1, outer, inner]))
}

fn tensor_to_vec<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>> {
    Ok(tensor
        .into_data()
        .to_vec()
        .expect("tensor data should always be contiguous"))
}

fn tensor_single_batch_sequence_to_rows<B: Backend>(tensor: Tensor<B, 3>) -> Result<Vec<Vec<f32>>> {
    let [batch_size, horizon, row_dim] = tensor.dims();
    if batch_size != 1 {
        anyhow::bail!("expected a batch size of 1, got {batch_size}");
    }

    let flat = tensor_to_vec(tensor)?;
    Ok(flat
        .chunks(row_dim)
        .take(horizon)
        .map(|row| row.to_vec())
        .collect())
}

fn flatten_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
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

fn l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| {
            let diff = lhs - rhs;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpc_compat::{
        CheckpointFormat, CheckpointMetadata, save_policy_checkpoint, save_world_model_checkpoint,
    };
    use gpc_core::config::TrainingConfig;
    use gpc_train::{PolicyTrainer, WorldModelTrainer};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_eval_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gpc_eval_{name}_{}_{}",
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
        save_tiny_checkpoints_with_action_horizon(dir, 1)
    }

    fn save_tiny_checkpoints_with_action_horizon(
        dir: &Path,
        action_horizon: usize,
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
            action_horizon,
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

    fn save_tiny_checkpoints_with_best(
        dir: &Path,
    ) -> Result<(
        PathBuf,
        PathBuf,
        PathBuf,
        PathBuf,
        PolicyConfig,
        WorldModelConfig,
    )> {
        let device = <EvalBackend as Backend>::Device::default();
        let dataset = generate_synthetic_episodes(4, 2, 4, 10, 4, 52);
        let dataset_config = gpc_train::data::GpcDatasetConfig {
            data_dir: "synthetic".to_string(),
            state_dim: 4,
            action_dim: 2,
            obs_dim: 4,
            obs_horizon: 2,
            pred_horizon: 4,
        };
        let gpc_dataset = gpc_train::data::GpcDataset::new(dataset, dataset_config);
        let split = gpc_dataset.split(0.25, 99)?;

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

        let policy_summary = PolicyTrainer::new(training_config.clone(), policy_config.clone())
            .train_with_validation_summary::<EvalBackend>(
            &split.train,
            Some(&split.validation),
            &device,
        );
        let phase1_summary =
            WorldModelTrainer::new(training_config.clone(), world_model_config.clone())
                .train_phase1_with_validation_summary::<EvalBackend>(
                    &split.train,
                    Some(&split.validation),
                    &device,
                );
        let phase2_summary = WorldModelTrainer::new(training_config, world_model_config.clone())
            .train_phase2_with_validation_summary::<EvalBackend>(
            &split.train,
            phase1_summary.best_model.clone(),
            2,
            Some(&split.validation),
            &device,
        );

        let policy_final_path = save_policy_checkpoint(
            policy_summary.training.model,
            &checkpoint_metadata(
                CheckpointKind::Policy,
                policy_summary.training.final_epoch,
                policy_summary.training.final_loss,
                serde_json::to_string(&policy_config)?,
            ),
            dir.join("policy_final"),
            CheckpointFormat::Bin,
        )?
        .checkpoint_path;

        let policy_best_path = save_policy_checkpoint(
            policy_summary.best_model,
            &checkpoint_metadata(
                CheckpointKind::Policy,
                policy_summary.best_epoch,
                policy_summary.best_validation_loss,
                serde_json::to_string(&policy_config)?,
            ),
            dir.join("policy_best"),
            CheckpointFormat::Bin,
        )?
        .checkpoint_path;

        let world_model_final_path = save_world_model_checkpoint(
            phase2_summary.training.model,
            &checkpoint_metadata(
                CheckpointKind::WorldModel,
                phase2_summary.training.final_epoch,
                phase2_summary.training.final_loss,
                serde_json::to_string(&world_model_config)?,
            ),
            dir.join("world_model_final"),
            CheckpointFormat::Bin,
        )?
        .checkpoint_path;

        let world_model_best_path = save_world_model_checkpoint(
            phase2_summary.best_model,
            &checkpoint_metadata(
                CheckpointKind::WorldModel,
                phase2_summary.best_epoch,
                phase2_summary.best_validation_loss,
                serde_json::to_string(&world_model_config)?,
            ),
            dir.join("world_model_best"),
            CheckpointFormat::Bin,
        )?
        .checkpoint_path;

        Ok((
            policy_final_path,
            policy_best_path,
            world_model_final_path,
            world_model_best_path,
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
            opt_learning_rate: None,
            seed: Some(42),
            output: None,
            details_output: None,
            demo: false,
        };
        let config = GpcConfig::default();

        let report = run_checkpoint_eval(&args, &config).unwrap();

        assert_eq!(report.strategy, EvaluationStrategy::Rank);
        assert_eq!(report.mode, EvaluationMode::SyntheticClosedLoop);
        assert_eq!(report.windows_evaluated, 4);
        assert!(report.mean_rollout_mse.is_finite());
        assert!(report.mean_terminal_distance.is_finite());
        assert!(report.mean_action_mse.is_finite());
        assert!(report.mean_reward.is_finite());
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        assert_eq!(report.num_candidates, 4);
        assert_eq!(report.opt_steps, 2);
        assert!((report.opt_learning_rate - config.gpc_opt.opt_learning_rate as f32).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_windows_match_checkpoint_dims() {
        let episodes = generate_synthetic_episodes(4, 2, 4, 8, 2, 42);
        let windows = build_windows(&episodes, 2, 4).unwrap();

        assert!(!windows.is_empty());
        assert_eq!(windows[0].obs_history.len(), 2);
        assert_eq!(windows[0].current_state.len(), 4);
        assert_eq!(windows[0].expert_actions.len(), 4);
        assert_eq!(windows[0].target_states.len(), 4);
    }

    #[test]
    fn dataset_eval_remains_open_loop_windowed() {
        let dir = temp_eval_dir("dataset_open_loop");
        let data_path = dir.join("episodes.json");
        let (policy_path, world_model_path, _, _) = save_tiny_checkpoints(&dir).unwrap();
        let episodes = generate_synthetic_episodes(4, 2, 4, 8, 2, 7);
        std::fs::write(&data_path, serde_json::to_vec(&episodes).unwrap()).unwrap();

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: Some(data_path.to_string_lossy().to_string()),
            synthetic: false,
            episodes: 1,
            episode_length: 6,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: None,
            seed: Some(42),
            output: None,
            details_output: None,
            demo: false,
        };

        let report = run_checkpoint_eval(&args, &GpcConfig::default()).unwrap();

        assert_eq!(report.mode, EvaluationMode::DatasetOpenLoop);
        assert_eq!(report.windows_evaluated, 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eval_writes_summary_json() {
        let dir = temp_eval_dir("summary_json");
        let (policy_path, world_model_path, _, _) = save_tiny_checkpoints(&dir).unwrap();
        let output_path = dir.join("reports/eval_summary.json");

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: true,
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: None,
            seed: Some(9),
            output: Some(output_path.to_string_lossy().to_string()),
            details_output: None,
            demo: false,
        };

        run_eval(args).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();
        assert_eq!(json["mode"].as_str().unwrap(), "synthetic-closed-loop");
        assert_eq!(json["strategy"].as_str().unwrap(), "rank");
        assert_eq!(json["report"]["windows_evaluated"].as_u64().unwrap(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eval_summary_uses_effective_run_settings() {
        let dir = temp_eval_dir("summary_effective_settings");
        let (policy_path, world_model_path, _, _) = save_tiny_checkpoints(&dir).unwrap();
        let output_path = dir.join("reports/eval_summary_effective.json");

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: false,
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: None,
            seed: None,
            output: Some(output_path.to_string_lossy().to_string()),
            details_output: None,
            demo: false,
        };

        run_eval(args).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();
        assert_eq!(json["mode"].as_str().unwrap(), "synthetic-closed-loop");
        assert!(json["synthetic"].as_bool().unwrap());
        assert_eq!(
            json["seed"].as_u64().unwrap(),
            GpcConfig::default().training.seed
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eval_writes_window_details_json() {
        let dir = temp_eval_dir("window_details");
        let data_path = dir.join("episodes.json");
        let output_path = dir.join("reports/eval_details.json");
        let (policy_path, world_model_path, _, _) = save_tiny_checkpoints(&dir).unwrap();
        let episodes = generate_synthetic_episodes(4, 2, 4, 8, 2, 17);
        std::fs::write(&data_path, serde_json::to_vec(&episodes).unwrap()).unwrap();

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: Some(data_path.to_string_lossy().to_string()),
            synthetic: false,
            episodes: 1,
            episode_length: 6,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: None,
            seed: Some(11),
            output: None,
            details_output: Some(output_path.to_string_lossy().to_string()),
            demo: false,
        };

        run_eval(args).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();
        assert_eq!(
            json["summary"]["mode"].as_str().unwrap(),
            "dataset-open-loop"
        );
        assert_eq!(json["windows"].as_array().unwrap().len(), 8);
        assert_eq!(json["episodes"].as_array().unwrap().len(), 0);
        assert_eq!(json["windows"][0]["episode_index"].as_u64().unwrap(), 0);
        assert!(json["windows"][0]["strategy_trace"]["candidate_rewards"].is_array());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eval_writes_episode_details_json() {
        let dir = temp_eval_dir("episode_details");
        let output_path = dir.join("reports/eval_episode_details.json");
        let (policy_path, world_model_path, _, _) = save_tiny_checkpoints(&dir).unwrap();

        let args = EvalArgs {
            strategy: "opt".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: true,
            episodes: 1,
            episode_length: 8,
            num_candidates: Some(1),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            seed: Some(5),
            output: None,
            details_output: Some(output_path.to_string_lossy().to_string()),
            demo: false,
        };

        run_eval(args).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();
        assert_eq!(
            json["summary"]["mode"].as_str().unwrap(),
            "synthetic-closed-loop"
        );
        assert_eq!(json["windows"].as_array().unwrap().len(), 0);
        assert_eq!(json["episodes"].as_array().unwrap().len(), 1);
        assert!(
            !json["episodes"][0]["replans_trace"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            json["episodes"][0]["replans_trace"][0]["strategy_trace"]["strategy"]
                .as_str()
                .unwrap(),
            "opt"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn preferred_checkpoint_paths_use_best_artifacts_when_available() {
        let dir = temp_eval_dir("preferred_best");
        let (
            policy_final_path,
            policy_best_path,
            world_model_final_path,
            world_model_best_path,
            _,
            _,
        ) = save_tiny_checkpoints_with_best(&dir).unwrap();

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: None,
            world_model_checkpoint: None,
            data: None,
            synthetic: true,
            episodes: 1,
            episode_length: 6,
            num_candidates: Some(2),
            opt_steps: Some(1),
            opt_learning_rate: None,
            seed: Some(42),
            output: None,
            details_output: None,
            demo: false,
        };

        assert_eq!(resolved_policy_checkpoint_path(&args), policy_best_path);
        assert_eq!(
            resolved_world_model_checkpoint_path(&args),
            world_model_best_path
        );
        assert_ne!(resolved_policy_checkpoint_path(&args), policy_final_path);
        assert_ne!(
            resolved_world_model_checkpoint_path(&args),
            world_model_final_path
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn preferred_checkpoint_paths_fall_back_to_final_artifacts() {
        let dir = temp_eval_dir("preferred_final");
        let (policy_path, world_model_path, _, _) = save_tiny_checkpoints(&dir).unwrap();

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: None,
            world_model_checkpoint: None,
            data: None,
            synthetic: true,
            episodes: 1,
            episode_length: 6,
            num_candidates: Some(2),
            opt_steps: Some(1),
            opt_learning_rate: None,
            seed: Some(42),
            output: None,
            details_output: None,
            demo: false,
        };

        assert_eq!(resolved_policy_checkpoint_path(&args), policy_path);
        assert_eq!(
            resolved_world_model_checkpoint_path(&args),
            world_model_path
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explicit_checkpoint_paths_override_directory_preference() {
        let dir = temp_eval_dir("explicit_override");
        let (
            policy_final_path,
            policy_best_path,
            world_model_final_path,
            world_model_best_path,
            _,
            _,
        ) = save_tiny_checkpoints_with_best(&dir).unwrap();

        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_final_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_final_path.to_string_lossy().to_string()),
            data: None,
            synthetic: true,
            episodes: 1,
            episode_length: 6,
            num_candidates: Some(2),
            opt_steps: Some(1),
            opt_learning_rate: None,
            seed: Some(42),
            output: None,
            details_output: None,
            demo: false,
        };

        assert_eq!(resolved_policy_checkpoint_path(&args), policy_final_path);
        assert_eq!(
            resolved_world_model_checkpoint_path(&args),
            world_model_final_path
        );
        assert_ne!(resolved_policy_checkpoint_path(&args), policy_best_path);
        assert_ne!(
            resolved_world_model_checkpoint_path(&args),
            world_model_best_path
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_closed_loop_honors_action_horizon() {
        let dir = temp_eval_dir("closed_loop_stride");
        let (policy_path, world_model_path, _, _) =
            save_tiny_checkpoints_with_action_horizon(&dir, 2).unwrap();
        let args = EvalArgs {
            strategy: "rank".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: true,
            episodes: 1,
            episode_length: 9,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: None,
            seed: Some(13),
            output: None,
            details_output: None,
            demo: false,
        };
        let config = GpcConfig::default();
        let device = <EvalBackend as Backend>::Device::default();
        let models = load_models(&args, &device).unwrap();
        let episodes = load_synthetic_episodes(&args, &models, config.training.seed).unwrap();

        let metrics = evaluate_closed_loop_episode(
            &models,
            &episodes[0],
            0,
            CheckpointEvalSettings {
                strategy: EvaluationStrategy::Rank,
                num_candidates: 4,
                opt_steps: 2,
                opt_learning_rate: config.gpc_opt.opt_learning_rate as f32,
            },
            args.seed.unwrap(),
            &device,
        )
        .unwrap();

        assert_eq!(metrics.executed_steps, episodes[0].actions.len());
        assert_eq!(metrics.replans, episodes[0].actions.len().div_ceil(2));
        assert!(metrics.rollout_mse.is_finite());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opt_path_honors_learning_rate_override() {
        let dir = temp_eval_dir("opt_lr");
        let (policy_path, world_model_path, _, _) = save_tiny_checkpoints(&dir).unwrap();

        let args = EvalArgs {
            strategy: "opt".to_string(),
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: true,
            episodes: 1,
            episode_length: 6,
            num_candidates: Some(1),
            opt_steps: Some(1),
            opt_learning_rate: Some(0.123),
            seed: Some(42),
            output: None,
            details_output: None,
            demo: false,
        };

        let config = GpcConfig::default();
        let report = run_checkpoint_eval(&args, &config).unwrap();

        assert_eq!(report.strategy, EvaluationStrategy::Opt);
        assert_eq!(report.mode, EvaluationMode::SyntheticClosedLoop);
        assert_eq!(report.opt_steps, 1);
        assert!((report.opt_learning_rate - 0.123).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_dataset_is_rejected() {
        let policy_config = PolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 4,
            action_horizon: 1,
            hidden_dim: 16,
            num_res_blocks: 1,
            noise_schedule: gpc_core::config::NoiseScheduleConfig::default(),
        };
        let world_model_config = WorldModelConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_dim: 16,
            num_layers: 1,
            dropout: 0.0,
        };
        let malformed = vec![Episode {
            states: vec![vec![0.0; 4], vec![0.0; 4]],
            actions: vec![vec![0.0; 2], vec![0.0; 2]],
            observations: vec![vec![0.0; 4]],
        }];

        let err = validate_episodes_for_evaluation(&malformed, &policy_config, &world_model_config)
            .expect_err("malformed dataset should be rejected");
        assert!(
            err.to_string()
                .contains("same number of states and observations")
        );
    }
}
