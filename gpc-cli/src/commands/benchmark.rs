//! Benchmark command implementation.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use clap::Args;
use serde::Serialize;

use gpc_core::config::GpcConfig;

use super::eval::{
    CheckpointEvalSettings, EvalArgs, EvaluationReport, EvaluationStrategy,
    evaluate_synthetic_closed_loop, generate_synthetic_episodes, load_models,
    validate_episodes_for_closed_loop,
};

type BenchmarkBackend = Autodiff<NdArray>;

/// Arguments for the benchmark command.
#[derive(Args, Debug, Clone)]
pub struct BenchmarkArgs {
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

    /// Strategy to include in the suite. Repeat to benchmark a subset.
    #[arg(long = "strategy")]
    strategies: Vec<String>,

    /// Seed for a synthetic benchmark run. Repeat to expand the suite.
    #[arg(long = "seed")]
    seeds: Vec<u64>,

    /// Number of synthetic episodes per benchmark run.
    #[arg(long, default_value = "12")]
    episodes: usize,

    /// Number of timesteps per synthetic episode.
    #[arg(long, default_value = "36")]
    episode_length: usize,

    /// Number of candidate trajectories for ranking.
    #[arg(long)]
    num_candidates: Option<usize>,

    /// Number of optimization steps for GPC-OPT.
    #[arg(long)]
    opt_steps: Option<usize>,

    /// Override the GPC-OPT learning rate.
    #[arg(long)]
    opt_learning_rate: Option<f32>,

    /// Optional output path for a machine-readable JSON report.
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkRun {
    seed: u64,
    report: EvaluationReport,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkDelta {
    mean_rollout_mse: f32,
    mean_terminal_distance: f32,
    mean_action_mse: f32,
    mean_reward: f32,
    success_rate: f32,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkSummary {
    strategy: EvaluationStrategy,
    runs: usize,
    episodes_evaluated: usize,
    mean_rollout_mse: f32,
    mean_terminal_distance: f32,
    mean_action_mse: f32,
    mean_reward: f32,
    success_rate: f32,
    num_candidates: usize,
    opt_steps: usize,
    opt_learning_rate: f32,
    delta_vs_policy: Option<BenchmarkDelta>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    checkpoint_dir: String,
    policy_checkpoint: Option<String>,
    world_model_checkpoint: Option<String>,
    strategies: Vec<EvaluationStrategy>,
    seeds: Vec<u64>,
    episodes_per_seed: usize,
    episode_length: usize,
    num_candidates: usize,
    opt_steps: usize,
    opt_learning_rate: f32,
    per_run: Vec<BenchmarkRun>,
    summary: Vec<BenchmarkSummary>,
}

#[derive(Debug, Clone, Default)]
struct MetricAccumulator {
    runs: usize,
    episodes_evaluated: usize,
    rollout_mse_total: f64,
    terminal_distance_total: f64,
    action_mse_total: f64,
    reward_total: f64,
    success_total: f64,
    num_candidates: usize,
    opt_steps: usize,
    opt_learning_rate: f32,
}

impl MetricAccumulator {
    fn push(&mut self, report: &EvaluationReport) {
        let weight = report.windows_evaluated.max(1);
        let weight_f64 = weight as f64;

        self.runs += 1;
        self.episodes_evaluated += report.windows_evaluated;
        self.rollout_mse_total += f64::from(report.mean_rollout_mse) * weight_f64;
        self.terminal_distance_total += f64::from(report.mean_terminal_distance) * weight_f64;
        self.action_mse_total += f64::from(report.mean_action_mse) * weight_f64;
        self.reward_total += f64::from(report.mean_reward) * weight_f64;
        self.success_total += f64::from(report.success_rate) * weight_f64;
        self.num_candidates = report.num_candidates;
        self.opt_steps = report.opt_steps;
        self.opt_learning_rate = report.opt_learning_rate;
    }

    fn finish(self, strategy: EvaluationStrategy) -> BenchmarkSummary {
        let denom = self.episodes_evaluated.max(1) as f64;

        BenchmarkSummary {
            strategy,
            runs: self.runs,
            episodes_evaluated: self.episodes_evaluated,
            mean_rollout_mse: (self.rollout_mse_total / denom) as f32,
            mean_terminal_distance: (self.terminal_distance_total / denom) as f32,
            mean_action_mse: (self.action_mse_total / denom) as f32,
            mean_reward: (self.reward_total / denom) as f32,
            success_rate: (self.success_total / denom) as f32,
            num_candidates: self.num_candidates,
            opt_steps: self.opt_steps,
            opt_learning_rate: self.opt_learning_rate,
            delta_vs_policy: None,
        }
    }
}

/// Run the benchmark command.
pub fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    let config = if let Some(config_path) = &args.config {
        let data = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&data)?
    } else {
        GpcConfig::default()
    };

    let strategies = parse_strategies(&args.strategies)?;
    let seeds = resolve_seeds(&args, &config);
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

    let eval_args = EvalArgs {
        strategy: EvaluationStrategy::Rank.label().to_string(),
        config: args.config.clone(),
        checkpoint_dir: args.checkpoint_dir.clone(),
        policy_checkpoint: args.policy_checkpoint.clone(),
        world_model_checkpoint: args.world_model_checkpoint.clone(),
        data: None,
        synthetic: true,
        episodes: args.episodes,
        episode_length: args.episode_length,
        num_candidates: Some(num_candidates),
        opt_steps: Some(opt_steps),
        opt_learning_rate: Some(opt_learning_rate),
        seed: None,
        demo: false,
    };

    let device = <BenchmarkBackend as Backend>::Device::default();
    let models = load_models(&eval_args, &device)?;
    let mut per_run = Vec::with_capacity(strategies.len() * seeds.len());
    let episode_length = args
        .episode_length
        .max(models.policy_config.pred_horizon + 1);
    let episodes_per_seed = args.episodes.max(1);

    for &seed in &seeds {
        let episodes = generate_synthetic_episodes(
            models.world_model_config.state_dim,
            models.world_model_config.action_dim,
            models.policy_config.obs_dim,
            episode_length,
            episodes_per_seed,
            seed,
        );
        validate_episodes_for_closed_loop(
            &episodes,
            &models.policy_config,
            &models.world_model_config,
        )?;

        for strategy in &strategies {
            let report = evaluate_synthetic_closed_loop(
                &models,
                &episodes,
                CheckpointEvalSettings {
                    strategy: *strategy,
                    num_candidates,
                    opt_steps,
                    opt_learning_rate,
                },
                seed,
                &device,
            )?;
            per_run.push(BenchmarkRun { seed, report });
        }
    }

    let summary = summarize_runs(&per_run);
    let report = BenchmarkReport {
        checkpoint_dir: args.checkpoint_dir.clone(),
        policy_checkpoint: args.policy_checkpoint.clone(),
        world_model_checkpoint: args.world_model_checkpoint.clone(),
        strategies,
        seeds,
        episodes_per_seed,
        episode_length,
        num_candidates,
        opt_steps,
        opt_learning_rate,
        per_run,
        summary,
    };

    log_benchmark_report(&report);

    if let Some(output_path) = &args.output {
        write_report(output_path, &report)?;
        tracing::info!("Benchmark JSON written to {}", output_path);
    }

    Ok(())
}

fn parse_strategies(values: &[String]) -> Result<Vec<EvaluationStrategy>> {
    if values.is_empty() {
        return Ok(vec![
            EvaluationStrategy::Policy,
            EvaluationStrategy::Rank,
            EvaluationStrategy::Opt,
        ]);
    }

    let mut strategies = Vec::with_capacity(values.len());
    for value in values {
        let parsed = EvaluationStrategy::parse(value)?;
        if !strategies.contains(&parsed) {
            strategies.push(parsed);
        }
    }

    Ok(strategies)
}

fn resolve_seeds(args: &BenchmarkArgs, config: &GpcConfig) -> Vec<u64> {
    if !args.seeds.is_empty() {
        return args.seeds.clone();
    }

    vec![
        config.training.seed,
        config.training.seed.wrapping_add(1),
        config.training.seed.wrapping_add(2),
    ]
}

fn summarize_runs(runs: &[BenchmarkRun]) -> Vec<BenchmarkSummary> {
    let mut grouped = BTreeMap::new();
    for run in runs {
        grouped
            .entry(run.report.strategy)
            .or_insert_with(MetricAccumulator::default)
            .push(&run.report);
    }

    let mut summary = grouped
        .into_iter()
        .map(|(strategy, accumulator)| accumulator.finish(strategy))
        .collect::<Vec<_>>();

    let policy_baseline = summary
        .iter()
        .find(|entry| entry.strategy == EvaluationStrategy::Policy)
        .cloned();
    if let Some(policy) = policy_baseline {
        for entry in &mut summary {
            if entry.strategy == EvaluationStrategy::Policy {
                continue;
            }

            entry.delta_vs_policy = Some(BenchmarkDelta {
                mean_rollout_mse: entry.mean_rollout_mse - policy.mean_rollout_mse,
                mean_terminal_distance: entry.mean_terminal_distance
                    - policy.mean_terminal_distance,
                mean_action_mse: entry.mean_action_mse - policy.mean_action_mse,
                mean_reward: entry.mean_reward - policy.mean_reward,
                success_rate: entry.success_rate - policy.success_rate,
            });
        }
    }

    summary
}

fn log_benchmark_report(report: &BenchmarkReport) {
    tracing::info!("Checkpoint benchmark complete");
    tracing::info!("  seeds: {:?}", report.seeds);
    tracing::info!("  episodes per seed: {}", report.episodes_per_seed);
    tracing::info!("  episode length: {}", report.episode_length);
    tracing::info!("  num candidates: {}", report.num_candidates);
    tracing::info!("  opt steps: {}", report.opt_steps);
    tracing::info!("  opt learning rate: {:.6}", report.opt_learning_rate);

    for entry in &report.summary {
        tracing::info!(
            "  [{}] success={:.2}% terminal={:.6} reward={:.6} rollout_mse={:.6} action_mse={:.6}",
            entry.strategy.label(),
            entry.success_rate * 100.0,
            entry.mean_terminal_distance,
            entry.mean_reward,
            entry.mean_rollout_mse,
            entry.mean_action_mse
        );

        if let Some(delta) = &entry.delta_vs_policy {
            tracing::info!(
                "    vs policy: success={:+.2}% terminal={:+.6} reward={:+.6} rollout_mse={:+.6} action_mse={:+.6}",
                delta.success_rate * 100.0,
                delta.mean_terminal_distance,
                delta.mean_reward,
                delta.mean_rollout_mse,
                delta.mean_action_mse
            );
        }
    }
}

fn write_report(path: &str, report: &BenchmarkReport) -> Result<()> {
    let path = PathBuf::from(path);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpc_compat::{
        CheckpointFormat, CheckpointKind, CheckpointMetadata, save_policy_checkpoint,
        save_world_model_checkpoint,
    };
    use gpc_core::config::{NoiseScheduleConfig, PolicyConfig, TrainingConfig, WorldModelConfig};
    use gpc_train::{PolicyTrainer, WorldModelTrainer};
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_benchmark_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gpc_benchmark_{name}_{}_{}",
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
            Err(_) => 7,
        }
    }

    fn current_timestamp() -> String {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => format!("{}Z", duration.as_secs()),
            Err(_) => "0Z".to_string(),
        }
    }

    fn checkpoint_metadata(model_type: CheckpointKind, config_json: String) -> CheckpointMetadata {
        CheckpointMetadata {
            model_type: model_type.as_str().to_string(),
            epoch: 1,
            loss: 0.0,
            timestamp: current_timestamp(),
            config_json,
        }
    }

    fn save_tiny_checkpoints(dir: &Path) -> Result<(PathBuf, PathBuf)> {
        let device = <BenchmarkBackend as Backend>::Device::default();
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
            noise_schedule: NoiseScheduleConfig {
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
            .train_with_summary::<BenchmarkBackend>(&gpc_dataset, &device)
            .model;
        let world_model = WorldModelTrainer::new(training_config, world_model_config.clone())
            .train_phase1_with_summary::<BenchmarkBackend>(&gpc_dataset, &device)
            .model;

        let policy_path = save_policy_checkpoint(
            policy,
            &checkpoint_metadata(
                CheckpointKind::Policy,
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
                serde_json::to_string(&world_model_config)?,
            ),
            dir.join("world_model_final"),
            CheckpointFormat::Bin,
        )?
        .checkpoint_path;

        Ok((policy_path, world_model_path))
    }

    #[test]
    fn benchmark_defaults_include_policy_rank_and_opt() {
        let strategies = parse_strategies(&[]).unwrap();
        assert_eq!(
            strategies,
            vec![
                EvaluationStrategy::Policy,
                EvaluationStrategy::Rank,
                EvaluationStrategy::Opt
            ]
        );
    }

    #[test]
    fn benchmark_runs_end_to_end_and_writes_json() {
        let dir = temp_benchmark_dir("end_to_end");
        let (policy_path, world_model_path) = save_tiny_checkpoints(&dir).unwrap();
        let output_path = dir.join("reports/benchmark.json");

        let args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            strategies: Vec::new(),
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(output_path.to_string_lossy().to_string()),
        };

        run_benchmark(args).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();

        assert_eq!(
            json["strategies"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>(),
            vec!["policy", "rank", "opt"]
        );
        assert_eq!(
            json["seeds"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|value| value.as_u64())
                .collect::<Vec<_>>(),
            vec![11, 17]
        );
        assert_eq!(json["per_run"].as_array().unwrap().len(), 6);
        assert_eq!(json["summary"].as_array().unwrap().len(), 3);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
