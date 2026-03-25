//! Benchmark command implementation.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use clap::Args;
use serde::{Deserialize, Serialize};

use gpc_core::config::GpcConfig;

use super::eval::{
    CheckpointEvalSettings, EvalArgs, EvaluationMode, EvaluationReport, EvaluationStrategy,
    evaluate_synthetic_closed_loop, evaluate_windows, generate_synthetic_episodes,
    load_evaluation_windows, load_models, resolved_policy_checkpoint_path,
    resolved_world_model_checkpoint_path, validate_episodes_for_closed_loop,
};
use super::reporting::{read_json_report, write_json_report};

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

    /// Path to training data directory or episodes.json.
    #[arg(long)]
    data: Option<String>,

    /// Force synthetic benchmarking data even when `--data` is set.
    #[arg(long)]
    synthetic: bool,

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

    /// Optional path to a previous benchmark JSON artifact used as a baseline.
    #[arg(long)]
    baseline: Option<String>,

    /// Maximum allowed increase in mean rollout MSE versus the baseline.
    #[arg(long)]
    max_rollout_mse_increase: Option<f32>,

    /// Maximum allowed increase in mean terminal distance versus the baseline.
    #[arg(long)]
    max_terminal_distance_increase: Option<f32>,

    /// Maximum allowed increase in mean action MSE versus the baseline.
    #[arg(long)]
    max_action_mse_increase: Option<f32>,

    /// Maximum allowed drop in mean reward versus the baseline.
    #[arg(long)]
    max_reward_drop: Option<f32>,

    /// Maximum allowed drop in success rate versus the baseline.
    #[arg(long)]
    max_success_rate_drop: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkRun {
    seed: Option<u64>,
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
    mode: EvaluationMode,
    checkpoint_dir: String,
    policy_checkpoint: Option<String>,
    world_model_checkpoint: Option<String>,
    data: Option<String>,
    strategies: Vec<EvaluationStrategy>,
    seeds: Vec<u64>,
    episodes_per_seed: usize,
    episode_length: usize,
    num_candidates: usize,
    opt_steps: usize,
    opt_learning_rate: f32,
    per_run: Vec<BenchmarkRun>,
    summary: Vec<BenchmarkSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<BenchmarkComparisonReport>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkComparisonReport {
    baseline_path: String,
    baseline_mode: String,
    current_mode: String,
    shared_strategies: usize,
    missing_in_current: Vec<String>,
    missing_in_baseline: Vec<String>,
    thresholds: BenchmarkRegressionThresholds,
    entries: Vec<BenchmarkComparisonEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkComparisonEntry {
    strategy: EvaluationStrategy,
    current: BenchmarkSummarySnapshot,
    baseline: BenchmarkSummarySnapshot,
    delta: BenchmarkDelta,
    regressions: Vec<BenchmarkRegression>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkSummarySnapshot {
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
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkRegression {
    metric: String,
    current: f32,
    baseline: f32,
    regression: f32,
    threshold: f32,
}

#[derive(Debug, Clone, Serialize, Default)]
struct BenchmarkRegressionThresholds {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_rollout_mse_increase: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_terminal_distance_increase: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_action_mse_increase: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_reward_drop: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_success_rate_drop: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LoadedBenchmarkReport {
    mode: String,
    checkpoint_dir: String,
    policy_checkpoint: Option<String>,
    world_model_checkpoint: Option<String>,
    data: Option<String>,
    strategies: Vec<String>,
    seeds: Vec<u64>,
    episodes_per_seed: usize,
    episode_length: usize,
    num_candidates: usize,
    opt_steps: usize,
    opt_learning_rate: f32,
    summary: Vec<LoadedBenchmarkSummary>,
}

#[derive(Debug, Deserialize)]
struct LoadedBenchmarkSummary {
    strategy: String,
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
}

impl From<&BenchmarkSummary> for BenchmarkSummarySnapshot {
    fn from(summary: &BenchmarkSummary) -> Self {
        Self {
            runs: summary.runs,
            episodes_evaluated: summary.episodes_evaluated,
            mean_rollout_mse: summary.mean_rollout_mse,
            mean_terminal_distance: summary.mean_terminal_distance,
            mean_action_mse: summary.mean_action_mse,
            mean_reward: summary.mean_reward,
            success_rate: summary.success_rate,
            num_candidates: summary.num_candidates,
            opt_steps: summary.opt_steps,
            opt_learning_rate: summary.opt_learning_rate,
        }
    }
}

impl BenchmarkRegressionThresholds {
    fn is_empty(&self) -> bool {
        self.max_rollout_mse_increase.is_none()
            && self.max_terminal_distance_increase.is_none()
            && self.max_action_mse_increase.is_none()
            && self.max_reward_drop.is_none()
            && self.max_success_rate_drop.is_none()
    }
}

impl LoadedBenchmarkSummary {
    fn snapshot(&self) -> BenchmarkSummarySnapshot {
        BenchmarkSummarySnapshot {
            runs: self.runs,
            episodes_evaluated: self.episodes_evaluated,
            mean_rollout_mse: self.mean_rollout_mse,
            mean_terminal_distance: self.mean_terminal_distance,
            mean_action_mse: self.mean_action_mse,
            mean_reward: self.mean_reward,
            success_rate: self.success_rate,
            num_candidates: self.num_candidates,
            opt_steps: self.opt_steps,
            opt_learning_rate: self.opt_learning_rate,
        }
    }
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

    let thresholds = regression_thresholds(&args);
    validate_threshold_values(&thresholds)?;

    if !thresholds.is_empty() && args.baseline.is_none() {
        anyhow::bail!("regression thresholds require --baseline");
    }

    let strategies = parse_strategies(&args.strategies)?;
    let mode = if args.synthetic || args.data.is_none() {
        EvaluationMode::SyntheticClosedLoop
    } else {
        EvaluationMode::DatasetOpenLoop
    };
    let seeds = resolve_seeds(&args, &config, mode);
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
        data: args.data.clone(),
        synthetic: args.synthetic,
        episodes: args.episodes,
        episode_length: args.episode_length,
        num_candidates: Some(num_candidates),
        opt_steps: Some(opt_steps),
        opt_learning_rate: Some(opt_learning_rate),
        seed: None,
        output: None,
        details_output: None,
        demo: false,
    };

    let device = <BenchmarkBackend as Backend>::Device::default();
    let models = load_models(&eval_args, &device)?;
    let mut per_run = Vec::with_capacity(strategies.len() * seeds.len().max(1));
    let episode_length = args
        .episode_length
        .max(models.policy_config.pred_horizon + 1);
    let episodes_per_seed = args.episodes.max(1);

    match mode {
        EvaluationMode::SyntheticClosedLoop => {
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
                    per_run.push(BenchmarkRun {
                        seed: Some(seed),
                        report,
                    });
                }
            }
        }
        EvaluationMode::DatasetOpenLoop => {
            let windows = load_evaluation_windows(&eval_args, &models)?;
            for strategy in &strategies {
                let report = evaluate_windows(
                    &models,
                    &windows,
                    *strategy,
                    num_candidates,
                    opt_steps,
                    opt_learning_rate,
                    &device,
                )?;
                per_run.push(BenchmarkRun { seed: None, report });
            }
        }
    }

    let summary = summarize_runs(&per_run);
    let mut report = BenchmarkReport {
        mode,
        checkpoint_dir: args.checkpoint_dir.clone(),
        policy_checkpoint: Some(
            resolved_policy_checkpoint_path(&eval_args)
                .to_string_lossy()
                .to_string(),
        ),
        world_model_checkpoint: Some(
            resolved_world_model_checkpoint_path(&eval_args)
                .to_string_lossy()
                .to_string(),
        ),
        data: args.data.clone(),
        strategies,
        seeds,
        episodes_per_seed,
        episode_length,
        num_candidates,
        opt_steps,
        opt_learning_rate,
        per_run,
        summary,
        comparison: None,
    };

    if let Some(baseline_path) = &args.baseline {
        report.comparison = Some(compare_against_baseline(
            baseline_path,
            &report,
            &thresholds,
        )?);
    }

    validate_benchmark_regressions(&report)?;

    log_benchmark_report(&report);

    if let Some(output_path) = &args.output {
        write_json_report(output_path, &report)?;
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

fn resolve_seeds(args: &BenchmarkArgs, config: &GpcConfig, mode: EvaluationMode) -> Vec<u64> {
    if mode == EvaluationMode::DatasetOpenLoop {
        return Vec::new();
    }

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

fn regression_thresholds(args: &BenchmarkArgs) -> BenchmarkRegressionThresholds {
    BenchmarkRegressionThresholds {
        max_rollout_mse_increase: args.max_rollout_mse_increase,
        max_terminal_distance_increase: args.max_terminal_distance_increase,
        max_action_mse_increase: args.max_action_mse_increase,
        max_reward_drop: args.max_reward_drop,
        max_success_rate_drop: args.max_success_rate_drop,
    }
}

fn compare_against_baseline(
    baseline_path: &str,
    current: &BenchmarkReport,
    thresholds: &BenchmarkRegressionThresholds,
) -> Result<BenchmarkComparisonReport> {
    let baseline: LoadedBenchmarkReport = read_json_report(baseline_path)?;
    validate_benchmark_compatibility(current, &baseline, baseline_path)?;
    let baseline_mode = baseline.mode;
    let baseline_summary = baseline.summary;

    let mut current_map = BTreeMap::new();
    for summary in &current.summary {
        current_map.insert(summary.strategy, summary);
    }

    let mut baseline_map = BTreeMap::new();
    for summary in baseline_summary {
        let strategy = EvaluationStrategy::parse(&summary.strategy)?;
        if baseline_map.insert(strategy, summary).is_some() {
            anyhow::bail!(
                "baseline report contains duplicate strategy '{}'",
                strategy.label()
            );
        }
    }

    let current_keys: BTreeSet<_> = current_map.keys().copied().collect();
    let baseline_keys: BTreeSet<_> = baseline_map.keys().copied().collect();
    let shared_keys = current_keys
        .intersection(&baseline_keys)
        .copied()
        .collect::<Vec<_>>();

    let mut entries = Vec::with_capacity(shared_keys.len());
    for strategy in shared_keys {
        let current = current_map[&strategy];
        let baseline = &baseline_map[&strategy];
        let regressions = collect_regressions(current, baseline, thresholds);

        entries.push(BenchmarkComparisonEntry {
            strategy,
            current: BenchmarkSummarySnapshot::from(current),
            baseline: baseline.snapshot(),
            delta: BenchmarkDelta {
                mean_rollout_mse: current.mean_rollout_mse - baseline.mean_rollout_mse,
                mean_terminal_distance: current.mean_terminal_distance
                    - baseline.mean_terminal_distance,
                mean_action_mse: current.mean_action_mse - baseline.mean_action_mse,
                mean_reward: current.mean_reward - baseline.mean_reward,
                success_rate: current.success_rate - baseline.success_rate,
            },
            regressions,
        });
    }

    Ok(BenchmarkComparisonReport {
        baseline_path: baseline_path.to_string(),
        baseline_mode,
        current_mode: current.mode.label().to_string(),
        shared_strategies: entries.len(),
        missing_in_current: Vec::new(),
        missing_in_baseline: Vec::new(),
        thresholds: (*thresholds).clone(),
        entries,
    })
}

fn validate_threshold_values(thresholds: &BenchmarkRegressionThresholds) -> Result<()> {
    let check = |name: &str, value: Option<f32>| -> Result<()> {
        if let Some(value) = value {
            if !value.is_finite() {
                anyhow::bail!("{} must be finite", name);
            }
            if value < 0.0 {
                anyhow::bail!("{} must be non-negative", name);
            }
        }
        Ok(())
    };

    check(
        "max_rollout_mse_increase",
        thresholds.max_rollout_mse_increase,
    )?;
    check(
        "max_terminal_distance_increase",
        thresholds.max_terminal_distance_increase,
    )?;
    check(
        "max_action_mse_increase",
        thresholds.max_action_mse_increase,
    )?;
    check("max_reward_drop", thresholds.max_reward_drop)?;
    check("max_success_rate_drop", thresholds.max_success_rate_drop)?;
    Ok(())
}

fn validate_benchmark_compatibility(
    current: &BenchmarkReport,
    baseline: &LoadedBenchmarkReport,
    baseline_path: &str,
) -> Result<()> {
    let mut mismatches = Vec::new();

    if baseline.mode != current.mode.label() {
        mismatches.push(format!(
            "mode (baseline '{}' vs current '{}')",
            baseline.mode,
            current.mode.label()
        ));
    }
    if baseline.checkpoint_dir != current.checkpoint_dir {
        mismatches.push(format!(
            "checkpoint_dir (baseline '{}' vs current '{}')",
            baseline.checkpoint_dir, current.checkpoint_dir
        ));
    }
    if baseline.policy_checkpoint != current.policy_checkpoint {
        mismatches.push(format!(
            "policy_checkpoint (baseline '{:?}' vs current '{:?}')",
            baseline.policy_checkpoint, current.policy_checkpoint
        ));
    }
    if baseline.world_model_checkpoint != current.world_model_checkpoint {
        mismatches.push(format!(
            "world_model_checkpoint (baseline '{:?}' vs current '{:?}')",
            baseline.world_model_checkpoint, current.world_model_checkpoint
        ));
    }
    if baseline.data != current.data {
        mismatches.push(format!(
            "data (baseline '{:?}' vs current '{:?}')",
            baseline.data, current.data
        ));
    }

    let current_strategies = normalize_strategy_labels(&current.strategies);
    let baseline_strategies =
        normalize_strategy_labels(&parse_strategies_from_labels(&baseline.strategies)?);
    if baseline_strategies != current_strategies {
        mismatches.push(format!(
            "strategies (baseline {:?} vs current {:?})",
            baseline_strategies, current_strategies
        ));
    }

    let current_seeds = normalize_seeds(&current.seeds);
    let baseline_seeds = normalize_seeds(&baseline.seeds);
    if baseline_seeds != current_seeds {
        mismatches.push(format!(
            "seeds (baseline {:?} vs current {:?})",
            baseline_seeds, current_seeds
        ));
    }

    if baseline.episodes_per_seed != current.episodes_per_seed {
        mismatches.push(format!(
            "episodes_per_seed (baseline {} vs current {})",
            baseline.episodes_per_seed, current.episodes_per_seed
        ));
    }
    if baseline.episode_length != current.episode_length {
        mismatches.push(format!(
            "episode_length (baseline {} vs current {})",
            baseline.episode_length, current.episode_length
        ));
    }
    if baseline.num_candidates != current.num_candidates {
        mismatches.push(format!(
            "num_candidates (baseline {} vs current {})",
            baseline.num_candidates, current.num_candidates
        ));
    }
    if baseline.opt_steps != current.opt_steps {
        mismatches.push(format!(
            "opt_steps (baseline {} vs current {})",
            baseline.opt_steps, current.opt_steps
        ));
    }
    if (baseline.opt_learning_rate - current.opt_learning_rate).abs() > f32::EPSILON {
        mismatches.push(format!(
            "opt_learning_rate (baseline {} vs current {})",
            baseline.opt_learning_rate, current.opt_learning_rate
        ));
    }

    if mismatches.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "baseline '{}' is not compatible with the current benchmark suite: {}",
        baseline_path,
        mismatches.join(", ")
    );
}

fn normalize_strategy_labels(strategies: &[EvaluationStrategy]) -> Vec<String> {
    let mut labels = strategies
        .iter()
        .map(|strategy| strategy.label().to_string())
        .collect::<Vec<_>>();
    labels.sort();
    labels
}

fn normalize_seeds(seeds: &[u64]) -> Vec<u64> {
    let mut seeds = seeds.to_vec();
    seeds.sort_unstable();
    seeds
}

fn parse_strategies_from_labels(values: &[String]) -> Result<Vec<EvaluationStrategy>> {
    values
        .iter()
        .map(|value| EvaluationStrategy::parse(value))
        .collect()
}

fn collect_regressions(
    current: &BenchmarkSummary,
    baseline: &LoadedBenchmarkSummary,
    thresholds: &BenchmarkRegressionThresholds,
) -> Vec<BenchmarkRegression> {
    let mut regressions = Vec::new();

    if let Some(threshold) = thresholds.max_rollout_mse_increase {
        push_increase_regression(
            &mut regressions,
            "mean_rollout_mse",
            current.mean_rollout_mse,
            baseline.mean_rollout_mse,
            threshold,
        );
    }
    if let Some(threshold) = thresholds.max_terminal_distance_increase {
        push_increase_regression(
            &mut regressions,
            "mean_terminal_distance",
            current.mean_terminal_distance,
            baseline.mean_terminal_distance,
            threshold,
        );
    }
    if let Some(threshold) = thresholds.max_action_mse_increase {
        push_increase_regression(
            &mut regressions,
            "mean_action_mse",
            current.mean_action_mse,
            baseline.mean_action_mse,
            threshold,
        );
    }
    if let Some(threshold) = thresholds.max_reward_drop {
        push_drop_regression(
            &mut regressions,
            "mean_reward",
            current.mean_reward,
            baseline.mean_reward,
            threshold,
        );
    }
    if let Some(threshold) = thresholds.max_success_rate_drop {
        push_drop_regression(
            &mut regressions,
            "success_rate",
            current.success_rate,
            baseline.success_rate,
            threshold,
        );
    }

    regressions
}

fn push_increase_regression(
    regressions: &mut Vec<BenchmarkRegression>,
    metric: &'static str,
    current: f32,
    baseline: f32,
    threshold: f32,
) {
    let regression = current - baseline;
    if regression > threshold {
        regressions.push(BenchmarkRegression {
            metric: metric.to_string(),
            current,
            baseline,
            regression,
            threshold,
        });
    }
}

fn push_drop_regression(
    regressions: &mut Vec<BenchmarkRegression>,
    metric: &'static str,
    current: f32,
    baseline: f32,
    threshold: f32,
) {
    let regression = baseline - current;
    if regression > threshold {
        regressions.push(BenchmarkRegression {
            metric: metric.to_string(),
            current,
            baseline,
            regression,
            threshold,
        });
    }
}

fn validate_benchmark_regressions(report: &BenchmarkReport) -> Result<()> {
    let Some(comparison) = &report.comparison else {
        return Ok(());
    };

    if comparison.thresholds.is_empty() {
        return Ok(());
    }

    let mut violations = Vec::new();
    for entry in &comparison.entries {
        for regression in &entry.regressions {
            violations.push(format!(
                "[{}] {} regressed by {:.6} (baseline {:.6} -> current {:.6}, threshold {:.6})",
                entry.strategy.label(),
                regression.metric,
                regression.regression,
                regression.baseline,
                regression.current,
                regression.threshold
            ));
        }
    }

    if !violations.is_empty() {
        anyhow::bail!(
            "benchmark regression thresholds exceeded:\n{}",
            violations.join("\n")
        );
    }

    Ok(())
}

fn log_benchmark_report(report: &BenchmarkReport) {
    tracing::info!("Checkpoint benchmark complete");
    tracing::info!("  mode: {}", report.mode.label());
    if let Some(data) = &report.data {
        tracing::info!("  data: {}", data);
    }
    if report.mode == EvaluationMode::SyntheticClosedLoop {
        tracing::info!("  seeds: {:?}", report.seeds);
        tracing::info!("  episodes per seed: {}", report.episodes_per_seed);
        tracing::info!("  episode length: {}", report.episode_length);
    }
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

    if let Some(comparison) = &report.comparison {
        tracing::info!("  baseline: {}", comparison.baseline_path);
        tracing::info!("  baseline mode: {}", comparison.baseline_mode);
        tracing::info!("  shared strategies: {}", comparison.shared_strategies);
        if !comparison.missing_in_current.is_empty() {
            tracing::info!(
                "  baseline-only strategies: {:?}",
                comparison.missing_in_current
            );
        }
        if !comparison.missing_in_baseline.is_empty() {
            tracing::info!(
                "  current-only strategies: {:?}",
                comparison.missing_in_baseline
            );
        }
        for entry in &comparison.entries {
            tracing::info!(
                "  [{}] baseline delta reward={:+.6} rollout_mse={:+.6} success={:+.2}%",
                entry.strategy.label(),
                entry.delta.mean_reward,
                entry.delta.mean_rollout_mse,
                entry.delta.success_rate * 100.0
            );
            for regression in &entry.regressions {
                tracing::info!(
                    "    regression {} by {:.6} (threshold {:.6})",
                    regression.metric,
                    regression.regression,
                    regression.threshold
                );
            }
        }
    }
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
    use std::path::{Path, PathBuf};
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
            data: None,
            synthetic: false,
            strategies: Vec::new(),
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(output_path.to_string_lossy().to_string()),
            baseline: None,
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: None,
            max_success_rate_drop: None,
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
        assert_eq!(json["mode"].as_str().unwrap(), "synthetic-closed-loop");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn benchmark_supports_dataset_open_loop_mode() {
        let dir = temp_benchmark_dir("dataset_mode");
        let (policy_path, world_model_path) = save_tiny_checkpoints(&dir).unwrap();
        let data_path = dir.join("episodes.json");
        let output_path = dir.join("reports/dataset_benchmark.json");
        let episodes = generate_synthetic_episodes(4, 2, 4, 8, 2, 23);
        std::fs::write(&data_path, serde_json::to_vec(&episodes).unwrap()).unwrap();

        let args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: Some(data_path.to_string_lossy().to_string()),
            synthetic: false,
            strategies: vec!["rank".to_string(), "opt".to_string()],
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(output_path.to_string_lossy().to_string()),
            baseline: None,
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: None,
            max_success_rate_drop: None,
        };

        run_benchmark(args).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();

        assert_eq!(json["mode"].as_str().unwrap(), "dataset-open-loop");
        assert_eq!(json["seeds"].as_array().unwrap().len(), 0);
        assert_eq!(json["per_run"].as_array().unwrap().len(), 2);
        assert_eq!(json["summary"].as_array().unwrap().len(), 2);
        assert!(
            json["per_run"]
                .as_array()
                .unwrap()
                .iter()
                .all(|run| run["seed"].is_null())
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn benchmark_reports_resolved_checkpoint_paths() {
        let dir = temp_benchmark_dir("resolved_paths");
        let (policy_path, world_model_path) = save_tiny_checkpoints(&dir).unwrap();
        let output_path = dir.join("reports/resolved_paths.json");

        let args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: None,
            world_model_checkpoint: None,
            data: None,
            synthetic: true,
            strategies: vec!["rank".to_string()],
            seeds: vec![11],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(output_path.to_string_lossy().to_string()),
            baseline: None,
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: None,
            max_success_rate_drop: None,
        };

        run_benchmark(args).unwrap();

        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&output_path).unwrap()).unwrap();
        assert_eq!(
            json["policy_checkpoint"].as_str().unwrap(),
            policy_path.to_string_lossy()
        );
        assert_eq!(
            json["world_model_checkpoint"].as_str().unwrap(),
            world_model_path.to_string_lossy()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn benchmark_compares_against_baseline_and_enforces_thresholds() {
        let dir = temp_benchmark_dir("baseline_compare");
        let (policy_path, world_model_path) = save_tiny_checkpoints(&dir).unwrap();
        let baseline_path = dir.join("reports/baseline.json");
        let comparison_path = dir.join("reports/comparison.json");

        let base_args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: false,
            strategies: Vec::new(),
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(baseline_path.to_string_lossy().to_string()),
            baseline: None,
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: None,
            max_success_rate_drop: None,
        };

        run_benchmark(base_args).unwrap();

        let comparison_args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: false,
            strategies: Vec::new(),
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(comparison_path.to_string_lossy().to_string()),
            baseline: Some(baseline_path.to_string_lossy().to_string()),
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: None,
            max_success_rate_drop: None,
        };

        run_benchmark(comparison_args).unwrap();

        let comparison_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&comparison_path).unwrap()).unwrap();
        let comparison = &comparison_json["comparison"];
        assert_eq!(
            comparison["baseline_path"].as_str().unwrap(),
            baseline_path.to_string_lossy()
        );
        assert_eq!(comparison["shared_strategies"].as_u64().unwrap(), 3);
        assert!(
            comparison["missing_in_current"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            comparison["missing_in_baseline"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(comparison["entries"].as_array().unwrap().len(), 3);
        assert!(
            comparison["entries"]
                .as_array()
                .unwrap()
                .iter()
                .all(|entry| entry["regressions"].as_array().unwrap().is_empty())
        );

        let mut baseline_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&baseline_path).unwrap()).unwrap();
        let summaries = baseline_json["summary"].as_array_mut().unwrap();
        for entry in summaries {
            if entry["strategy"].as_str() == Some("rank") {
                let boosted = entry["mean_reward"].as_f64().unwrap() + 1.0;
                entry["mean_reward"] = serde_json::Value::from(boosted);
            }
        }
        std::fs::write(
            &baseline_path,
            serde_json::to_vec_pretty(&baseline_json).unwrap(),
        )
        .unwrap();

        let failing_args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: false,
            strategies: Vec::new(),
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(
                dir.join("reports/failure.json")
                    .to_string_lossy()
                    .to_string(),
            ),
            baseline: Some(baseline_path.to_string_lossy().to_string()),
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: Some(0.0),
            max_success_rate_drop: None,
        };

        let err = run_benchmark(failing_args).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("benchmark regression thresholds exceeded"));
        assert!(message.contains("[rank] mean_reward"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn benchmark_rejects_incompatible_baseline_suite() {
        let dir = temp_benchmark_dir("baseline_mismatch");
        let (policy_path, world_model_path) = save_tiny_checkpoints(&dir).unwrap();
        let baseline_path = dir.join("reports/baseline.json");

        let args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: false,
            strategies: Vec::new(),
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(baseline_path.to_string_lossy().to_string()),
            baseline: None,
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: None,
            max_success_rate_drop: None,
        };

        run_benchmark(args).unwrap();

        let mut baseline_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&baseline_path).unwrap()).unwrap();
        baseline_json["episode_length"] = serde_json::Value::from(99);
        std::fs::write(
            &baseline_path,
            serde_json::to_vec_pretty(&baseline_json).unwrap(),
        )
        .unwrap();

        let comparison_args = BenchmarkArgs {
            config: None,
            checkpoint_dir: dir.to_string_lossy().to_string(),
            policy_checkpoint: Some(policy_path.to_string_lossy().to_string()),
            world_model_checkpoint: Some(world_model_path.to_string_lossy().to_string()),
            data: None,
            synthetic: false,
            strategies: Vec::new(),
            seeds: vec![11, 17],
            episodes: 2,
            episode_length: 8,
            num_candidates: Some(4),
            opt_steps: Some(2),
            opt_learning_rate: Some(0.05),
            output: Some(
                dir.join("reports/mismatch.json")
                    .to_string_lossy()
                    .to_string(),
            ),
            baseline: Some(baseline_path.to_string_lossy().to_string()),
            max_rollout_mse_increase: None,
            max_terminal_distance_increase: None,
            max_action_mse_increase: None,
            max_reward_drop: None,
            max_success_rate_drop: None,
        };

        let err = run_benchmark(comparison_args).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("not compatible"));
        assert!(message.contains("episode_length"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn benchmark_rejects_non_finite_thresholds() {
        let nan_thresholds = BenchmarkRegressionThresholds {
            max_rollout_mse_increase: Some(f32::NAN),
            ..Default::default()
        };
        let err = validate_threshold_values(&nan_thresholds).unwrap_err();
        assert!(format!("{err:#}").contains("must be finite"));

        let inf_thresholds = BenchmarkRegressionThresholds {
            max_reward_drop: Some(f32::INFINITY),
            ..Default::default()
        };
        let err = validate_threshold_values(&inf_thresholds).unwrap_err();
        assert!(format!("{err:#}").contains("must be finite"));
    }
}
