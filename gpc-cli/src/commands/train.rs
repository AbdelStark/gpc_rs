//! Train command implementation.

use anyhow::Result;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use clap::Args;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use gpc_core::config::{GpcConfig, PolicyConfig, TrainingConfig, WorldModelConfig};
use gpc_train::{
    GpcDataset, GpcDatasetConfig, PolicyTrainer, PolicyValidationSummary, WorldModelTrainer,
    WorldModelValidationSummary,
};

use super::reporting::{default_train_report_path, write_json_report};
use gpc_compat::{
    CheckpointArtifact, CheckpointFormat, CheckpointKind, CheckpointMetadata,
    save_policy_checkpoint, save_world_model_checkpoint, verify_checkpoint_artifact,
};

type TrainBackend = Autodiff<NdArray>;

struct TrainingReportContext<'a> {
    args: &'a TrainArgs,
    config: &'a GpcConfig,
    training_config: &'a TrainingConfig,
    dataset_config: &'a GpcDatasetConfig,
    dataset: &'a GpcDataset,
    dataset_split: Option<&'a gpc_train::data::GpcDatasetSplit>,
    policy: Option<PolicyTrainingReport>,
    world_model: Option<WorldModelTrainingReport>,
    artifacts: &'a [CheckpointArtifact],
}

/// Arguments for the train command.
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to configuration file (JSON).
    #[arg(short, long)]
    config: Option<String>,

    /// Path to training data directory.
    #[arg(short, long)]
    data: Option<String>,

    /// Component to train: "policy", "world-model", or "all".
    #[arg(long, default_value = "all")]
    component: String,

    /// Number of training epochs (overrides config).
    #[arg(long)]
    epochs: Option<usize>,

    /// Batch size for training.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Learning rate for AdamW.
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Weight decay for AdamW.
    #[arg(long)]
    weight_decay: Option<f64>,

    /// Gradient clipping max norm.
    #[arg(long)]
    grad_clip_norm: Option<f64>,

    /// Warmup steps for the learning-rate schedule.
    #[arg(long)]
    warmup_steps: Option<usize>,

    /// How often to save checkpoints, in epochs.
    #[arg(long)]
    checkpoint_every: Option<usize>,

    /// How often to log metrics, in steps.
    #[arg(long)]
    log_every: Option<usize>,

    /// Random seed for dataset generation and training.
    #[arg(long)]
    seed: Option<u64>,

    /// Use synthetic data for testing.
    #[arg(long)]
    synthetic: bool,

    /// Output directory for checkpoints.
    #[arg(short, long, default_value = "checkpoints")]
    output: String,

    /// World model prediction horizon for phase 2.
    #[arg(long, default_value = "8")]
    horizon: usize,

    /// Fraction of episodes to hold out for validation.
    #[arg(long, default_value = "0.0")]
    validation_split: f32,

    /// Optional output path for a machine-readable JSON training report.
    #[arg(long)]
    report_output: Option<String>,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            config: None,
            data: None,
            component: "all".to_string(),
            epochs: None,
            batch_size: None,
            learning_rate: None,
            weight_decay: None,
            grad_clip_norm: None,
            warmup_steps: None,
            checkpoint_every: None,
            log_every: None,
            seed: None,
            synthetic: false,
            output: "checkpoints".to_string(),
            horizon: 8,
            validation_split: 0.0,
            report_output: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingReport {
    schema_version: u32,
    generated_at: String,
    request: TrainingRequestReport,
    settings: EffectiveTrainingSettingsReport,
    dataset: DatasetReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy: Option<PolicyTrainingReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    world_model: Option<WorldModelTrainingReport>,
    artifacts: Vec<WrittenCheckpointReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingRequestReport {
    component_requested: String,
    component_resolved: String,
    synthetic_requested: bool,
    config_path: Option<String>,
    data_path: Option<String>,
    output_dir: String,
    report_output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EffectiveTrainingSettingsReport {
    training: TrainingConfig,
    policy: PolicyConfig,
    world_model: WorldModelConfig,
    dataset: GpcDatasetConfig,
    horizon: usize,
    validation_split: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetReport {
    source: DatasetSourceReport,
    total_episodes: usize,
    total_transitions: usize,
    train_episodes: usize,
    train_transitions: usize,
    validation_episodes: usize,
    validation_transitions: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    split: Option<DatasetSplitReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetSourceReport {
    synthetic: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetSplitReport {
    validation_split: f32,
    train_episodes: usize,
    train_transitions: usize,
    validation_episodes: usize,
    validation_transitions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingStageReport {
    final_epoch: Option<usize>,
    final_loss: Option<f32>,
    epoch_losses: Vec<f32>,
    validation_losses: Vec<f32>,
    best_epoch: Option<usize>,
    best_validation_loss: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyTrainingReport {
    training: TrainingStageReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorldModelTrainingReport {
    phase1: TrainingStageReport,
    phase2: TrainingStageReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WrittenCheckpointReport {
    name: String,
    checkpoint_path: String,
    metadata_path: String,
}

/// Run the train command.
pub fn run_train(args: TrainArgs) -> Result<()> {
    validate_validation_split(args.validation_split)?;
    let report_output = args
        .report_output
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_train_report_path(&args.output));

    let config = if let Some(config_path) = &args.config {
        let data = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&data)?
    } else {
        GpcConfig::default()
    };

    let mut config = config;
    apply_training_overrides(&mut config.training, &args);
    config.validate()?;
    let training_config = config.training.clone();

    let device = <TrainBackend as burn::prelude::Backend>::Device::default();

    // Load or generate dataset
    let dataset_config = GpcDatasetConfig {
        data_dir: args.data.clone().unwrap_or_else(|| "data".to_string()),
        state_dim: config.world_model.state_dim,
        action_dim: config.policy.action_dim,
        obs_dim: config.policy.obs_dim,
        obs_horizon: config.policy.obs_horizon,
        pred_horizon: config.policy.pred_horizon,
    };

    let dataset = if args.synthetic {
        tracing::info!("Generating synthetic dataset");
        GpcDataset::generate_synthetic(dataset_config.clone(), 50, 100, training_config.seed)
    } else if let Some(data_path) = &args.data {
        GpcDataset::from_path(data_path, dataset_config.clone())?
    } else {
        tracing::info!("No data path specified, using synthetic data");
        GpcDataset::generate_synthetic(dataset_config.clone(), 50, 100, training_config.seed)
    };

    let validation_report = dataset.validate(&config.policy, &config.world_model)?;
    tracing::info!(
        episodes = validation_report.episode_count,
        transitions = validation_report.transition_count,
        open_loop_windows = validation_report.open_loop_window_count,
        closed_loop_compatible_episodes = validation_report.closed_loop_compatible_episode_count,
        "dataset validated"
    );

    if validation_report.episode_count == 0 {
        anyhow::bail!("dataset does not contain any episodes");
    }

    let trains_policy = matches!(args.component.as_str(), "policy" | "all");
    if trains_policy && validation_report.open_loop_window_count == 0 {
        anyhow::bail!("dataset does not contain any usable open-loop windows for policy training");
    }

    tracing::info!(
        "Dataset loaded: {} episodes, {} transitions",
        dataset.num_episodes(),
        dataset.num_transitions()
    );

    std::fs::create_dir_all(&args.output)?;
    if args.validation_split == 0.0 {
        cleanup_stale_best_artifacts(&args.output, &args.component)?;
    }

    let dataset_split = if args.validation_split > 0.0 {
        Some(dataset.split(args.validation_split, training_config.seed)?)
    } else {
        None
    };
    let train_dataset = dataset_split
        .as_ref()
        .map(|split| &split.train)
        .unwrap_or(&dataset);
    let validation_dataset = dataset_split.as_ref().and_then(|split| {
        if split.validation.num_episodes() > 0 {
            Some(&split.validation)
        } else {
            None
        }
    });

    if let Some(split) = &dataset_split {
        tracing::info!(
            train_episodes = split.train.num_episodes(),
            validation_episodes = split.validation.num_episodes(),
            validation_split = args.validation_split,
            "dataset split into train and validation subsets"
        );
    }

    let (policy_report, world_model_report, artifacts) = match args.component.as_str() {
        "world-model" | "wm" => {
            let (world_model_report, artifacts) = train_world_model(
                &training_config,
                &config.world_model,
                train_dataset,
                validation_dataset,
                &device,
                &args,
            )?;
            (None, Some(world_model_report), artifacts)
        }
        "policy" => {
            let (policy_report, artifacts) = train_policy(
                &training_config,
                &config.policy,
                train_dataset,
                validation_dataset,
                &device,
                &args.output,
            )?;
            (Some(policy_report), None, artifacts)
        }
        "all" => {
            let (world_model_report, mut artifacts) = train_world_model(
                &training_config,
                &config.world_model,
                train_dataset,
                validation_dataset,
                &device,
                &args,
            )?;
            let (policy_report, policy_artifacts) = train_policy(
                &training_config,
                &config.policy,
                train_dataset,
                validation_dataset,
                &device,
                &args.output,
            )?;
            artifacts.extend(policy_artifacts);
            (Some(policy_report), Some(world_model_report), artifacts)
        }
        other => {
            anyhow::bail!("Unknown component: {other}. Use 'policy', 'world-model', or 'all'.");
        }
    };

    let report = build_training_report(TrainingReportContext {
        args: &args,
        config: &config,
        training_config: &training_config,
        dataset_config: &dataset_config,
        dataset: &dataset,
        dataset_split: dataset_split.as_ref(),
        policy: policy_report,
        world_model: world_model_report,
        artifacts: &artifacts,
    });
    write_json_report(&report_output, &report)?;

    log_artifacts(&artifacts);
    tracing::info!("Training report written to {}", report_output.display());

    tracing::info!("Training complete!");
    Ok(())
}

fn train_world_model(
    training_config: &TrainingConfig,
    world_model_config: &WorldModelConfig,
    dataset: &GpcDataset,
    validation_dataset: Option<&GpcDataset>,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    args: &TrainArgs,
) -> Result<(WorldModelTrainingReport, Vec<CheckpointArtifact>)> {
    let trainer = WorldModelTrainer::new(training_config.clone(), world_model_config.clone());

    tracing::info!("=== Phase 1: Single-step world model training ===");
    let phase1_summary = trainer.train_phase1_with_validation_summary::<TrainBackend>(
        dataset,
        validation_dataset,
        device,
    );
    let phase1_artifacts = save_world_model_validation_stages(
        &phase1_summary,
        world_model_config,
        "world_model_phase1",
        &args.output,
        device,
    )?;

    let phase1_model = if validation_dataset.is_some() {
        phase1_summary.best_model.clone()
    } else {
        phase1_summary.training.model.clone()
    };

    tracing::info!("=== Phase 2: Multi-step world model training ===");
    let phase2_summary = trainer.train_phase2_with_validation_summary::<TrainBackend>(
        dataset,
        phase1_model,
        args.horizon,
        validation_dataset,
        device,
    );
    let phase2_artifacts = save_world_model_validation_stages(
        &phase2_summary,
        world_model_config,
        "world_model_final",
        &args.output,
        device,
    )?;

    tracing::info!("World model training complete");
    let mut artifacts = phase1_artifacts;
    artifacts.extend(phase2_artifacts);
    Ok((
        WorldModelTrainingReport {
            phase1: training_stage_report(
                phase1_summary.training.final_epoch,
                phase1_summary.training.final_loss,
                &phase1_summary.training.epoch_losses,
                &phase1_summary.validation_losses,
                phase1_summary.best_epoch,
                phase1_summary.best_validation_loss,
            ),
            phase2: training_stage_report(
                phase2_summary.training.final_epoch,
                phase2_summary.training.final_loss,
                &phase2_summary.training.epoch_losses,
                &phase2_summary.validation_losses,
                phase2_summary.best_epoch,
                phase2_summary.best_validation_loss,
            ),
        },
        artifacts,
    ))
}

fn train_policy(
    training_config: &TrainingConfig,
    policy_config: &PolicyConfig,
    dataset: &GpcDataset,
    validation_dataset: Option<&GpcDataset>,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
    output_dir: &str,
) -> Result<(PolicyTrainingReport, Vec<CheckpointArtifact>)> {
    let trainer = PolicyTrainer::new(training_config.clone(), policy_config.clone());

    tracing::info!("=== Diffusion policy training ===");
    let summary =
        trainer.train_with_validation_summary::<TrainBackend>(dataset, validation_dataset, device);
    let artifacts = save_policy_validation_stages(&summary, policy_config, output_dir, device)?;

    tracing::info!("Policy training complete");
    Ok((
        PolicyTrainingReport {
            training: training_stage_report(
                summary.training.final_epoch,
                summary.training.final_loss,
                &summary.training.epoch_losses,
                &summary.validation_losses,
                summary.best_epoch,
                summary.best_validation_loss,
            ),
        },
        artifacts,
    ))
}

fn save_policy_validation_stages(
    summary: &PolicyValidationSummary<TrainBackend>,
    policy_config: &PolicyConfig,
    output_dir: &str,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
) -> Result<Vec<CheckpointArtifact>> {
    let mut artifacts = Vec::new();
    artifacts.push(save_policy_stage(
        CheckpointKind::Policy,
        &summary.training.model,
        policy_config,
        "policy_final",
        output_dir,
        summary.training.final_epoch,
        summary.training.final_loss,
        device,
    )?);

    if summary.best_epoch.is_some() {
        artifacts.push(save_policy_stage(
            CheckpointKind::Policy,
            &summary.best_model,
            policy_config,
            "policy_best",
            output_dir,
            summary.best_epoch,
            summary.best_validation_loss.or(summary.training.final_loss),
            device,
        )?);
    } else {
        remove_checkpoint_artifact(&Path::new(output_dir).join("policy_best.bin"))?;
    }

    Ok(artifacts)
}

fn save_world_model_validation_stages(
    summary: &WorldModelValidationSummary<TrainBackend>,
    world_model_config: &WorldModelConfig,
    base_name: &str,
    output_dir: &str,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
) -> Result<Vec<CheckpointArtifact>> {
    let mut artifacts = Vec::new();
    artifacts.push(save_world_model_stage(
        CheckpointKind::WorldModel,
        &summary.training.model,
        world_model_config,
        base_name,
        output_dir,
        summary.training.final_epoch,
        summary.training.final_loss,
        device,
    )?);

    if summary.best_epoch.is_some() {
        let best_name = if base_name.ends_with("_phase1") {
            "world_model_phase1_best"
        } else {
            "world_model_best"
        };
        artifacts.push(save_world_model_stage(
            CheckpointKind::WorldModel,
            &summary.best_model,
            world_model_config,
            best_name,
            output_dir,
            summary.best_epoch,
            summary.best_validation_loss.or(summary.training.final_loss),
            device,
        )?);
    } else if base_name.ends_with("_phase1") {
        remove_checkpoint_artifact(&Path::new(output_dir).join("world_model_phase1_best.bin"))?;
    } else {
        remove_checkpoint_artifact(&Path::new(output_dir).join("world_model_best.bin"))?;
    }

    Ok(artifacts)
}

#[allow(clippy::too_many_arguments)]
fn save_policy_stage(
    model_type: CheckpointKind,
    model: &gpc_policy::DiffusionPolicy<TrainBackend>,
    policy_config: &PolicyConfig,
    name: &str,
    output_dir: &str,
    epoch: Option<usize>,
    loss: Option<f32>,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
) -> Result<CheckpointArtifact> {
    let metadata = checkpoint_metadata(
        model_type,
        epoch,
        loss,
        serde_json::to_string_pretty(policy_config)?,
    );
    let checkpoint = save_policy_checkpoint::<TrainBackend>(
        model.clone(),
        &metadata,
        Path::new(output_dir).join(name),
        CheckpointFormat::Bin,
    )?;
    tracing::info!(
        "Saved policy checkpoint at {}",
        checkpoint.checkpoint_path.display()
    );
    verify_checkpoint_artifact::<TrainBackend>(&checkpoint.checkpoint_path, device)?;
    tracing::info!(
        "Verified policy checkpoint at {}",
        checkpoint.checkpoint_path.display()
    );
    Ok(checkpoint)
}

#[allow(clippy::too_many_arguments)]
fn save_world_model_stage(
    model_type: CheckpointKind,
    model: &gpc_world::StateWorldModel<TrainBackend>,
    world_model_config: &WorldModelConfig,
    name: &str,
    output_dir: &str,
    epoch: Option<usize>,
    loss: Option<f32>,
    device: &<TrainBackend as burn::prelude::Backend>::Device,
) -> Result<CheckpointArtifact> {
    let metadata = checkpoint_metadata(
        model_type,
        epoch,
        loss,
        serde_json::to_string_pretty(world_model_config)?,
    );
    let checkpoint = save_world_model_checkpoint::<TrainBackend>(
        model.clone(),
        &metadata,
        Path::new(output_dir).join(name),
        CheckpointFormat::Bin,
    )?;
    tracing::info!(
        "Saved world-model checkpoint at {}",
        checkpoint.checkpoint_path.display()
    );
    verify_checkpoint_artifact::<TrainBackend>(&checkpoint.checkpoint_path, device)?;
    tracing::info!(
        "Verified world-model checkpoint at {}",
        checkpoint.checkpoint_path.display()
    );
    Ok(checkpoint)
}

fn build_training_report(ctx: TrainingReportContext<'_>) -> TrainingReport {
    let TrainingReportContext {
        args,
        config,
        training_config,
        dataset_config,
        dataset,
        dataset_split,
        policy,
        world_model,
        artifacts,
    } = ctx;

    let (train_episodes, train_transitions, validation_episodes, validation_transitions, split) =
        if let Some(split) = dataset_split {
            let train_episodes = split.train.num_episodes();
            let train_transitions = split.train.num_transitions();
            let validation_episodes = split.validation.num_episodes();
            let validation_transitions = split.validation.num_transitions();
            (
                train_episodes,
                train_transitions,
                validation_episodes,
                validation_transitions,
                Some(DatasetSplitReport {
                    validation_split: args.validation_split,
                    train_episodes,
                    train_transitions,
                    validation_episodes,
                    validation_transitions,
                }),
            )
        } else {
            (
                dataset.num_episodes(),
                dataset.num_transitions(),
                0,
                0,
                None,
            )
        };

    TrainingReport {
        schema_version: 1,
        generated_at: current_timestamp(),
        request: TrainingRequestReport {
            component_requested: args.component.clone(),
            component_resolved: normalize_component_label(&args.component).to_string(),
            synthetic_requested: args.synthetic,
            config_path: args.config.clone(),
            data_path: args.data.clone(),
            output_dir: args.output.clone(),
            report_output: args.report_output.clone().unwrap_or_else(|| {
                default_train_report_path(&args.output)
                    .display()
                    .to_string()
            }),
        },
        settings: EffectiveTrainingSettingsReport {
            training: training_config.clone(),
            policy: config.policy.clone(),
            world_model: config.world_model.clone(),
            dataset: dataset_config.clone(),
            horizon: args.horizon,
            validation_split: args.validation_split,
        },
        dataset: DatasetReport {
            source: DatasetSourceReport {
                synthetic: args.synthetic || args.data.is_none(),
                path: if args.synthetic || args.data.is_none() {
                    None
                } else {
                    args.data.clone()
                },
            },
            total_episodes: dataset.num_episodes(),
            total_transitions: dataset.num_transitions(),
            train_episodes,
            train_transitions,
            validation_episodes,
            validation_transitions,
            split,
        },
        policy,
        world_model,
        artifacts: artifacts.iter().map(checkpoint_artifact_report).collect(),
    }
}

fn checkpoint_artifact_report(artifact: &CheckpointArtifact) -> WrittenCheckpointReport {
    WrittenCheckpointReport {
        name: checkpoint_name(&artifact.checkpoint_path),
        checkpoint_path: artifact.checkpoint_path.display().to_string(),
        metadata_path: artifact.metadata_path.display().to_string(),
    }
}

fn checkpoint_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn normalize_component_label(component: &str) -> &str {
    match component {
        "wm" => "world-model",
        other => other,
    }
}

fn training_stage_report(
    final_epoch: Option<usize>,
    final_loss: Option<f32>,
    epoch_losses: &[f32],
    validation_losses: &[f32],
    best_epoch: Option<usize>,
    best_validation_loss: Option<f32>,
) -> TrainingStageReport {
    TrainingStageReport {
        final_epoch,
        final_loss,
        epoch_losses: epoch_losses.to_vec(),
        validation_losses: validation_losses.to_vec(),
        best_epoch,
        best_validation_loss,
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

fn validate_validation_split(validation_split: f32) -> Result<()> {
    if !(0.0..1.0).contains(&validation_split) {
        anyhow::bail!("validation_split must be in [0.0, 1.0)");
    }

    Ok(())
}

fn apply_training_overrides(training_config: &mut TrainingConfig, args: &TrainArgs) {
    if let Some(epochs) = args.epochs {
        training_config.num_epochs = epochs;
    }
    if let Some(batch_size) = args.batch_size {
        training_config.batch_size = batch_size;
    }
    if let Some(learning_rate) = args.learning_rate {
        training_config.learning_rate = learning_rate;
    }
    if let Some(weight_decay) = args.weight_decay {
        training_config.weight_decay = weight_decay;
    }
    if let Some(grad_clip_norm) = args.grad_clip_norm {
        training_config.grad_clip_norm = grad_clip_norm;
    }
    if let Some(warmup_steps) = args.warmup_steps {
        training_config.warmup_steps = warmup_steps;
    }
    if let Some(checkpoint_every) = args.checkpoint_every {
        training_config.checkpoint_every = checkpoint_every;
    }
    if let Some(log_every) = args.log_every {
        training_config.log_every = log_every;
    }
    if let Some(seed) = args.seed {
        training_config.seed = seed;
    }
}

fn cleanup_stale_best_artifacts(output_dir: &str, component: &str) -> Result<()> {
    match component {
        "policy" => remove_checkpoint_artifact(&Path::new(output_dir).join("policy_best.bin"))?,
        "world-model" | "wm" => {
            remove_checkpoint_artifact(&Path::new(output_dir).join("world_model_phase1_best.bin"))?;
            remove_checkpoint_artifact(&Path::new(output_dir).join("world_model_best.bin"))?;
        }
        "all" => {
            remove_checkpoint_artifact(&Path::new(output_dir).join("policy_best.bin"))?;
            remove_checkpoint_artifact(&Path::new(output_dir).join("world_model_phase1_best.bin"))?;
            remove_checkpoint_artifact(&Path::new(output_dir).join("world_model_best.bin"))?;
        }
        _ => {}
    }

    Ok(())
}

fn remove_checkpoint_artifact(path: &Path) -> Result<()> {
    let metadata_path = gpc_compat::metadata_path_for(path);

    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    match std::fs::remove_file(metadata_path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    Ok(())
}

fn current_timestamp() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => format!("{}Z", duration.as_secs()),
        Err(_) => "0Z".to_string(),
    }
}

fn log_artifacts(artifacts: &[CheckpointArtifact]) {
    if artifacts.is_empty() {
        tracing::warn!("No checkpoint artifacts were written");
        return;
    }

    tracing::info!("Artifacts written:");
    for artifact in artifacts {
        tracing::info!("  - {}", artifact.checkpoint_path.display());
        tracing::info!("    metadata: {}", artifact.metadata_path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::reporting::{default_train_report_path, write_json_report};
    use burn::prelude::Backend;
    use gpc_compat::verify_checkpoint_artifact;
    use gpc_core::config::{NoiseScheduleConfig, PolicyConfig, TrainingConfig, WorldModelConfig};
    use gpc_train::data::Episode;
    use std::path::Path;

    fn temp_train_dir(name: &str) -> std::path::PathBuf {
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

    #[test]
    fn validation_split_rejects_negative_values() {
        let err = validate_validation_split(-0.01).expect_err("negative split should be rejected");
        assert!(
            err.to_string()
                .contains("validation_split must be in [0.0, 1.0)")
        );
    }

    #[test]
    fn run_train_rejects_invalid_training_overrides() {
        let dir = temp_train_dir("invalid_training_overrides");
        let data_dir = dir.join("data");
        let output_dir = dir.join("runs");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = gpc_core::config::GpcConfig {
            policy: PolicyConfig {
                obs_dim: 4,
                action_dim: 2,
                obs_horizon: 2,
                pred_horizon: 2,
                action_horizon: 2,
                hidden_dim: 8,
                num_res_blocks: 1,
                noise_schedule: NoiseScheduleConfig {
                    num_timesteps: 8,
                    beta_start: 1e-4,
                    beta_end: 0.02,
                },
            },
            world_model: WorldModelConfig {
                state_dim: 4,
                action_dim: 2,
                hidden_dim: 8,
                num_layers: 1,
                dropout: 0.0,
            },
            training: TrainingConfig {
                num_epochs: 1,
                batch_size: 2,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
                warmup_steps: 0,
                checkpoint_every: 1,
                log_every: 1,
                seed: 7,
            },
            ..Default::default()
        };

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let episodes = vec![Episode {
            states: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![0.2, 0.1, 0.0, 0.0],
            ],
            actions: vec![vec![0.1, 0.0], vec![0.1, 0.1]],
            observations: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![0.2, 0.1, 0.0, 0.0],
            ],
        }];
        let episodes_path = data_dir.join("episodes.json");
        std::fs::write(
            &episodes_path,
            serde_json::to_string_pretty(&episodes).unwrap(),
        )
        .unwrap();

        let args = TrainArgs {
            config: Some(config_path.display().to_string()),
            data: Some(episodes_path.display().to_string()),
            component: "policy".to_string(),
            epochs: Some(1),
            batch_size: Some(1),
            learning_rate: Some(5e-4),
            weight_decay: Some(-0.1),
            grad_clip_norm: Some(0.5),
            warmup_steps: Some(2),
            checkpoint_every: Some(1),
            log_every: Some(1),
            seed: Some(17),
            synthetic: false,
            output: output_dir.display().to_string(),
            horizon: 2,
            validation_split: 0.0,
            report_output: None,
        };

        let err = run_train(args).expect_err("negative weight decay should be rejected");
        assert!(
            err.to_string()
                .contains("weight_decay must be finite and >= 0")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_train_accepts_direct_episodes_file() {
        let dir = temp_train_dir("direct_file_input");
        let data_dir = dir.join("data");
        let output_dir = dir.join("runs");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = gpc_core::config::GpcConfig {
            policy: PolicyConfig {
                obs_dim: 4,
                action_dim: 2,
                obs_horizon: 2,
                pred_horizon: 2,
                action_horizon: 2,
                hidden_dim: 8,
                num_res_blocks: 1,
                noise_schedule: NoiseScheduleConfig {
                    num_timesteps: 8,
                    beta_start: 1e-4,
                    beta_end: 0.02,
                },
            },
            world_model: WorldModelConfig {
                state_dim: 4,
                action_dim: 2,
                hidden_dim: 8,
                num_layers: 1,
                dropout: 0.0,
            },
            training: TrainingConfig {
                num_epochs: 1,
                batch_size: 2,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
                warmup_steps: 0,
                checkpoint_every: 1,
                log_every: 1,
                seed: 7,
            },
            ..Default::default()
        };

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let episodes = vec![Episode {
            states: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![0.2, 0.1, 0.0, 0.0],
            ],
            actions: vec![vec![0.1, 0.0], vec![0.1, 0.1]],
            observations: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![0.2, 0.1, 0.0, 0.0],
            ],
        }];
        let episodes_path = data_dir.join("episodes.json");
        std::fs::write(
            &episodes_path,
            serde_json::to_string_pretty(&episodes).unwrap(),
        )
        .unwrap();

        let report_output = default_train_report_path(&output_dir);
        let args = TrainArgs {
            config: Some(config_path.display().to_string()),
            data: Some(episodes_path.display().to_string()),
            component: "policy".to_string(),
            epochs: Some(1),
            batch_size: Some(1),
            learning_rate: Some(5e-4),
            weight_decay: Some(0.0),
            grad_clip_norm: Some(0.5),
            warmup_steps: Some(2),
            checkpoint_every: Some(1),
            log_every: Some(1),
            seed: Some(17),
            synthetic: false,
            output: output_dir.display().to_string(),
            horizon: 2,
            validation_split: 0.0,
            report_output: None,
        };

        run_train(args).unwrap();

        assert!(
            report_output.exists(),
            "expected train report to be written"
        );
        let report: TrainingReport =
            serde_json::from_str(&std::fs::read_to_string(&report_output).unwrap()).unwrap();
        assert_eq!(
            report.request.data_path.as_deref(),
            Some(episodes_path.to_str().unwrap())
        );
        assert_eq!(report.dataset.total_episodes, 1);
        assert_eq!(report.dataset.train_episodes, 1);
        assert_eq!(report.dataset.validation_episodes, 0);
        assert_eq!(report.settings.training.num_epochs, 1);
        assert_eq!(report.settings.training.batch_size, 1);
        assert_eq!(report.settings.training.learning_rate, 5e-4);
        assert_eq!(report.settings.training.weight_decay, 0.0);
        assert_eq!(report.settings.training.grad_clip_norm, 0.5);
        assert_eq!(report.settings.training.warmup_steps, 2);
        assert_eq!(report.settings.training.checkpoint_every, 1);
        assert_eq!(report.settings.training.log_every, 1);
        assert_eq!(report.settings.training.seed, 17);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_train_accepts_short_dataset_for_world_model_only() {
        let dir = temp_train_dir("world_model_short_dataset");
        let data_dir = dir.join("data");
        let output_dir = dir.join("runs");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = gpc_core::config::GpcConfig {
            policy: PolicyConfig {
                obs_dim: 4,
                action_dim: 2,
                obs_horizon: 2,
                pred_horizon: 4,
                action_horizon: 2,
                hidden_dim: 8,
                num_res_blocks: 1,
                noise_schedule: NoiseScheduleConfig {
                    num_timesteps: 8,
                    beta_start: 1e-4,
                    beta_end: 0.02,
                },
            },
            world_model: WorldModelConfig {
                state_dim: 4,
                action_dim: 2,
                hidden_dim: 8,
                num_layers: 1,
                dropout: 0.0,
            },
            training: TrainingConfig {
                num_epochs: 1,
                batch_size: 1,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
                warmup_steps: 0,
                checkpoint_every: 1,
                log_every: 1,
                seed: 7,
            },
            ..Default::default()
        };

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let episodes = vec![Episode {
            states: vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.1, 0.0, 0.0, 0.0]],
            actions: vec![vec![0.1, 0.0]],
            observations: vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.1, 0.0, 0.0, 0.0]],
        }];
        let episodes_path = data_dir.join("episodes.json");
        std::fs::write(
            &episodes_path,
            serde_json::to_string_pretty(&episodes).unwrap(),
        )
        .unwrap();

        let report_output = default_train_report_path(&output_dir);
        let args = TrainArgs {
            config: Some(config_path.display().to_string()),
            data: Some(episodes_path.display().to_string()),
            component: "world-model".to_string(),
            epochs: Some(1),
            synthetic: false,
            output: output_dir.display().to_string(),
            horizon: 4,
            validation_split: 0.0,
            report_output: None,
            ..TrainArgs::default()
        };

        run_train(args).unwrap();

        assert!(
            report_output.exists(),
            "expected train report to be written"
        );
        let report: TrainingReport =
            serde_json::from_str(&std::fs::read_to_string(&report_output).unwrap()).unwrap();
        assert_eq!(report.request.component_resolved, "world-model");
        assert_eq!(report.dataset.total_episodes, 1);
        assert_eq!(report.dataset.total_transitions, 1);
        assert_eq!(report.dataset.train_episodes, 1);
        assert_eq!(report.dataset.validation_episodes, 0);
        assert!(report.world_model.is_some());
        assert!(report.policy.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_train_rejects_short_dataset_for_policy_training() {
        let dir = temp_train_dir("policy_short_dataset");
        let data_dir = dir.join("data");
        let output_dir = dir.join("runs");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = gpc_core::config::GpcConfig {
            policy: PolicyConfig {
                obs_dim: 4,
                action_dim: 2,
                obs_horizon: 2,
                pred_horizon: 4,
                action_horizon: 2,
                hidden_dim: 8,
                num_res_blocks: 1,
                noise_schedule: NoiseScheduleConfig {
                    num_timesteps: 8,
                    beta_start: 1e-4,
                    beta_end: 0.02,
                },
            },
            world_model: WorldModelConfig {
                state_dim: 4,
                action_dim: 2,
                hidden_dim: 8,
                num_layers: 1,
                dropout: 0.0,
            },
            training: TrainingConfig {
                num_epochs: 1,
                batch_size: 1,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
                warmup_steps: 0,
                checkpoint_every: 1,
                log_every: 1,
                seed: 7,
            },
            ..Default::default()
        };

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let episodes = vec![Episode {
            states: vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.1, 0.0, 0.0, 0.0]],
            actions: vec![vec![0.1, 0.0]],
            observations: vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.1, 0.0, 0.0, 0.0]],
        }];
        let episodes_path = data_dir.join("episodes.json");
        std::fs::write(
            &episodes_path,
            serde_json::to_string_pretty(&episodes).unwrap(),
        )
        .unwrap();

        let args = TrainArgs {
            config: Some(config_path.display().to_string()),
            data: Some(episodes_path.display().to_string()),
            component: "policy".to_string(),
            epochs: Some(1),
            synthetic: false,
            output: output_dir.display().to_string(),
            horizon: 4,
            validation_split: 0.0,
            report_output: None,
            ..TrainArgs::default()
        };

        let err =
            run_train(args).expect_err("short dataset should be rejected for policy training");
        assert!(
            err.to_string()
                .contains("usable open-loop windows for policy training")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_train_rejects_malformed_dataset_via_shared_loader() {
        let dir = temp_train_dir("malformed_dataset");
        let data_dir = dir.join("data");
        let output_dir = dir.join("runs");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = gpc_core::config::GpcConfig {
            policy: PolicyConfig {
                obs_dim: 4,
                action_dim: 2,
                obs_horizon: 2,
                pred_horizon: 2,
                action_horizon: 2,
                hidden_dim: 8,
                num_res_blocks: 1,
                noise_schedule: NoiseScheduleConfig {
                    num_timesteps: 8,
                    beta_start: 1e-4,
                    beta_end: 0.02,
                },
            },
            world_model: WorldModelConfig {
                state_dim: 4,
                action_dim: 2,
                hidden_dim: 8,
                num_layers: 1,
                dropout: 0.0,
            },
            training: TrainingConfig {
                num_epochs: 1,
                batch_size: 2,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
                warmup_steps: 0,
                checkpoint_every: 1,
                log_every: 1,
                seed: 7,
            },
            ..Default::default()
        };

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let malformed = vec![Episode {
            states: vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.1, 0.0, 0.0, 0.0]],
            actions: vec![vec![0.1, 0.0]],
            observations: vec![vec![0.0, 0.0, 0.0]],
        }];
        let episodes_path = data_dir.join("episodes.json");
        std::fs::write(
            &episodes_path,
            serde_json::to_string_pretty(&malformed).unwrap(),
        )
        .unwrap();

        let args = TrainArgs {
            config: Some(config_path.display().to_string()),
            data: Some(episodes_path.display().to_string()),
            component: "policy".to_string(),
            epochs: Some(1),
            synthetic: false,
            output: output_dir.display().to_string(),
            horizon: 2,
            validation_split: 0.0,
            report_output: None,
            ..TrainArgs::default()
        };

        let err = run_train(args).expect_err("malformed dataset should be rejected");
        assert!(
            err.to_string()
                .contains("must have the same number of states and observations")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_best_artifacts_are_removed_for_non_validation_runs() {
        let dir = temp_train_dir("cleanup");
        let best_paths = [
            dir.join("policy_best.bin"),
            dir.join("policy_best.meta.json"),
            dir.join("world_model_phase1_best.bin"),
            dir.join("world_model_phase1_best.meta.json"),
            dir.join("world_model_best.bin"),
            dir.join("world_model_best.meta.json"),
        ];
        let final_paths = [
            dir.join("policy_final.bin"),
            dir.join("policy_final.meta.json"),
        ];

        for path in best_paths.iter().chain(final_paths.iter()) {
            std::fs::write(path, b"stale").unwrap();
        }

        cleanup_stale_best_artifacts(dir.to_str().unwrap(), "all").unwrap();

        for path in &best_paths {
            assert!(
                !path.exists(),
                "expected stale artifact to be removed: {path:?}"
            );
        }
        for path in &final_paths {
            assert!(path.exists(), "expected final artifact to remain: {path:?}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_report_path_lives_under_output_dir() {
        let path = default_train_report_path("/tmp/gpc-output");
        assert_eq!(path, PathBuf::from("/tmp/gpc-output/train_report.json"));
    }

    #[test]
    fn report_writing_round_trips_through_json() {
        let dir = temp_train_dir("report_roundtrip");
        let report_path = dir.join("train_report.json");
        let report = TrainingReport {
            schema_version: 1,
            generated_at: "123Z".to_string(),
            request: TrainingRequestReport {
                component_requested: "all".to_string(),
                component_resolved: "all".to_string(),
                synthetic_requested: false,
                config_path: Some("config.json".to_string()),
                data_path: Some("data".to_string()),
                output_dir: "runs/out".to_string(),
                report_output: report_path.display().to_string(),
            },
            settings: EffectiveTrainingSettingsReport {
                training: TrainingConfig {
                    num_epochs: 1,
                    batch_size: 2,
                    learning_rate: 1e-3,
                    weight_decay: 0.0,
                    grad_clip_norm: 1.0,
                    warmup_steps: 0,
                    checkpoint_every: 1,
                    log_every: 1,
                    seed: 7,
                },
                policy: PolicyConfig {
                    obs_dim: 4,
                    action_dim: 2,
                    obs_horizon: 2,
                    pred_horizon: 2,
                    action_horizon: 2,
                    hidden_dim: 8,
                    num_res_blocks: 1,
                    noise_schedule: NoiseScheduleConfig::default(),
                },
                world_model: WorldModelConfig {
                    state_dim: 4,
                    action_dim: 2,
                    hidden_dim: 8,
                    num_layers: 1,
                    dropout: 0.0,
                },
                dataset: GpcDatasetConfig {
                    data_dir: "data".to_string(),
                    state_dim: 4,
                    action_dim: 2,
                    obs_dim: 4,
                    obs_horizon: 2,
                    pred_horizon: 2,
                },
                horizon: 2,
                validation_split: 0.5,
            },
            dataset: DatasetReport {
                source: DatasetSourceReport {
                    synthetic: false,
                    path: Some("data/episodes.json".to_string()),
                },
                total_episodes: 2,
                total_transitions: 8,
                train_episodes: 1,
                train_transitions: 4,
                validation_episodes: 1,
                validation_transitions: 4,
                split: Some(DatasetSplitReport {
                    validation_split: 0.5,
                    train_episodes: 1,
                    train_transitions: 4,
                    validation_episodes: 1,
                    validation_transitions: 4,
                }),
            },
            policy: Some(PolicyTrainingReport {
                training: TrainingStageReport {
                    final_epoch: Some(1),
                    final_loss: Some(0.123),
                    epoch_losses: vec![0.123],
                    validation_losses: vec![0.111],
                    best_epoch: Some(1),
                    best_validation_loss: Some(0.111),
                },
            }),
            world_model: None,
            artifacts: vec![WrittenCheckpointReport {
                name: "policy_final".to_string(),
                checkpoint_path: "runs/out/policy_final.bin".to_string(),
                metadata_path: "runs/out/policy_final.meta.json".to_string(),
            }],
        };

        write_json_report(&report_path, &report).unwrap();
        let round_trip: TrainingReport =
            serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
        assert_eq!(round_trip.request.component_requested, "all");
        assert_eq!(round_trip.dataset.train_episodes, 1);
        assert_eq!(round_trip.artifacts[0].name, "policy_final");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_train_writes_default_report_for_tiny_dataset() {
        let dir = temp_train_dir("run_train_report");
        let data_dir = dir.join("data");
        let output_dir = dir.join("runs");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = gpc_core::config::GpcConfig {
            policy: PolicyConfig {
                obs_dim: 4,
                action_dim: 2,
                obs_horizon: 2,
                pred_horizon: 2,
                action_horizon: 2,
                hidden_dim: 8,
                num_res_blocks: 1,
                noise_schedule: NoiseScheduleConfig {
                    num_timesteps: 8,
                    beta_start: 1e-4,
                    beta_end: 0.02,
                },
            },
            world_model: WorldModelConfig {
                state_dim: 4,
                action_dim: 2,
                hidden_dim: 8,
                num_layers: 1,
                dropout: 0.0,
            },
            training: TrainingConfig {
                num_epochs: 1,
                batch_size: 2,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
                warmup_steps: 0,
                checkpoint_every: 1,
                log_every: 1,
                seed: 7,
            },
            ..Default::default()
        };

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let episodes = vec![
            Episode {
                states: vec![
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.1, 0.0, 0.0, 0.0],
                    vec![0.2, 0.1, 0.0, 0.0],
                    vec![0.3, 0.1, 0.0, 0.0],
                    vec![0.4, 0.2, 0.0, 0.0],
                ],
                actions: vec![
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                ],
                observations: vec![
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.1, 0.0, 0.0, 0.0],
                    vec![0.2, 0.1, 0.0, 0.0],
                    vec![0.3, 0.1, 0.0, 0.0],
                    vec![0.4, 0.2, 0.0, 0.0],
                ],
            },
            Episode {
                states: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![1.1, 0.0, 0.0, 0.0],
                    vec![1.2, 0.1, 0.0, 0.0],
                    vec![1.3, 0.1, 0.0, 0.0],
                    vec![1.4, 0.2, 0.0, 0.0],
                ],
                actions: vec![
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                ],
                observations: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![1.1, 0.0, 0.0, 0.0],
                    vec![1.2, 0.1, 0.0, 0.0],
                    vec![1.3, 0.1, 0.0, 0.0],
                    vec![1.4, 0.2, 0.0, 0.0],
                ],
            },
        ];
        let episodes_path = data_dir.join("episodes.json");
        std::fs::write(
            &episodes_path,
            serde_json::to_string_pretty(&episodes).unwrap(),
        )
        .unwrap();

        let report_output = default_train_report_path(&output_dir);
        let args = TrainArgs {
            config: Some(config_path.display().to_string()),
            data: Some(data_dir.display().to_string()),
            component: "all".to_string(),
            epochs: Some(1),
            synthetic: false,
            output: output_dir.display().to_string(),
            horizon: 2,
            validation_split: 0.5,
            report_output: None,
            ..TrainArgs::default()
        };

        run_train(args).unwrap();

        assert!(
            report_output.exists(),
            "expected train report to be written"
        );
        let report: TrainingReport =
            serde_json::from_str(&std::fs::read_to_string(&report_output).unwrap()).unwrap();
        assert_eq!(report.request.component_resolved, "all");
        assert_eq!(report.dataset.total_episodes, 2);
        assert_eq!(report.dataset.train_episodes, 1);
        assert_eq!(report.dataset.validation_episodes, 1);
        assert!(report.policy.is_some());
        assert!(report.world_model.is_some());
        assert!(!report.artifacts.is_empty());
        assert!(
            report
                .artifacts
                .iter()
                .any(|artifact| artifact.name == "policy_final")
        );
        let device = <TrainBackend as Backend>::Device::default();
        for artifact in &report.artifacts {
            verify_checkpoint_artifact::<TrainBackend>(
                Path::new(&artifact.checkpoint_path),
                &device,
            )
            .unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_train_honors_custom_report_output_path() {
        let dir = temp_train_dir("run_train_custom_report");
        let data_dir = dir.join("data");
        let output_dir = dir.join("runs");
        let custom_report_output = dir.join("artifacts").join("custom-train-report.json");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = gpc_core::config::GpcConfig {
            policy: PolicyConfig {
                obs_dim: 4,
                action_dim: 2,
                obs_horizon: 2,
                pred_horizon: 2,
                action_horizon: 2,
                hidden_dim: 8,
                num_res_blocks: 1,
                noise_schedule: NoiseScheduleConfig {
                    num_timesteps: 8,
                    beta_start: 1e-4,
                    beta_end: 0.02,
                },
            },
            world_model: WorldModelConfig {
                state_dim: 4,
                action_dim: 2,
                hidden_dim: 8,
                num_layers: 1,
                dropout: 0.0,
            },
            training: TrainingConfig {
                num_epochs: 1,
                batch_size: 2,
                learning_rate: 1e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
                warmup_steps: 0,
                checkpoint_every: 1,
                log_every: 1,
                seed: 7,
            },
            ..Default::default()
        };

        let config_path = dir.join("config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let episodes = vec![
            Episode {
                states: vec![
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.1, 0.0, 0.0, 0.0],
                    vec![0.2, 0.1, 0.0, 0.0],
                    vec![0.3, 0.1, 0.0, 0.0],
                    vec![0.4, 0.2, 0.0, 0.0],
                ],
                actions: vec![
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                ],
                observations: vec![
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.1, 0.0, 0.0, 0.0],
                    vec![0.2, 0.1, 0.0, 0.0],
                    vec![0.3, 0.1, 0.0, 0.0],
                    vec![0.4, 0.2, 0.0, 0.0],
                ],
            },
            Episode {
                states: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![1.1, 0.0, 0.0, 0.0],
                    vec![1.2, 0.1, 0.0, 0.0],
                    vec![1.3, 0.1, 0.0, 0.0],
                    vec![1.4, 0.2, 0.0, 0.0],
                ],
                actions: vec![
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                    vec![0.1, 0.0],
                    vec![0.1, 0.1],
                ],
                observations: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![1.1, 0.0, 0.0, 0.0],
                    vec![1.2, 0.1, 0.0, 0.0],
                    vec![1.3, 0.1, 0.0, 0.0],
                    vec![1.4, 0.2, 0.0, 0.0],
                ],
            },
        ];
        let episodes_path = data_dir.join("episodes.json");
        std::fs::write(
            &episodes_path,
            serde_json::to_string_pretty(&episodes).unwrap(),
        )
        .unwrap();

        let default_report_output = default_train_report_path(&output_dir);
        let args = TrainArgs {
            config: Some(config_path.display().to_string()),
            data: Some(data_dir.display().to_string()),
            component: "all".to_string(),
            epochs: Some(1),
            synthetic: false,
            output: output_dir.display().to_string(),
            horizon: 2,
            validation_split: 0.5,
            report_output: Some(custom_report_output.display().to_string()),
            ..TrainArgs::default()
        };

        run_train(args).unwrap();

        assert!(
            custom_report_output.exists(),
            "expected custom train report to be written"
        );
        assert!(
            !default_report_output.exists(),
            "default report path should not be used when an explicit path is provided"
        );
        let report: TrainingReport =
            serde_json::from_str(&std::fs::read_to_string(&custom_report_output).unwrap()).unwrap();
        assert_eq!(
            report.request.report_output,
            custom_report_output.display().to_string()
        );
        assert_eq!(report.dataset.total_episodes, 2);
        assert!(report.policy.is_some());
        assert!(report.world_model.is_some());
        let device = <TrainBackend as Backend>::Device::default();
        for artifact in &report.artifacts {
            verify_checkpoint_artifact::<TrainBackend>(
                Path::new(&artifact.checkpoint_path),
                &device,
            )
            .unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
