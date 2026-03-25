//! Checkpoint command implementation.

use anyhow::Result;
use clap::Args;
use std::path::{Path, PathBuf};

use gpc_compat::{
    CheckpointConfigSummary, CheckpointFormat, CheckpointInspectionReport, CheckpointKind,
    checkpoint_path, convert_policy_checkpoint, convert_world_model_checkpoint,
    inspect_checkpoint_artifact, load_metadata, verify_checkpoint_artifact,
};

/// Arguments for the checkpoint command.
#[derive(Args, Debug)]
pub struct CheckpointArgs {
    /// Subcommand: "inspect", "verify", or "convert".
    #[arg(long, default_value = "inspect")]
    action: String,

    /// Path to checkpoint or ONNX file.
    #[arg(short, long)]
    path: Option<String>,

    /// Optional output stem for checkpoint conversion.
    #[arg(short, long)]
    output: Option<String>,
}

/// Run the checkpoint command.
pub fn run_checkpoint(args: CheckpointArgs) -> Result<()> {
    match args.action.as_str() {
        "inspect" => {
            let path = args
                .path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--path required for inspect"))?;

            if path.ends_with(".onnx") {
                inspect_onnx(path)?;
            } else {
                inspect_checkpoint(Path::new(path))?;
            }
        }
        "verify" => {
            let path = args
                .path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--path required for verify"))?;

            if path.ends_with(".onnx") {
                verify_onnx(path)?;
            } else {
                verify_checkpoint(Path::new(path))?;
            }
        }
        "convert" => {
            let path = args
                .path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--path required for convert"))?;

            let input_path = checkpoint_path(Path::new(path));
            let metadata = load_metadata(&input_path)?;
            let kind = CheckpointKind::from_metadata(&metadata)?;
            let input_format = CheckpointFormat::from_path(&input_path).ok_or_else(|| {
                anyhow::anyhow!("unsupported checkpoint extension: {}", input_path.display())
            })?;

            let output_format = args
                .output
                .as_deref()
                .and_then(|output| CheckpointFormat::from_path(Path::new(output)))
                .unwrap_or_else(|| input_format.opposite());

            let output_path = args
                .output
                .as_deref()
                .map(PathBuf::from)
                .unwrap_or_else(|| input_path.with_extension(output_format.extension()));

            let device =
                <burn::backend::ndarray::NdArray as burn::prelude::Backend>::Device::default();
            let artifact = match kind {
                CheckpointKind::Policy => convert_policy_checkpoint::<
                    burn::backend::ndarray::NdArray,
                >(
                    input_path, output_path, output_format, &device
                )?,
                CheckpointKind::WorldModel => convert_world_model_checkpoint::<
                    burn::backend::ndarray::NdArray,
                >(
                    input_path, output_path, output_format, &device
                )?,
            };

            tracing::info!(
                "Converted {} checkpoint to {}",
                metadata.model_type,
                artifact.checkpoint_path.display()
            );
            tracing::info!("Metadata sidecar: {}", artifact.metadata_path.display());
            let verification = verify_checkpoint_artifact::<burn::backend::ndarray::NdArray>(
                &artifact.checkpoint_path,
                &device,
            )?;
            tracing::info!(
                "Verified converted checkpoint: {} ({})",
                verification.checkpoint_path.display(),
                verification
                    .kind
                    .map(CheckpointKind::as_str)
                    .unwrap_or("unknown")
            );
        }
        other => {
            anyhow::bail!("Unknown action: {other}. Use 'inspect', 'verify', or 'convert'.");
        }
    }

    Ok(())
}

fn inspect_onnx(path: &str) -> Result<()> {
    let inspector = gpc_compat::OnnxInspector::load(Path::new(path))?;
    println!("{}", inspector.summary());
    Ok(())
}

fn verify_onnx(path: &str) -> Result<()> {
    let inspector = gpc_compat::OnnxInspector::load(Path::new(path))?;
    println!("{}", inspector.summary());
    println!("  Verified: yes");
    Ok(())
}

fn inspect_checkpoint(path: &Path) -> Result<()> {
    let report = inspect_checkpoint_artifact(path);
    print_checkpoint_report(&report, false);
    Ok(())
}

fn verify_checkpoint(path: &Path) -> Result<()> {
    let device = <burn::backend::ndarray::NdArray as burn::prelude::Backend>::Device::default();
    let report = verify_checkpoint_artifact::<burn::backend::ndarray::NdArray>(path, &device)?;
    print_checkpoint_report(&report, true);
    Ok(())
}

fn print_checkpoint_report(report: &CheckpointInspectionReport, verified: bool) {
    tracing::info!("Checkpoint file: {}", report.checkpoint_path.display());
    tracing::info!("Metadata file: {}", report.metadata_path.display());

    println!("Checkpoint Inspection:");
    println!("  Checkpoint: {}", report.checkpoint_path.display());
    println!("  Metadata: {}", report.metadata_path.display());
    println!(
        "  Format: {}",
        report
            .format
            .map(CheckpointFormat::as_str)
            .unwrap_or("unknown")
    );
    println!(
        "  Checkpoint size: {}",
        report
            .checkpoint_size_bytes
            .map(|size| format!("{size} bytes"))
            .unwrap_or_else(|| "missing".to_string())
    );
    println!(
        "  Metadata size: {}",
        report
            .metadata_size_bytes
            .map(|size| format!("{size} bytes"))
            .unwrap_or_else(|| "missing".to_string())
    );

    if let Some(metadata) = &report.metadata {
        println!("  Model type: {}", metadata.model_type);
        println!("  Epoch: {}", metadata.epoch);
        println!("  Loss: {:.6}", metadata.loss);
        println!("  Timestamp: {}", metadata.timestamp);
    }

    if let Some(kind) = report.kind {
        println!("  Kind: {}", kind.as_str());
    }

    if let Some(config) = &report.config {
        println!("  Config: {}", checkpoint_config_label(config));
        println!("  Config summary: {}", config.describe());
    }

    if verified {
        println!("  Verified: yes");
    }

    if !report.issues.is_empty() {
        println!("  Issues:");
        for issue in &report.issues {
            println!("    - {issue}");
        }
    }
}

fn checkpoint_config_label(config: &CheckpointConfigSummary) -> &'static str {
    match config {
        CheckpointConfigSummary::Policy(_) => "policy",
        CheckpointConfigSummary::WorldModel(_) => "world_model",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use gpc_compat::{CheckpointKind, CheckpointMetadata, save_policy_checkpoint};
    use gpc_core::config::{NoiseScheduleConfig, PolicyConfig};

    #[test]
    fn test_output_path_defaults_to_swapped_format() {
        let bin = Path::new("/tmp/model.bin");
        let mpk_output = bin.with_extension(CheckpointFormat::Mpk.extension());
        assert_eq!(mpk_output, PathBuf::from("/tmp/model.mpk"));

        let mpk = Path::new("/tmp/model.mpk");
        let bin_output = mpk.with_extension(CheckpointFormat::Bin.extension());
        assert_eq!(bin_output, PathBuf::from("/tmp/model.bin"));
    }

    #[test]
    fn test_metadata_paths_normalize_checkpoint_inputs() {
        let meta = Path::new("/tmp/model.meta.json");
        assert_eq!(checkpoint_path(meta), PathBuf::from("/tmp/model"));
        assert_eq!(
            gpc_compat::metadata_path_for(meta),
            PathBuf::from("/tmp/model.meta.json")
        );
    }

    #[test]
    fn verify_action_loads_valid_checkpoint() {
        let dir = temp_checkpoint_dir("verify_action");
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let config = tiny_policy_config();
        let model =
            gpc_policy::DiffusionPolicyConfig::from_policy_config(&config).init::<NdArray>(&device);
        let metadata = CheckpointMetadata {
            model_type: CheckpointKind::Policy.as_str().to_string(),
            epoch: 1,
            loss: 0.5,
            timestamp: "2026-03-25T00:00:00Z".to_string(),
            config_json: serde_json::to_string(&config).unwrap(),
        };
        let artifact = save_policy_checkpoint(
            model,
            &metadata,
            dir.join("policy_final"),
            CheckpointFormat::Bin,
        )
        .unwrap();

        verify_checkpoint(&artifact.checkpoint_path).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_action_rejects_corrupt_metadata() {
        let dir = temp_checkpoint_dir("verify_corrupt");
        let checkpoint_path = dir.join("policy_final.bin");
        std::fs::write(&checkpoint_path, b"weights").unwrap();
        std::fs::write(dir.join("policy_final.meta.json"), "{not valid json").unwrap();

        let err =
            verify_checkpoint(&checkpoint_path).expect_err("expected corrupt metadata to fail");
        assert!(err.to_string().contains("checkpoint verification failed"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tiny_policy_config() -> PolicyConfig {
        PolicyConfig {
            obs_dim: 4,
            action_dim: 2,
            obs_horizon: 2,
            pred_horizon: 3,
            action_horizon: 1,
            hidden_dim: 8,
            num_res_blocks: 1,
            noise_schedule: NoiseScheduleConfig {
                num_timesteps: 4,
                beta_start: 1e-4,
                beta_end: 0.02,
            },
        }
    }

    fn temp_checkpoint_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gpc_checkpoint_cli_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs() ^ u64::from(duration.subsec_nanos()))
                .unwrap_or(7)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
