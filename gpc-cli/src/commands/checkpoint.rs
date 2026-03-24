//! Checkpoint command implementation.

use anyhow::Result;
use clap::Args;
use std::path::{Path, PathBuf};

use gpc_compat::{
    CheckpointFormat, CheckpointKind, checkpoint_path, convert_policy_checkpoint,
    convert_world_model_checkpoint, load_metadata, metadata_path_for,
};

/// Arguments for the checkpoint command.
#[derive(Args, Debug)]
pub struct CheckpointArgs {
    /// Subcommand: "inspect" or "convert".
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
        }
        other => {
            anyhow::bail!("Unknown action: {other}. Use 'inspect' or 'convert'.");
        }
    }

    Ok(())
}

fn inspect_onnx(path: &str) -> Result<()> {
    let inspector = gpc_compat::OnnxInspector::load(Path::new(path))?;
    println!("{}", inspector.summary());
    Ok(())
}

fn inspect_checkpoint(path: &Path) -> Result<()> {
    let checkpoint_path = checkpoint_path(path);
    let metadata_path = metadata_path_for(&checkpoint_path);

    tracing::info!("Checkpoint file: {}", checkpoint_path.display());
    tracing::info!("Metadata file: {}", metadata_path.display());

    if metadata_path.exists() {
        let metadata = load_metadata(&checkpoint_path)?;
        println!("Checkpoint Metadata:");
        println!("  Model type: {}", metadata.model_type);
        println!("  Epoch: {}", metadata.epoch);
        println!("  Loss: {:.6}", metadata.loss);
        println!("  Timestamp: {}", metadata.timestamp);
        println!("  Format: {}", checkpoint_format_label(&checkpoint_path));
    } else {
        tracing::info!(
            "No metadata found. File size: {} bytes",
            std::fs::metadata(&checkpoint_path)?.len()
        );
    }

    Ok(())
}

fn checkpoint_format_label(path: &Path) -> &'static str {
    match CheckpointFormat::from_path(path) {
        Some(CheckpointFormat::Bin) => "bin",
        Some(CheckpointFormat::Mpk) => "mpk",
        None => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            metadata_path_for(meta),
            PathBuf::from("/tmp/model.meta.json")
        );
    }
}
