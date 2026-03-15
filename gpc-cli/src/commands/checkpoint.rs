//! Checkpoint command implementation.

use anyhow::Result;
use clap::Args;
use std::path::Path;

/// Arguments for the checkpoint command.
#[derive(Args, Debug)]
pub struct CheckpointArgs {
    /// Subcommand: "inspect" or "convert".
    #[arg(long, default_value = "inspect")]
    action: String,

    /// Path to checkpoint or ONNX file.
    #[arg(short, long)]
    path: Option<String>,
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
            } else if path.ends_with(".meta.json") {
                inspect_metadata(path)?;
            } else {
                tracing::info!("Checkpoint file: {path}");
                // Try to load metadata
                let meta_path = format!("{}.meta.json", path.strip_suffix(".bin").unwrap_or(path));
                if Path::new(&meta_path).exists() {
                    inspect_metadata(&meta_path)?;
                } else {
                    tracing::info!(
                        "No metadata found. File size: {} bytes",
                        std::fs::metadata(path)?.len()
                    );
                }
            }
        }
        "convert" => {
            tracing::info!("Checkpoint conversion not yet implemented");
            tracing::info!("Supported future conversions: Burn <-> ONNX, PyTorch -> Burn");
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

fn inspect_metadata(path: &str) -> Result<()> {
    let metadata = gpc_compat::checkpoint::load_metadata(Path::new(path))?;
    println!("Checkpoint Metadata:");
    println!("  Model type: {}", metadata.model_type);
    println!("  Epoch: {}", metadata.epoch);
    println!("  Loss: {:.6}", metadata.loss);
    println!("  Timestamp: {}", metadata.timestamp);
    Ok(())
}
