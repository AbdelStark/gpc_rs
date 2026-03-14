use anyhow::Result;
use clap::Parser;

/// Generative Robot Policies — Rust implementation.
#[derive(Parser, Debug)]
#[command(name = "gpc", version, about)]
enum Cli {
    /// Train the diffusion policy or world model.
    Train,
    /// Evaluate trajectories using GPC-RANK or GPC-OPT.
    Eval,
    /// Inspect or convert model checkpoints.
    Checkpoint,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli {
        Cli::Train => {
            tracing::info!("Training not yet implemented");
        }
        Cli::Eval => {
            tracing::info!("Evaluation not yet implemented");
        }
        Cli::Checkpoint => {
            tracing::info!("Checkpoint inspection not yet implemented");
        }
    }

    Ok(())
}
