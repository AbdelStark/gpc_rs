use anyhow::Result;
use clap::Parser;

mod commands;

/// Generative Robot Policies — Rust implementation.
///
/// GPC: inference-time method for improving pretrained behavior-cloning
/// policies via predictive world modeling.
#[derive(Parser, Debug)]
#[command(name = "gpc", version, about, long_about = None)]
enum Cli {
    /// Train the diffusion policy or world model.
    Train(commands::TrainArgs),
    /// Evaluate trajectories using GPC-RANK or GPC-OPT.
    Eval(commands::EvalArgs),
    /// Benchmark checkpoints across policy, GPC-RANK, and GPC-OPT.
    Benchmark(commands::BenchmarkArgs),
    /// Inspect or convert model checkpoints.
    Checkpoint(commands::CheckpointArgs),
    /// Generate a default configuration file.
    InitConfig(commands::InitConfigArgs),
    /// Run a quick demo with synthetic data.
    Demo(commands::DemoArgs),
}

fn main() -> Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let cli = Cli::parse();

    match cli {
        Cli::Train(args) => commands::run_train(args),
        Cli::Eval(args) => commands::run_eval(args),
        Cli::Benchmark(args) => commands::run_benchmark(args),
        Cli::Checkpoint(args) => commands::run_checkpoint(args),
        Cli::InitConfig(args) => commands::run_init_config(args),
        Cli::Demo(args) => commands::run_demo(args),
    }
}
