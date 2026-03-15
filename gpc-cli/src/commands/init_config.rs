//! Init-config command implementation.

use anyhow::Result;
use clap::Args;

use gpc_core::config::GpcConfig;

/// Arguments for the init-config command.
#[derive(Args, Debug)]
pub struct InitConfigArgs {
    /// Output path for the configuration file.
    #[arg(short, long, default_value = "gpc_config.json")]
    output: String,
}

/// Generate a default configuration file.
pub fn run_init_config(args: InitConfigArgs) -> Result<()> {
    let config = GpcConfig::default();
    let json = serde_json::to_string_pretty(&config)?;
    std::fs::write(&args.output, &json)?;
    tracing::info!("Default configuration written to {}", args.output);
    println!("{json}");
    Ok(())
}
