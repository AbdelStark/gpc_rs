//! Eval command implementation.

use anyhow::Result;
use burn::backend::NdArray;
use burn::prelude::*;
use clap::Args;

use gpc_core::config::GpcConfig;
use gpc_core::traits::Evaluator;
use gpc_eval::{GpcOptBuilder, GpcRankBuilder};
use gpc_policy::DiffusionPolicyConfig;
use gpc_world::reward::L2RewardFunctionConfig;
use gpc_world::world_model::StateWorldModelConfig;

type EvalBackend = NdArray;

/// Arguments for the eval command.
#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Evaluation strategy: "rank" or "opt".
    #[arg(long, default_value = "rank")]
    strategy: String,

    /// Path to configuration file (JSON).
    #[arg(short, long)]
    config: Option<String>,

    /// Number of candidate trajectories (GPC-RANK).
    #[arg(long)]
    num_candidates: Option<usize>,

    /// Number of optimization steps (GPC-OPT).
    #[arg(long)]
    opt_steps: Option<usize>,

    /// Run a demo evaluation with random models.
    #[arg(long)]
    demo: bool,
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

    tracing::info!(
        "Evaluation with strategy: {} (use --demo for a quick test)",
        args.strategy
    );
    Ok(())
}

fn run_eval_demo(config: &GpcConfig, args: &EvalArgs) -> Result<()> {
    let device = <EvalBackend as Backend>::Device::default();

    tracing::info!("Running evaluation demo with random models");

    // Create models with random weights
    let policy_config = DiffusionPolicyConfig {
        obs_dim: config.policy.obs_dim,
        action_dim: config.policy.action_dim,
        obs_horizon: config.policy.obs_horizon,
        pred_horizon: config.policy.pred_horizon,
        hidden_dim: 32,
        time_embed_dim: 16,
        num_res_blocks: 1,
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

    // Create dummy observation and state
    let obs = Tensor::<EvalBackend, 3>::zeros(
        [1, config.policy.obs_horizon, config.policy.obs_dim],
        &device,
    );
    let state = Tensor::<EvalBackend, 2>::zeros([1, config.world_model.state_dim], &device);

    match args.strategy.as_str() {
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

            let evaluator = GpcOptBuilder::new(policy, world_model, reward_fn)
                .num_opt_steps(steps)
                .build();

            let best = evaluator.select_action(&obs, &state, &device)?;
            tracing::info!("Optimized action shape: {:?}", best.dims());
        }
        other => {
            anyhow::bail!("Unknown strategy: {other}. Use 'rank' or 'opt'.");
        }
    }

    tracing::info!("Evaluation demo complete");
    Ok(())
}
