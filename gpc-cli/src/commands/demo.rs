//! Demo command: end-to-end pipeline with synthetic data.

use std::io::IsTerminal;

use anyhow::Result;
use clap::Args;

use super::demo_pipeline::{DemoRunSummary, run_demo_pipeline};
use super::demo_tui;

/// Arguments for the demo command.
#[derive(Args, Debug, Clone)]
pub struct DemoArgs {
    /// Number of training epochs.
    #[arg(long, default_value = "5")]
    pub epochs: usize,

    /// Number of synthetic episodes to generate.
    #[arg(long, default_value = "12")]
    pub episodes: usize,

    /// Number of timesteps per synthetic episode.
    #[arg(long, default_value = "36")]
    pub episode_length: usize,

    /// Number of GPC-RANK candidates.
    #[arg(long, default_value = "12")]
    pub num_candidates: usize,

    /// State/observation dimensionality.
    #[arg(long, default_value = "4")]
    pub state_dim: usize,

    /// Action dimensionality.
    #[arg(long, default_value = "2")]
    pub action_dim: usize,

    /// Random seed for synthetic data generation.
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Disable the interactive TUI and print plain logs instead.
    #[arg(long)]
    pub plain: bool,
}

/// Run the end-to-end demo.
pub fn run_demo(args: DemoArgs) -> Result<()> {
    let args = normalize_args(args);

    if args.plain || !std::io::stdout().is_terminal() {
        return run_plain_demo(args);
    }

    demo_tui::run_showcase(args)
}

fn run_plain_demo(args: DemoArgs) -> Result<()> {
    tracing::info!("=== GPC End-to-End Demo ===");
    tracing::info!(
        "Config: state_dim={}, action_dim={}, epochs={}, candidates={}, episodes={}, episode_length={}",
        args.state_dim,
        args.action_dim,
        args.epochs,
        args.num_candidates,
        args.episodes,
        args.episode_length
    );

    let summary = run_demo_pipeline(&args)?;
    log_demo_summary(&summary);

    tracing::info!("=== Demo Complete ===");
    Ok(())
}

fn log_demo_summary(summary: &DemoRunSummary) {
    tracing::info!(
        "Pipeline summary: dataset={} episodes / {} transitions",
        summary.dataset_episodes,
        summary.dataset_transitions
    );
    tracing::info!(
        "World model epochs/losses: phase1={:?}/{:?} phase2={:?}/{:?}",
        summary.world_model_phase1.final_epoch,
        summary.world_model_phase1.final_loss,
        summary.world_model_phase2.final_epoch,
        summary.world_model_phase2.final_loss
    );
    tracing::info!(
        "Policy epoch/loss: {:?}/{:?}",
        summary.policy.final_epoch,
        summary.policy.final_loss
    );
    tracing::info!(
        "Selected action sequence: [{}, {}, {}]",
        summary.evaluation.selected_action_shape[0],
        summary.evaluation.selected_action_shape[1],
        summary.evaluation.selected_action_shape[2]
    );
    tracing::info!(
        "First action preview: {}",
        preview_vector(&summary.evaluation.selected_action_values, 4)
    );
}

fn normalize_args(mut args: DemoArgs) -> DemoArgs {
    args.epochs = args.epochs.max(1);
    args.episodes = args.episodes.max(4);
    args.episode_length = args.episode_length.max(8);
    args.num_candidates = args.num_candidates.max(4);
    args.state_dim = args.state_dim.max(2);
    args.action_dim = args.action_dim.max(1);
    args
}

fn preview_vector(values: &[f32], limit: usize) -> String {
    let mut parts = values
        .iter()
        .take(limit)
        .map(|value| format!("{value:.2}"))
        .collect::<Vec<_>>();
    if values.len() > limit {
        parts.push("...".to_string());
    }
    format!("[{}]", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_args_enforces_minimums() {
        let args = DemoArgs {
            epochs: 0,
            episodes: 1,
            episode_length: 6,
            num_candidates: 0,
            state_dim: 1,
            action_dim: 0,
            seed: 7,
            plain: true,
        };

        let normalized = normalize_args(args);

        assert_eq!(normalized.epochs, 1);
        assert_eq!(normalized.episodes, 4);
        assert_eq!(normalized.episode_length, 8);
        assert_eq!(normalized.num_candidates, 4);
        assert_eq!(normalized.state_dim, 2);
        assert_eq!(normalized.action_dim, 1);
    }

    #[test]
    fn demo_pipeline_returns_coherent_summary() {
        let args = DemoArgs {
            epochs: 1,
            episodes: 4,
            episode_length: 8,
            num_candidates: 4,
            state_dim: 4,
            action_dim: 2,
            seed: 42,
            plain: true,
        };

        let normalized = normalize_args(args);
        let summary = run_demo_pipeline(&normalized).unwrap();

        assert_eq!(summary.dataset_episodes, 4);
        assert_eq!(summary.dataset_transitions, 4 * (8 - 1));
        assert_eq!(summary.world_model_phase1.final_epoch, Some(1));
        assert_eq!(summary.world_model_phase2.final_epoch, Some(1));
        assert_eq!(summary.policy.final_epoch, Some(1));
        assert_eq!(summary.evaluation.selected_action_shape, [1, 4, 2]);
        assert_eq!(summary.evaluation.selected_action_values.len(), 8);
    }
}
