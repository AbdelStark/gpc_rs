# gpc-rs

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/gpc_rs/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/AbdelStark/gpc_rs/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-93450a.svg?style=flat-square)](rust-toolchain.toml)
[![Paper](https://img.shields.io/badge/arXiv-2502.00622-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2502.00622)

`gpc-rs` is a Rust workspace for building and evaluating Generative Robot Policies with predictive world modeling.

The repository implements the core pieces of the GPC pipeline:

- a diffusion-based action policy
- a state-based world model
- inference-time evaluators for sampling, ranking, and optimization
- training loops, synthetic data generation, and a CLI

It is built on [Burn](https://burn.dev) for model code and [Tract](https://github.com/sonos/tract) for ONNX inspection/runtime utilities.

## Showcase

<p align="center">
  <img src="docs/assets/img/gpc-rs-screenshot-tui-1.png" alt="gpc-rs interactive demo showing pipeline progress, metrics, and ranking telemetry" width="48%" />
  <img src="docs/assets/img/gpc-rs-screenshot-tui-2.png" alt="gpc-rs interactive demo showing training curves, candidate ranking, and rollout summaries" width="48%" />
</p>

The demo TUI exposes the core stages of the pipeline: dataset generation, world-model warmup, diffusion-policy training, candidate rollout, ranking, and selected-action inspection.

## Scope

This repository is a clean systems-oriented implementation of the ideas in *Generative Robot Policies via Predictive World Modeling*. It is useful today for:

- experimenting with the GPC decomposition in Rust
- training on synthetic data
- exercising the crate APIs from Rust code
- inspecting ONNX models and checkpoint metadata
- running a full end-to-end demo through the CLI

It is not yet a full benchmarked reproduction of the reference training stack. If you are looking for pretrained models, published benchmark numbers, or a finished real-dataset evaluation pipeline, those are not here yet.

## What Is Implemented

Implemented now:

- `gpc-policy`: diffusion policy with DDPM-style denoising
- `gpc-world`: state world model with single-step and rollout training support
- `gpc-eval`: `GpcRank` and `GpcOpt`
- `gpc-train`: dataset utilities, synthetic data generation, batchers, and trainers
- `gpc-compat`: checkpoint metadata I/O and ONNX inspection/runtime helpers
- `gpc-cli`: training, evaluation demo, checkpoint inspection, config generation, and an interactive TUI showcase

Not implemented yet:

- CLI checkpoint conversion
- automatic checkpoint persistence from the training command
- non-demo evaluation that loads trained weights end to end
- published task benchmarks or model zoo artifacts

## Pipeline

The codebase follows the same high-level split as the paper:

1. Train a diffusion policy on demonstrations.
2. Train a world model to predict future states from actions.
3. At inference time, sample candidate action sequences from the policy.
4. Roll each candidate through the world model.
5. Score the resulting trajectories with a reward function.
6. Either pick the best candidate (`GPC-RANK`) or refine actions with finite-difference optimization (`GPC-OPT`).

## Quick Start

### Requirements

- Rust `1.85+`
- `cargo`
- `rustfmt` and `clippy` if you want to run the full quality gates

The workspace is pinned to stable Rust in [rust-toolchain.toml](rust-toolchain.toml).

### Build and Verify

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

### Run the Demo

Interactive TUI showcase:

```bash
cargo run -p gpc-cli -- demo
```

Plain terminal output:

```bash
cargo run -p gpc-cli -- demo --plain
```

Smaller smoke-test run:

```bash
cargo run -p gpc-cli -- demo --plain --epochs 1 --episodes 4 --episode-length 8 --num-candidates 4
```

## CLI

The `gpc` binary is the easiest way to exercise the workspace.

| Command | Purpose | Notes |
| --- | --- | --- |
| `demo` | Run the end-to-end synthetic pipeline | Interactive TUI by default, `--plain` for log output |
| `train` | Train policy, world model, or both | Uses synthetic data or `DATA_DIR/episodes.json` |
| `eval` | Run evaluator demos | The runnable path today is `--demo` |
| `checkpoint` | Inspect `.onnx`, `.bin`, and `.meta.json` files | `convert` is not implemented yet |
| `init-config` | Write a default JSON config file | Prints the generated config to stdout |

### CLI Examples

Generate a default config:

```bash
cargo run -p gpc-cli -- init-config --output gpc_config.json
```

Train on synthetic data:

```bash
cargo run -p gpc-cli -- train --synthetic --component all --epochs 20
```

Train from a dataset directory containing `episodes.json`:

```bash
cargo run -p gpc-cli -- train --data data --component world-model --epochs 50 --horizon 8
```

Run evaluator demos with randomly initialized models:

```bash
cargo run -p gpc-cli -- eval --demo --strategy rank --num-candidates 64
cargo run -p gpc-cli -- eval --demo --strategy opt --opt-steps 10
```

Inspect artifacts:

```bash
cargo run -p gpc-cli -- checkpoint --action inspect --path model.onnx
cargo run -p gpc-cli -- checkpoint --action inspect --path checkpoints/model.bin
cargo run -p gpc-cli -- checkpoint --action inspect --path checkpoints/model.meta.json
```

## Workspace Layout

| Crate | Responsibility |
| --- | --- |
| `gpc-core` | Shared config types, error handling, core traits, tensor utilities, and diffusion schedules |
| `gpc-policy` | Diffusion policy and denoising network |
| `gpc-world` | State dynamics model and reward functions |
| `gpc-eval` | `GpcRank` and `GpcOpt` evaluators |
| `gpc-train` | Dataset handling and training orchestration |
| `gpc-compat` | Checkpoint helpers and ONNX inspection/runtime |
| `gpc-cli` | Command-line entrypoint and showcase demo |

The workspace root pins shared dependencies, toolchain, and quality gates:

- Burn `0.20.1`
- stable Rust `1.85`
- CI for `check`, `test`, `clippy -D warnings`, and `fmt --check`

## Minimal Library Example

This example wires together randomly initialized components and runs `GpcRank`. It is a good starting point for understanding the public API shape.

```rust
use burn::backend::NdArray;
use burn::prelude::*;
use gpc_core::traits::Evaluator;
use gpc_eval::GpcRankBuilder;
use gpc_policy::DiffusionPolicyConfig;
use gpc_world::reward::L2RewardFunctionConfig;
use gpc_world::world_model::StateWorldModelConfig;

type B = NdArray;

fn main() -> gpc_core::Result<()> {
    let device = <B as Backend>::Device::default();

    let policy = DiffusionPolicyConfig {
        obs_dim: 20,
        action_dim: 2,
        obs_horizon: 2,
        pred_horizon: 16,
        hidden_dim: 128,
        time_embed_dim: 64,
        num_res_blocks: 2,
    }
    .init::<B>(&device);

    let world_model = StateWorldModelConfig {
        state_dim: 20,
        action_dim: 2,
        hidden_dim: 128,
        num_layers: 2,
    }
    .init::<B>(&device);

    let reward = L2RewardFunctionConfig { state_dim: 20 }
        .init::<B>(&device)
        .with_goal(Tensor::<B, 1>::zeros([20], &device));

    let evaluator = GpcRankBuilder::new(policy, world_model, reward)
        .num_candidates(32)
        .build();

    let obs = Tensor::<B, 3>::zeros([1, 2, 20], &device);
    let state = Tensor::<B, 2>::zeros([1, 20], &device);

    let action = evaluator.select_action(&obs, &state, &device)?;
    assert_eq!(action.dims(), [1, 16, 2]);

    Ok(())
}
```

For actual training loops, use `gpc-train` directly or the `gpc train` CLI command.

## Configuration

The top-level config type is `gpc_core::config::GpcConfig`. It contains five sections:

- `policy`
- `world_model`
- `training`
- `gpc_rank`
- `gpc_opt`

Generate a valid config with:

```bash
cargo run -p gpc-cli -- init-config --output gpc_config.json
```

All config structs are `serde`-serializable and expose validation methods in [gpc-core/src/config.rs](gpc-core/src/config.rs).

## Development

CI runs the same commands you should use locally:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

The workflow definition lives in [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Limitations

Current limitations worth knowing before you build on this:

- The `eval` command is mainly a demo surface today. It does not yet load trained checkpoints and run a full deployment path.
- The `checkpoint convert` command is stubbed out.
- The training command creates an output directory but does not currently persist trained model weights.
- The repository has solid unit coverage and CI, but it does not yet ship benchmark scripts or task-level regression suites.

## References

- Paper: [*Generative Robot Policies via Predictive World Modeling*](https://arxiv.org/pdf/2502.00622)
- Burn: <https://burn.dev>
- Tract: <https://github.com/sonos/tract>

## License

MIT. See [LICENSE](LICENSE).
