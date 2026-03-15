<p align="center">
  <h1 align="center">gpc-rs</h1>
  <p align="center">
    <strong>Generative Robot Policies via Predictive World Modeling — in Rust</strong>
  </p>
  <p align="center">
    <a href="https://github.com/AbdelStark/gpc_rs/actions"><img src="https://img.shields.io/github/actions/workflow/status/AbdelStark/gpc_rs/ci.yml?branch=main&style=flat-square&logo=github&label=CI" alt="CI"></a>
    <a href="https://github.com/AbdelStark/gpc_rs/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
    <a href="https://arxiv.org/pdf/2502.00622"><img src="https://img.shields.io/badge/arXiv-2502.00622-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  </p>
</p>

---

Rust implementation of **GPC** (Generative Predictive Coding for robotic policies) from the paper [*Generative Robot Policies via Predictive World Modeling*](https://arxiv.org/pdf/2502.00622). Built on [Burn](https://burn.dev) for training and [Tract](https://github.com/sonos/tract) for ONNX inference.

**gpc-rs** provides a modular, backend-agnostic framework for training diffusion-based action policies, learning predictive world models, and combining them at inference time via trajectory ranking (GPC-RANK) or gradient-based optimization (GPC-OPT).

```
                          ┌────────────────────┐
                          │   Demonstrations    │
                          └────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
          ┌──────────────────┐          ┌──────────────────┐
          │  Diffusion Policy │          │   World Model    │
          │  (gpc-policy)     │          │   (gpc-world)    │
          │                   │          │                   │
          │  DDPM denoising   │          │  Phase 1: 1-step  │
          │  Obs → Actions    │          │  Phase 2: N-step  │
          └────────┬─────────┘          └────────┬─────────┘
                   │                              │
                   │    ┌────────────────────┐    │
                   └───►│    GPC Evaluation   │◄──┘
                        │    (gpc-eval)       │
                        │                     │
                        │  GPC-RANK: sample K │
                        │   trajectories,     │
                        │   pick the best     │
                        │                     │
                        │  GPC-OPT: refine    │
                        │   via gradients     │
                        └─────────┬───────────┘
                                  │
                                  ▼
                          ┌───────────────┐
                          │  Best Action  │
                          └───────────────┘
```

## Why Rust?

| | gpc-rs (Rust) | Reference impl (Python) |
|---|---|---|
| **Runtime** | Native binary, no Python/CUDA dependency | Requires Python + PyTorch + CUDA |
| **Memory** | Rust ownership model, zero GC pauses | Python GC + PyTorch allocator |
| **Backend** | Any Burn backend (CPU, GPU, WebGPU, WASM) | CUDA-centric |
| **Type safety** | Compile-time tensor shape checks | Runtime shape errors |
| **Deployment** | Single static binary | Docker + Python environment |
| **Inference** | Tract ONNX runtime, no framework overhead | Full PyTorch runtime |

## Quick Start

### Build & Run

```bash
# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Generate a default config file
cargo run -p gpc-cli -- init-config

# Run the end-to-end demo with synthetic data
cargo run -p gpc-cli -- demo --epochs 10 --num-candidates 8
```

### CLI

The `gpc` binary provides a unified CLI for training, evaluation, and model management:

```bash
# Train the diffusion policy on demonstration data
cargo run -p gpc-cli -- train --data demos.json --component policy --epochs 100

# Train the world model (both phases)
cargo run -p gpc-cli -- train --data demos.json --component world-model --epochs 50

# Train everything together
cargo run -p gpc-cli -- train --data demos.json --component all --epochs 100

# Train with synthetic data (no dataset required)
cargo run -p gpc-cli -- train --synthetic --epochs 20 --horizon 16

# Evaluate using GPC-RANK (sample & rank trajectories)
cargo run -p gpc-cli -- eval --strategy rank --num-candidates 64

# Evaluate using GPC-OPT (gradient-based refinement)
cargo run -p gpc-cli -- eval --strategy opt --opt-steps 10

# Run eval demo with random models
cargo run -p gpc-cli -- eval --demo

# Inspect a checkpoint
cargo run -p gpc-cli -- checkpoint --action inspect --path model.bin

# Inspect an ONNX model
cargo run -p gpc-cli -- checkpoint --action inspect --path model.onnx
```

### Using gpc-rs as a Library

```rust
use burn::prelude::*;
use burn_ndarray::{NdArray, NdArrayDevice};
use gpc_core::config::{PolicyConfig, WorldModelConfig, GpcRankConfig};
use gpc_core::traits::{Policy, WorldModel, Evaluator};
use gpc_core::types::{Observation, State};
use gpc_policy::DiffusionPolicy;
use gpc_world::StateWorldModel;
use gpc_eval::GpcRank;

type B = NdArray<f32>;

fn main() {
    let device = NdArrayDevice::Cpu;

    // Configure and build a diffusion policy
    let policy_config = PolicyConfig {
        action_dim: 4,
        obs_dim: 8,
        horizon: 16,
        hidden_dim: 256,
        num_diffusion_steps: 50,
        num_residual_blocks: 4,
        time_embed_dim: 64,
    };
    let policy = DiffusionPolicy::<B>::new(&policy_config, &device);

    // Configure and build a world model
    let world_config = WorldModelConfig {
        state_dim: 8,
        action_dim: 4,
        hidden_dim: 256,
        num_residual_blocks: 4,
    };
    let world_model = StateWorldModel::<B>::new(&world_config, &device);

    // At inference: generate candidates, score via world model, pick the best
    let obs = Observation::new(Tensor::zeros([1, 8], &device));
    let state = State::new(Tensor::zeros([1, 8], &device));

    // Sample K candidate action sequences from the policy
    let candidates = policy.sample_k(&obs, 64, &device);

    // Score each candidate via world model rollout
    // GPC-RANK selects the highest-scoring trajectory
}
```

## Architecture

```
gpc-rs/
├── gpc-core          Core traits, error types, shared abstractions
│   ├── Policy           Trait: generative action policy (sample, sample_k)
│   ├── WorldModel       Trait: predictive dynamics (predict, rollout)
│   ├── RewardFunction   Trait: trajectory scoring
│   ├── Evaluator        Trait: action selection strategy
│   ├── GpcError         Typed error hierarchy (thiserror)
│   ├── DdpmSchedule     DDPM noise scheduling with precomputed alphas
│   └── tensor_utils     Flatten, unflatten, MSE, sinusoidal embeddings
│
├── gpc-policy        Diffusion-based action policy (DDPM)
│   ├── DiffusionPolicy  Iterative denoising: noise → action sequence
│   ├── DenoisingNetwork Conditional MLP with time + obs conditioning
│   └── ResidualBlock    Residual block with time & conditioning projections
│
├── gpc-world         Predictive world model
│   ├── StateWorldModel  Residual dynamics: next = state + f(state, action)
│   ├── DynamicsNetwork  MLP-based state transition predictor
│   ├── L2RewardFunction Negative L2 distance to goal state
│   └── LearnedReward    Neural network reward predictor
│
├── gpc-eval          Inference-time evaluation strategies
│   ├── GpcRank          Sample K → score each → select best trajectory
│   └── GpcOpt           Finite-difference gradient optimization of actions
│
├── gpc-train         Training orchestration and data loading
│   ├── PolicyTrainer    DDPM training loop (noise, perturb, predict, MSE)
│   ├── WorldModelTrainer Two-phase training (single-step → multi-step)
│   └── GpcDataset       Episode data, batchers, synthetic generation
│
├── gpc-compat        Checkpoint loading, ONNX, PyTorch interop
│   ├── Checkpoint       Save/load weights + metadata (binary + JSON)
│   └── OnnxInspector    Tract-based ONNX model inspection
│
└── gpc-cli           CLI binary with 5 subcommands
    ├── train            Train policy, world model, or both
    ├── eval             GPC-RANK or GPC-OPT evaluation
    ├── checkpoint       Inspect/convert model checkpoints
    ├── init-config      Generate default configuration
    └── demo             End-to-end pipeline with synthetic data
```

All model code is generic over `B: burn::tensor::backend::Backend`, enabling transparent execution on CPU (NdArray), GPU (WGPU), or WebAssembly backends.

## The GPC Framework

GPC separates robot policy learning into two independently trained components that combine at inference time:

### 1. Diffusion Policy (gpc-policy)

A DDPM-based generative model that produces candidate action sequences from observations. Given the current observation, it iteratively denoises Gaussian noise into plausible action trajectories through learned reverse diffusion steps.

- Trained independently on demonstration data
- At inference, generates K candidate trajectories in parallel
- Architecture: conditional MLP with residual blocks, timestep embeddings, and observation conditioning

### 2. World Model (gpc-world)

A predictive dynamics model that forecasts future states given a current state and action sequence. Uses residual prediction (`next_state = state + Δ`) for stable multi-step rollouts.

- **Phase 1**: Single-step prediction warmup — learn to predict one step ahead
- **Phase 2**: Multi-step rollout training — jointly supervise entire trajectories

### 3. Evaluation Strategies (gpc-eval)

At inference time, the policy and world model combine through one of two strategies:

**GPC-RANK** — Sample and rank:
1. Sample K candidate action sequences from the diffusion policy
2. Roll out each candidate through the world model
3. Score each predicted trajectory using a reward function
4. Select the highest-scoring trajectory

**GPC-OPT** — Gradient-based refinement:
1. Warm-start from a policy sample
2. Compute reward gradients via finite differences
3. Iteratively update actions: `a ← a + η · ∇ₐ R(W(s, a))`
4. Return the optimized trajectory

## Build & Test

```bash
# Full workspace build
cargo build --workspace

# Run all tests
cargo test --workspace

# Lint with zero-warning policy
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check

# Test a single crate
cargo test -p gpc-core
cargo test -p gpc-policy
cargo test -p gpc-eval
```

## Project Status

> As of 2026-03-15, this project is **alpha**. It is suitable for research, experimentation,
> and extension. It is not yet suitable for production robotics deployments. Breaking changes
> may occur in any release.

### What works

- Complete diffusion policy with DDPM sampling and training
- State-based world model with two-phase training (single-step + multi-step rollout)
- GPC-RANK trajectory ranking with configurable candidate count
- GPC-OPT finite-difference gradient optimization
- CLI with training, evaluation, checkpoint inspection, and demo commands
- Checkpoint save/load with metadata (binary weights + JSON metadata)
- ONNX model inspection via Tract
- Synthetic data generation for quick experimentation
- Comprehensive test suite with NdArray backend
- Config validation for all component configurations
- Typed error handling with context throughout the stack

### Known limitations

- Training loops use NdArray backend only (WGPU backend support is structural but untested for training)
- ONNX support covers inspection only — full graph execution is not yet production-ready
- No pretrained model registry yet (planned)
- Data loading is single-threaded
- GPC-OPT uses finite-difference gradients (autodiff integration planned)
- No real-world environment integration yet (gym/mujoco bindings)

## References

| Resource | Link |
|----------|------|
| **Paper** | [Generative Robot Policies via Predictive World Modeling](https://arxiv.org/pdf/2502.00622) |
| **Reference implementation** (Python) | [han20192019/gpc_code](https://github.com/han20192019/gpc_code) |
| **Sister project** (JEPA in Rust) | [AbdelStark/jepa-rs](https://github.com/AbdelStark/jepa-rs) |
| **Burn** (ML framework) | [burn.dev](https://burn.dev) |
| **Tract** (ONNX runtime) | [sonos/tract](https://github.com/sonos/tract) |

## Contributing

Contributions are welcome! Please ensure your changes:

1. Pass all checks: `cargo clippy --workspace --all-targets -- -D warnings`
2. Pass all tests: `cargo test --workspace`
3. Are formatted: `cargo fmt --all`
4. Follow the commit convention: `type(scope): description` (e.g., `feat(policy): add noise schedule warmup`)

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

<p align="center">
  <sub>Built with <a href="https://burn.dev">Burn</a> and <a href="https://github.com/sonos/tract">Tract</a></sub>
</p>
