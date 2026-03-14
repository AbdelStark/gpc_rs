<identity>
Rust implementation of Generative Robot Policies (GPC) via Predictive World Modeling, using Burn and Tract ML frameworks.
Paper: https://arxiv.org/pdf/2502.00622
Reference impl (Python): https://github.com/han20192019/gpc_code
Sister project: https://github.com/AbdelStark/jepa-rs
</identity>

<stack>

| Layer       | Technology   | Version  | Notes                                    |
|-------------|--------------|----------|------------------------------------------|
| Language    | Rust         | 2024 ed  | rust-version = 1.85, stable toolchain    |
| ML (train)  | Burn         | 0.16     | NdArray + WGPU backends                  |
| ML (infer)  | Tract        | 0.21     | ONNX model loading and inference         |
| Serializer  | serde        | 1.x      | JSON + derive                            |
| Error       | thiserror    | 2.x      | Typed errors in gpc-core                 |
| CLI         | clap         | 4.x      | Derive-based subcommands                 |
| Async       | tokio        | 1.x      | Full features, used in training loops    |
| Logging     | tracing      | 0.1      | With env-filter subscriber               |
| Testing     | cargo test   | built-in | + approx for float comparisons           |

</stack>

<structure>
gpc_rs/
├── gpc-core/         # Core traits, error types, shared abstractions [create/modify]
├── gpc-policy/       # Diffusion-based action policy [create/modify]
├── gpc-world/        # Predictive world model (phase 1 + phase 2) [create/modify]
├── gpc-eval/         # GPC-RANK and GPC-OPT evaluation strategies [create/modify]
├── gpc-train/        # Training orchestration and data loading [create/modify]
├── gpc-compat/       # Checkpoint loading, ONNX, PyTorch interop [create/modify]
├── gpc-cli/          # CLI binary (gpc command) [create/modify]
├── .codex/skills/    # Agentic skill definitions [modify with care]
├── .github/workflows # CI pipeline [READ ONLY — modifications need approval]
├── Cargo.toml        # Workspace root [modify with care]
└── README.md         # Project description [modify]
</structure>

<commands>

| Task             | Command                                    | Notes                            |
|------------------|--------------------------------------------|----------------------------------|
| Check            | `cargo check --workspace`                  | Fast type/syntax validation      |
| Build            | `cargo build --workspace`                  | Full debug build                 |
| Build (release)  | `cargo build --workspace --release`        | Optimized — slow, use sparingly  |
| Test             | `cargo test --workspace`                   | All unit + integration tests     |
| Test (single)    | `cargo test -p gpc-core`                   | Test one crate only              |
| Clippy           | `cargo clippy --workspace --all-targets -- -D warnings` | Must pass with zero warnings |
| Format           | `cargo fmt --all`                          | Apply rustfmt                    |
| Format (check)   | `cargo fmt --all -- --check`               | CI check — no modifications      |
| Run CLI          | `cargo run -p gpc-cli -- <subcommand>`     | e.g. `train`, `eval`, `checkpoint` |

</commands>

<conventions>
<code_style>
  Naming: snake_case for functions/variables/modules, PascalCase for types/traits/enums, SCREAMING_SNAKE for constants.
  Files: snake_case.rs — one module per file, mod.rs only when directory has multiple children.
  Imports: Group by std → external crates → workspace crates → local modules. Separate groups with blank line.
  Generics: Use `B: burn::tensor::backend::Backend` for backend-generic code (follow jepa-rs pattern).
  Errors: Use `gpc_core::Result<T>` and `GpcError` for fallible operations. Propagate with `?`.
  Visibility: Default to private. Use `pub` only for API surface. Use `pub(crate)` for internal sharing.
  Documentation: `///` doc comments on all public items. No doc comments on private items unless non-obvious.
</code_style>

<patterns>
  <do>
    — Use Burn's generic Backend pattern: `fn forward<B: Backend>(tensor: Tensor<B, 2>)` for all model code.
    — Implement `Module` trait from Burn for neural network layers.
    — Use `#[derive(Config)]` from Burn for model hyperparameters.
    — Use `#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]` on data structures.
    — Write unit tests in the same file: `#[cfg(test)] mod tests { ... }`.
    — Use `tracing::info!` / `tracing::debug!` for logging, not `println!`.
    — Use `approx::assert_relative_eq!` for floating point comparisons in tests.
    — Keep functions short. Extract helpers when a function exceeds ~50 lines.
  </do>
  <dont>
    — Don't use `unwrap()` or `expect()` in library code — use `?` with proper error types.
    — Don't hardcode backend types — always be generic over `B: Backend`.
    — Don't use `println!` — use `tracing` macros.
    — Don't create God modules — split into focused submodules.
    — Don't use `unsafe` unless absolutely necessary and well-documented.
    — Don't add dependencies to workspace root — add to individual crate Cargo.toml.
  </dont>
</patterns>

<commit_conventions>
  Format: `type(scope): description` — lowercase, imperative mood, no period.
  Types: feat, fix, refactor, test, docs, chore, ci.
  Scope: crate name without `gpc-` prefix (e.g., `feat(core): add tensor utilities`).
  Keep commits atomic — one logical change per commit.
</commit_conventions>
</conventions>

<architecture_overview>
The GPC framework has two independently trained components that combine at inference:

1. **Diffusion Policy** (gpc-policy): Generative model producing candidate action sequences from observations.
   - Trained independently on demonstration data.
   - At inference, generates N candidate trajectories.

2. **World Model** (gpc-world): Predicts future states given current state + actions.
   - Phase 1: Single-step prediction warmup.
   - Phase 2: Multi-step rollout training.

3. **Evaluation** (gpc-eval): Combines both at inference time.
   - GPC-RANK: Sample N trajectories → score each via world model → select best.
   - GPC-OPT: Iteratively refine trajectories using world model gradients.

Data flow: observations → policy → candidate actions → world model → predicted states → scoring → best action.
</architecture_overview>

<workflows>
<new_feature>
  1. Identify target crate(s) based on the feature domain.
  2. Implement in the appropriate crate's `src/` directory.
  3. Add public API to the crate's `lib.rs` via `pub mod` and re-exports.
  4. Write tests: `#[cfg(test)] mod tests` in the same file.
  5. Run `cargo test -p <crate>` — all must pass.
  6. Run `cargo clippy --workspace --all-targets -- -D warnings` — zero warnings.
  7. Run `cargo fmt --all` — apply formatting.
  8. Commit: `feat(scope): description`.
</new_feature>

<bug_fix>
  1. Write a failing test that reproduces the bug.
  2. Fix the code.
  3. Verify the test passes.
  4. Run full `cargo test --workspace`.
  5. Commit: `fix(scope): description`.
</bug_fix>

<add_dependency>
  1. Add to `[workspace.dependencies]` in root Cargo.toml with version.
  2. Add to the specific crate's `[dependencies]` with `workspace = true`.
  3. Run `cargo check --workspace` to verify resolution.
  4. Commit: `chore(scope): add dependency-name`.
</add_dependency>
</workflows>

<boundaries>
<forbidden>
  DO NOT modify under any circumstances:
  — .env, .env.* (credentials, secrets)
  — Any file containing API keys, tokens, or passwords
</forbidden>

<gated>
  Modify ONLY with explicit human approval:
  — .github/workflows/* (CI pipeline)
  — Cargo.toml workspace root (dependency versions)
  — rust-toolchain.toml (toolchain version)
</gated>

<safety_checks>
  Before ANY destructive operation:
  1. State what you're about to do and why.
  2. State what could go wrong.
  3. Wait for confirmation.
</safety_checks>
</boundaries>

<troubleshooting>
<known_issues>

| Symptom                                  | Cause                         | Fix                                         |
|------------------------------------------|-------------------------------|---------------------------------------------|
| `unresolved import` after adding dep     | Missing `workspace = true`    | Add dep to crate's Cargo.toml properly      |
| Burn compile errors with backend         | Missing generic `B` parameter | Make function generic over `B: Backend`     |
| Tract model loading fails                | ONNX opset mismatch           | Check tract-onnx supported ops              |
| `cargo test` hangs on GPU tests          | WGPU backend timeout          | Use `burn-ndarray` backend for tests        |

</known_issues>

<recovery_patterns>
  1. Read the full error message — Rust errors are precise and usually contain the fix.
  2. Run `cargo check --workspace` to isolate compilation errors.
  3. Check `Cargo.lock` for version conflicts if dependency errors occur.
  4. For Burn API questions, reference jepa-rs as the canonical pattern source.
  5. If still stuck, state the problem clearly and ask for help.
</recovery_patterns>
</troubleshooting>

<environment>
  Harness: Claude Code / Codex / compatible agents
  File system scope: Full workspace access
  Network access: Available for crate downloads
  Tool access: git, cargo, rustup, shell
  CI: GitHub Actions (check, test, clippy, fmt)
</environment>

<skills>
Modular skills are in .codex/skills/ (symlinked at .claude/skills/ and .agents/skills/).

Available skills:
— burn-patterns.md: Burn ML framework patterns, Module trait, Backend generics, tensor ops.
— testing.md: Testing strategies for ML code — unit tests, property-based, approx comparisons.
— model-porting.md: Porting PyTorch models to Burn/Tract — weight mapping, architecture translation.
— debugging.md: Debugging Rust ML code — common errors, tensor shape issues, backend problems.
</skills>

<memory>
<project_decisions>
  2026-03-14: Workspace with 7 crates mirroring jepa-rs architecture — Separation of concerns, independent compilation, clear dependency graph — Single crate (too monolithic), 3 crates (insufficient separation).
  2026-03-14: Burn 0.16 + Tract 0.21 — Match jepa-rs versions for compatibility — Latest versions (untested compatibility).
  2026-03-14: Edition 2024, rust-version 1.85 — Latest stable features — Edition 2021 (missing newer features).
  2026-03-14: GpcError with thiserror — Typed, composable errors — anyhow in library code (loses type info).
</project_decisions>

<lessons_learned>
  — Always use burn-ndarray backend in tests, not WGPU (deterministic, no GPU required).
  — When porting from Python reference, map numpy ops to ndarray or Burn tensor ops.
  — Tract is for inference only — use Burn for training.
</lessons_learned>
</memory>
