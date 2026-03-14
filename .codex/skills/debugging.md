---
name: debugging
description: Debugging Rust ML code — compilation errors, tensor shape mismatches, backend issues, numerical problems. Activate when encountering errors, investigating unexpected behavior, or diagnosing test failures.
prerequisites: None
---

# Debugging Guide

<purpose>
Systematic debugging for common issues in Rust ML development with Burn and Tract.
</purpose>

<context>
— Rust compiler errors are precise — read the full message before acting.
— Most runtime bugs are shape mismatches — validate tensor dimensions early.
— Use `RUST_LOG=debug` env var to enable tracing output.
</context>

<procedure>
1. **Read** the full error message. Rust errors usually contain the fix.
2. **Classify** the error:
   - Compilation → type/trait/lifetime issue (see table below)
   - Runtime panic → shape mismatch, index out of bounds, unwrap on None
   - Silent wrong output → numerical bug, weight loading error
3. **Isolate** with a minimal test case.
4. **Fix** and verify with `cargo test`.
</procedure>

<patterns>
<do>
— Print tensor shapes with `tensor.dims()` at each step when debugging.
— Use `RUST_LOG=debug cargo test` for verbose output.
— Check the Burn GitHub issues for known backend-specific bugs.
— When stuck on generics, annotate types explicitly to narrow the error.
</do>
<dont>
— Don't ignore compiler suggestions — `help:` lines are usually correct.
— Don't add `unwrap()` to silence errors during debugging — use `dbg!()` instead.
— Don't debug GPU issues on GPU — reproduce with NdArray first.
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `the trait bound B: Backend is not satisfied` | Missing trait bound on impl/fn | Add `where B: Backend` or `<B: Backend>` |
| `expected struct Tensor, found ()` | Missing return value | Remove trailing `;` from last expression |
| `use of moved value` | Tensor consumed by previous op | Clone before reuse: `x.clone()` |
| `mismatched types Tensor<B,2> vs Tensor<B,3>` | Wrong tensor rank | Use `.unsqueeze()`, `.squeeze()`, or `.reshape()` |
| Numerical NaN in output | Exploding gradients or division by zero | Add epsilon to denominators, check learning rate |
| `cargo test` OOM | Large tensors in tests | Use small dimensions (batch=2, dim=8) in tests |
| Dependency version conflict | Workspace vs crate version mismatch | Ensure crate uses `{ workspace = true }` |
| `unresolved import` | Dep in workspace but not in crate | Add dependency to the specific crate's Cargo.toml |

</troubleshooting>

<references>
— Rust error index: https://doc.rust-lang.org/error_codes/
— Burn troubleshooting: https://burn.dev/docs
</references>
