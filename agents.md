# Multi-Agent Orchestration — GPC-RS

<roles>

| Role           | Model Tier | Responsibility                                          | Boundaries                                |
|----------------|------------|---------------------------------------------------------|-------------------------------------------|
| Orchestrator   | Frontier   | Decompose tasks, plan architecture, delegate, review    | NEVER writes implementation code          |
| Implementer    | Mid-tier   | Write Rust code, implement models, run tests            | NEVER makes architectural decisions       |
| Reviewer       | Frontier   | Validate correctness, catch numerical bugs, review diffs | NEVER implements fixes (sends back)       |
| ML Specialist  | Frontier   | Model porting, training loops, numerical validation     | Only operates on gpc-policy/world/eval    |
| Infra Agent    | Mid-tier   | CI, tooling, dependency management, project config      | Only operates on config/CI files          |

</roles>

<delegation_protocol>
The Orchestrator follows this decision tree:

1. **ANALYZE**: Classify the task — model implementation, infrastructure, testing, debugging.
2. **DECOMPOSE**: Break into crate-scoped sub-tasks with clear file boundaries.
3. **CLASSIFY**:
   — Routine (add test, fix clippy, update docs) → Implementer
   — Model architecture (new layer, porting from PyTorch) → ML Specialist
   — CI/tooling (workflow, dependencies, config) → Infra Agent
   — Quality gate (review PR, validate numerics) → Reviewer
4. **DELEGATE**: Issue task with full context (see task format below).
5. **INTEGRATE**: Merge results, run `cargo check --workspace`, verify coherence.
6. **REVIEW**: Final quality gate — clippy, fmt, tests all pass.
</delegation_protocol>

<task_format>
Every delegated task must include:

## Task: [Clear, actionable title]

**Objective**: [One sentence — what "done" looks like]

**Context**:
- Crate: [gpc-core | gpc-policy | gpc-world | gpc-eval | gpc-train | gpc-compat | gpc-cli]
- Files to read: [exact paths]
- Files to modify/create: [exact paths]
- Related types/traits: [names from gpc-core or other crates]
- Reference Python code: [path in reference repo if porting]

**Acceptance criteria**:
- [ ] `cargo check --workspace` passes
- [ ] `cargo test -p <crate>` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — zero warnings
- [ ] `cargo fmt --all -- --check` passes

**Constraints**:
- Do NOT modify files outside the specified crate
- Do NOT change public API of other crates without approval
- Use `B: Backend` generics for all model code
</task_format>

<parallel_execution>
Safe to parallelize:
— Work in non-overlapping crates (e.g., gpc-policy + gpc-world simultaneously)
— Writing tests for stable interfaces
— Documentation updates
— Independent feature implementations in separate crates

Must serialize:
— Changes to gpc-core (all other crates depend on it)
— Workspace Cargo.toml modifications
— CI pipeline changes
— Cross-crate API changes

Conflict resolution:
1. Check crate dependency graph before parallelizing.
2. gpc-core changes block all downstream crates.
3. If overlap detected, serialize with gpc-core first.
</parallel_execution>

<escalation>
Escalate to human when:
— Architectural decisions needed (new crate, public API design)
— Burn API limitations discovered (may need upstream issue)
— Python reference has ambiguous implementation details
— Performance requirements unclear (CPU vs GPU, latency targets)
— Security implications (model file loading, untrusted inputs)

Format:
**ESCALATION**: [one-line summary]
**Context**: [what was being done]
**Blocker**: [specific issue]
**Options**: [numbered alternatives with tradeoffs]
**Recommendation**: [which option and why]
</escalation>
