# GPC Web Demo — What It Shows and Why

This document explains what the interactive web demo demonstrates, from first principles up through the full GPC framework. It is written for anyone who wants to understand the paper, the Rust implementation, and how every visual element in the demo maps to a concept from the research. The demo exposes three planner modes: `policy`, `rank`, and `opt`.

**Paper:** [Generative Robot Policies via Predictive World Modeling](https://arxiv.org/pdf/2502.00622) (arXiv 2502.00622)

---

## The Core Problem

You want a robot arm to reach a goal while avoiding obstacles. The naive approach is to learn a policy from expert demonstrations — "here's how an expert moved, learn to imitate that." But imitation is blind. The policy generates actions without reasoning about what will happen next. If an obstacle shifts or the starting position changes, the policy has no way to evaluate whether its proposed actions will actually work.

**The GPC paper's insight:** Don't make the policy smarter. Instead, keep the policy as a *proposal generator* and add a separate *world model* that can *imagine the consequences* of each proposal before anything executes.

This decomposition is the key idea: proposals + imagination + scoring, composed at inference time but trained independently.

---

## The Two Components

### Diffusion Policy (`gpc-policy`)

A generative model trained on expert demonstrations. It doesn't output a single action — it outputs a *distribution* over entire future action sequences.

**How it works:** Start with pure random noise (Gaussian). Over 24 denoising steps, gradually sculpt that noise into a plausible action trajectory, conditioned on what the robot has recently observed (the last 2 states). This is DDPM (Denoising Diffusion Probabilistic Models) applied to action space rather than images.

**Key property:** Because it's a generative model, you can draw *multiple independent samples*. Each sample is a different plausible trajectory — some will be good, some bad. The policy doesn't know which. It just needs to cover the space of reasonable motions.

**Architecture:**

- Flattened observation history `[batch, obs_horizon × obs_dim]` is the conditioning signal
- A sinusoidal timestep embedding tells the network where it is in the denoising chain
- N residual blocks each compute: `x + linear₂(gelu(linear₁(x) + obs_proj(obs) + time_proj(t_emb)))`
- Output: predicted noise over the flattened action sequence

**Training objective:** For a random diffusion timestep t, add noise to a ground-truth action sequence from the demonstrations, predict that noise, minimize MSE. Standard DDPM noise prediction.

**In the demo:** When you see the cluster of faint ghost paths radiating from the arm, those are 18 independent samples from the policy. Each is a different "proposal" for how the arm could move over the next 6 timesteps.

### World Model (`gpc-world`)

A dynamics network that predicts: given the current state and an action, what is the next state?

**Architecture:** An MLP that predicts *state deltas* (residual formulation):

```
next_state = current_state + DynamicsNetwork(concat(state, action))
```

The residual formulation is more stable than predicting absolute next states because the network only needs to learn the small changes, starting from a near-zero baseline.

**The `rollout` operation:** Chain predictions forward. Take the current state, predict the next state for action 1, then use that predicted state to predict for action 2, and so on for 6 steps. This gives you an *imagined trajectory* — where the arm would be if it followed a given action sequence, according to the model's learned dynamics.

**Two-phase training:**

| Phase | What it learns | Epochs | Learning rate |
|-------|---------------|--------|---------------|
| Phase 1 — single-step | Accurate one-step `(state, action) → next_state` | 12 | 3e-3 |
| Phase 2 — multi-step rollout | Accurate over full 6-step rollouts | 8 | 1.6e-3 |

Phase 2 is critical because single-step errors compound over 6 prediction steps. Phase 2 teaches the model to minimize that compounding. The lower learning rate prevents destabilizing the single-step accuracy learned in Phase 1.

**In the demo:** The "forecast error" number in the stage footer measures exactly how wrong the world model was — the Euclidean distance between where it predicted the effector would end up vs. where it actually ended up after executing the action.

---

## The Planner Modes

The policy proposes. The world model imagines consequences. The planner mode determines whether the demo shows the raw proposal, ranks multiple proposals, or refines one with gradient search.

### Policy Baseline

This is the raw diffusion sample, shown without ranking or refinement.

1. Sample one action sequence from the policy
2. Roll it through the world model
3. Execute the first action from that sample

The policy mode is useful as a baseline: it shows exactly what the generative policy proposes before the evaluator changes anything. In the demo, the orange dashed path is the raw policy trajectory, and the bright selected path follows that same line when `policy` is active.

### GPC-RANK

The simplest evaluator. Pure sampling + scoring:

1. Sample K=18 candidate action sequences from the policy
2. Roll each one through the world model → 18 imagined futures
3. Score each imagined future with a reward function
4. Pick the one with the highest reward

**The reward function** in the demo's arena:

```
reward = 1.8 × progress_toward_goal
       - 3.3 × final_distance_to_goal
       - 4.4 × collision_penalty
       + 0.45 × clearance_bonus
```

It rewards trajectories that make progress, get close to the goal, avoid obstacles, and maintain safe clearance. The collision penalty multiplier (4.4×) is severe because in real robotics, collisions are catastrophic.

**The key tradeoff:** More candidates = better coverage of the policy distribution, but more compute. The slider in the demo lets you see this tradeoff directly when `rank` is selected.

**In the demo:** The green/lime path is the RANK winner. The "top trajectories" panel on the right shows the full ranking table — each candidate's reward, clearance, and terminal distance.

The **reward spread** metric tells you how discriminative the world model is. High spread = the model sees big differences between good and bad trajectories — it's useful for selection. Near-zero spread = all trajectories look the same to the model — selection is almost random.

### GPC-OPT

The gradient-based evaluator. Instead of sampling many and picking the best, start with one policy sample and iteratively improve it:

1. Start with one action sequence from the policy
2. Roll it through the world model, compute reward
3. Estimate the gradient of reward with respect to each action dimension using finite differences — perturb each action scalar by ε=0.001, measure the reward change
4. Nudge all actions in the direction of higher reward: `actions += 0.015 × ∇reward`
5. Repeat (2 optimization steps in the demo)

The demo uses finite differences instead of automatic differentiation because the NdArray backend (used at inference) does not provide autodiff. The Autodiff wrapper is only used during training and stripped via `.valid()` to produce inference models.

**In the demo:** The blue dashed path is the OPT result. It's typically close to the RANK winner but can find slightly different paths because it continuously refines rather than selecting from a fixed candidate set.

**RANK vs. OPT tradeoff:**

| Property | GPC-RANK | GPC-OPT |
|----------|----------|---------|
| Parallelism | Embarrassingly parallel (all K rollouts independent) | Sequential (each gradient step depends on the previous) |
| Coverage | Finds the best among what the policy already proposes | Can potentially find actions the policy would never sample |
| Risk | Limited by policy quality | Can drift into regions where the world model is inaccurate |
| Compute | Linear in K | Linear in action_dims × opt_steps |

---

## The Closed-Loop Replanning

This is what makes GPC a *control* algorithm, not just a planning algorithm:

```
for each timestep:
    1. Observe current state
    2. Sample 18 candidate trajectories (each covering 6 future steps)
    3. Score all with world model + reward
    4. Select best (RANK) or refine (OPT)
    5. Execute ONLY THE FIRST ACTION of the winning trajectory
    6. Discard steps 2–6 of the plan
    7. Go back to step 1
```

**Why discard steps 2–6?** Because the world model isn't perfect. After executing step 1, the real state will differ slightly from what the model predicted. By replanning from the *actual* state every step, you self-correct. This is Model Predictive Control (MPC) — plan far, execute short, replan.

**In the demo:** The white trail is the executed path — the actual sequence of first-actions from each replanning cycle. Notice it doesn't follow any single predicted trajectory exactly. It's a composite: step 1 from plan A, then step 1 from plan B (replanned from the new state), then step 1 from plan C, etc. This is the closed loop in action.

---

## The 2-Link Arm Arena

The demo simulates a planar 2-DOF robot arm:

- **Link lengths:** L₁ = 0.78, L₂ = 0.58, base at (0, −0.02)
- **Forward kinematics:** elbow = base + L₁·(cos θ₁, sin θ₁), effector = elbow + L₂·(cos(θ₁+θ₂), sin(θ₁+θ₂))
- **Action (2 dims):** Δθ₁, Δθ₂ — joint angle changes, clipped to ±0.16 rad per step
- **State (12 dims):** θ₁, θ₂, effector.x, effector.y, goal.x, goal.y, obstacle₁.x, obstacle₁.y, obstacle₁.radius, obstacle₂.x, obstacle₂.y, obstacle₂.radius

### The Three Missions

Each mission places two circular obstacles to force non-trivial planning:

| Mission | Challenge | Difficulty |
|---------|-----------|------------|
| **Slipstream Dock** | Two obstacles force the arm to flare wide before settling into the dock | Medium |
| **Gate Thread** | A narrow aperture between two obstacles requires precise threading | Hard |
| **Late Pivot** | An obstacle near the goal forces a last-moment wrist rotation | Medium |

### Training Data Generation

Training demonstrations come from a hand-coded expert controller:

1. Jacobian-based inverse kinematics toward the goal (maps desired Cartesian velocity to joint velocities)
2. Quadratic repulsion field from obstacles (pushes the effector away when close)
3. Small Gaussian noise for diversity
4. Filtering: only episodes where final distance < 0.16 AND minimum clearance > 0.03 are kept

72 episodes × 14 timesteps each = 936 transitions for training both models.

---

## Visual Element Reference

### In the SVG Stage

| Visual | CSS Class | Paper Concept |
|--------|-----------|---------------|
| Cluster of faint paths, fading by rank | `robot-stage__candidate` | The K sampled trajectories used by `rank` and `opt` — each is a diffusion policy sample rolled through the world model |
| Orange dashed line | `robot-stage__policy` | The raw policy sample — the `policy` mode baseline with no evaluation |
| Green/lime solid line | `robot-stage__ranked` | The GPC-RANK winner: highest reward among K candidates |
| Blue dashed line | `robot-stage__optimized` | The GPC-OPT result: the gradient-refined trajectory |
| Bright glowing line | `robot-stage__selected` | The active planner mode's selected trajectory, whether that is `policy`, `rank`, or `opt` |
| White trail | `robot-stage__executed` | Ground truth — the actual closed-loop path composed of first-actions from many replanning cycles |
| Filled circles with outer rings | `robot-stage__obstacle` | Collision hazard zones that the reward function penalizes |
| Crosshair target | `robot-stage__goal` | The target position the reward function pulls toward |
| Arm links and joints | `robot-stage__arm` | Forward kinematics at the current joint angles (θ₁, θ₂) |
| Dashed circle around base | `robot-stage__reach` | The kinematic reachability boundary (L₁ + L₂ radius) |

### In the Control Rail

| Control | Paper Concept |
|---------|---------------|
| Policy / RANK / OPT toggle | The three planner modes exposed by the demo |
| Candidates slider (8–28) | K in `rank` and `opt` — trades compute for coverage of the policy distribution |
| Top trajectories list | The ranking table used by `rank`; hidden or inactive in `policy` mode |
| Reward spread | How discriminative the world model is across candidates |
| Best reward | The score of the winning trajectory |

### In the Evidence Section

| Element | Paper Concept |
|---------|---------------|
| World-model loss curve (descending) | Phase 1 + Phase 2 training convergence |
| Diffusion-policy loss curve (descending) | DDPM noise prediction loss — the policy learning the demonstration distribution |
| Forecast error (stage footer) | Direct measurement of world model accuracy at each step |
| Bootstrap time (top bar) | Training time for both models (~1.2s in native Rust) |

---

## Architecture Diagram

```
Training (offline, ~1.2s on startup):

  Expert demos → [state, action, next_state] pairs
      │
      ├── Phase 1: single-step MSE (12 epochs) ──→ WorldModel
      └── Phase 2: rollout MSE (8 epochs) ────────→ WorldModel (refined)

  Expert demos → [obs_history, action_sequence] pairs
      │
      └── DDPM noise prediction loss (16 epochs) ─→ DiffusionPolicy


Inference (per step, closed-loop):

  obs_history [1, 2, 12]
      │
      ├── policy.sample() ──→ [1, 6, 2] raw policy actions
      │
      ├── policy.sample_k(K=18) ──→ [18, 6, 2] candidate action sequences
      │           │
      │           ▼
      │   world_model.rollout() ──→ [18, 6, 12] predicted state trajectories
      │           │
      │           ▼
      │   ArenaReward.compute_reward() ──→ [18] scalar scores
      │           │
      │           ├── `policy`: execute raw policy actions
      │           ├── `rank`: argmax ──→ best_actions [1, 6, 2]
      │           └── `opt`: finite-diff gradient ascent ──→ refined_actions [1, 6, 2]
      │                     │
      │                     ▼
      └───── execute first action only: [Δθ₁, Δθ₂]
                            │
                            ▼
                    next robot state
                            │
                            ▼
                    slide obs_history window
                            │
                            ▼
                    repeat until goal reached or step limit
```

---

## Running the Demo

```bash
# 1. Start the native Rust server (trains models on startup in ~1.2s)
cargo run --release -p gpc-demo-server

# 2. In another terminal, start the frontend
cd web-demo && npm run dev

# 3. Open http://localhost:5174
```

The server trains both models from synthetic expert demonstrations on startup, then serves two REST endpoints:
- `GET /api/snapshot` — runtime overview (training stats, loss curves, missions)
- `POST /api/simulate` — runs the closed-loop replanning for a selected mission and planner mode (`policy`, `rank`, or `opt`) and returns all planning frames

The frontend polls for server readiness, fetches the snapshot, then triggers a simulation whenever you change the mission, planner mode, or candidate count.

---

## The Big Picture

The demo is a **complete, end-to-end implementation of the GPC paper** running in real time:

1. **Train** a world model and a diffusion policy from synthetic expert demonstrations
2. **At each step:** either inspect the raw policy baseline or generate diverse candidate futures via diffusion → imagine consequences via world model → score and select/refine → execute only the first action → replan from actual state
3. **Visualize** the entire reasoning process: what the policy proposes, what the world model predicts, what the evaluator selects, and what actually happens

The fundamental claim of the paper is that this decomposition — generative proposals + learned forward simulation + scoring — produces more robust behavior than any single end-to-end policy. The demo lets you watch that claim play out in real time: the ghost trajectories scatter, the world model ranks them, the best one executes, and the arm navigates around obstacles to reach the goal.
