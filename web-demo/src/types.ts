export type PlannerMode = 'rank' | 'opt'

export interface RuntimeBuildConfig {
  dataset_seed: number
  dataset_episodes: number
  episode_length: number
  world_phase1_epochs: number
  world_phase2_epochs: number
  policy_epochs: number
  batch_size: number
  recommended_candidates: number
  recommended_opt_steps: number
}

export const DEFAULT_RUNTIME_BUILD_CONFIG: RuntimeBuildConfig = {
  dataset_seed: 42,
  dataset_episodes: 72,
  episode_length: 14,
  world_phase1_epochs: 12,
  world_phase2_epochs: 8,
  policy_epochs: 16,
  batch_size: 24,
  recommended_candidates: 18,
  recommended_opt_steps: 2,
}

export interface Vec2 {
  x: number
  y: number
}

export interface Obstacle {
  x: number
  y: number
  radius: number
}

export interface ArmPose {
  base: Vec2
  elbow: Vec2
  effector: Vec2
  theta1: number
  theta2: number
}

export interface MissionSpec {
  id: string
  title: string
  eyebrow: string
  summary: string
  accent: string
  difficulty: string
  start_angles: [number, number]
  goal: Vec2
  obstacles: Obstacle[]
  max_steps: number
}

export interface CandidateSummary {
  rank: number
  reward: number
  clearance: number
  terminal_distance: number
  effector_path: Vec2[]
}

export interface PlanningFrame {
  step: number
  pose: ArmPose
  executed_path: Vec2[]
  policy_path: Vec2[]
  ranked_path: Vec2[]
  optimized_path: Vec2[]
  candidates: CandidateSummary[]
  selected_action: [number, number]
  goal_distance: number
  min_clearance: number
  world_model_error: number
  reward_mean: number
  reward_best: number
  reward_spread: number
}

export interface MissionSummary {
  success: boolean
  final_goal_distance: number
  min_clearance: number
  average_world_error: number
  executed_steps: number
  mode: string
}

export interface MissionPlayback {
  mission: MissionSpec
  frames: PlanningFrame[]
  summary: MissionSummary
}

export interface RuntimeOverview {
  dataset_episodes: number
  dataset_transitions: number
  state_dim: number
  action_dim: number
  obs_horizon: number
  pred_horizon: number
  bootstrap_ms: number
  world_loss_curve: number[]
  policy_loss_curve: number[]
  recommended_candidates: number
  recommended_opt_steps: number
  build_config: RuntimeBuildConfig
}

export interface RuntimeSnapshot {
  overview: RuntimeOverview
  missions: MissionSpec[]
}

export type WorkerRequest =
  | { type: 'bootstrap' }
  | {
      type: 'rebuild'
      config: RuntimeBuildConfig
    }
  | {
      type: 'simulate'
      missionId: string
      mode: PlannerMode
      numCandidates: number
    }

export type WorkerResponse =
  | { type: 'status'; phase: 'initializing' | 'rebuilding' | 'planning'; message: string }
  | { type: 'snapshot'; snapshot: RuntimeSnapshot }
  | { type: 'playback'; playback: MissionPlayback }
  | { type: 'error'; message: string }
