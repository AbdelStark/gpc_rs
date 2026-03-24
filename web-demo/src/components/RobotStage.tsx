import type { CSSProperties } from 'react'

import type { MissionSpec, PlannerMode, PlanningFrame, Vec2 } from '../types'

interface RobotStageProps {
  mission: MissionSpec
  frame: PlanningFrame
  mode: PlannerMode
}

const VIEW_BOX = '-1.22 -0.32 2.64 1.86'

function pathFromPoints(points: Vec2[]): string {
  if (points.length === 0) return ''
  return points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(4)} ${p.y.toFixed(4)}`)
    .join(' ')
}

function formatSigned(value: number) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`
}

export function RobotStage({ mission, frame, mode }: RobotStageProps) {
  const strategyPath = mode === 'opt' ? frame.optimized_path : frame.ranked_path
  const rootStyle = { '--mission-accent': mission.accent } as CSSProperties

  return (
    <section className="robot-stage" style={rootStyle}>
      <div className="robot-stage__header">
        <div className="robot-stage__header-left">
          <p className="eyebrow">live planner</p>
          <strong>
            step {frame.step + 1}/{mission.max_steps}
          </strong>
        </div>
        <div className="robot-stage__legend">
          <span className="robot-stage__legend-item" data-color="executed">executed</span>
          <span className="robot-stage__legend-item" data-color="ranked">ranked</span>
          <span className="robot-stage__legend-item" data-color="optimized">optimized</span>
          <span className="robot-stage__legend-item" data-color="policy">policy</span>
          <span className="robot-stage__legend-item" data-color="candidates">candidates</span>
        </div>
      </div>

      <svg
        viewBox={VIEW_BOX}
        aria-label={`${mission.title} mission stage`}
        className="robot-stage__svg"
        role="img"
      >
        <defs>
          <radialGradient id="stageGlow" cx="50%" cy="40%" r="60%">
            <stop offset="0%" stopColor="oklch(0.20 0.02 250)" />
            <stop offset="100%" stopColor="oklch(0.10 0.012 255)" />
          </radialGradient>
          <pattern id="gridPattern" width="0.14" height="0.14" patternUnits="userSpaceOnUse">
            <path d="M 0 0.14 L 0 0 0.14 0" className="robot-stage__grid-line" />
          </pattern>
          <filter id="pathGlow">
            <feGaussianBlur stdDeviation="0.015" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="jointGlow">
            <feGaussianBlur stdDeviation="0.025" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="obstacleGrad" cx="40%" cy="35%" r="70%">
            <stop offset="0%" stopColor="var(--mission-accent, oklch(0.74 0.16 42))" stopOpacity="0.15" />
            <stop offset="100%" stopColor="var(--mission-accent, oklch(0.74 0.16 42))" stopOpacity="0.03" />
          </radialGradient>
        </defs>

        {/* background */}
        <rect x="-1.25" y="-0.35" width="2.7" height="1.9" className="robot-stage__backplate" />
        <rect x="-1.25" y="-0.35" width="2.7" height="1.9" fill="url(#gridPattern)" />
        <rect x="-1.25" y="-0.35" width="2.7" height="1.9" fill="url(#stageGlow)" />
        <circle cx="0" cy="-0.02" r="1.36" className="robot-stage__reach" />

        {/* candidate trajectories (ghost trails) */}
        {frame.candidates
          .slice()
          .reverse()
          .map((c) => (
            <path
              key={`c-${c.rank}`}
              d={pathFromPoints(c.effector_path)}
              className="robot-stage__candidate"
              style={{ opacity: Math.max(0.06, 0.3 - c.rank * 0.035) } as CSSProperties}
            />
          ))}

        {/* strategy paths */}
        <path d={pathFromPoints(frame.policy_path)} className="robot-stage__policy" />
        <path d={pathFromPoints(frame.ranked_path)} className="robot-stage__ranked" />
        <path d={pathFromPoints(frame.optimized_path)} className="robot-stage__optimized" />
        <path d={pathFromPoints(strategyPath)} className="robot-stage__selected" filter="url(#pathGlow)" />

        {/* executed path */}
        <path d={pathFromPoints(frame.executed_path)} className="robot-stage__executed" />

        {/* obstacles */}
        {mission.obstacles.map((obs, i) => (
          <g key={`obs-${i}`}>
            <circle cx={obs.x} cy={obs.y} r={obs.radius} fill="url(#obstacleGrad)" />
            <circle cx={obs.x} cy={obs.y} r={obs.radius} className="robot-stage__obstacle" />
            <circle cx={obs.x} cy={obs.y} r={obs.radius + 0.04} className="robot-stage__obstacle-ring" />
          </g>
        ))}

        {/* goal target */}
        <g className="robot-stage__goal">
          <circle cx={mission.goal.x} cy={mission.goal.y} r="0.05" />
          <circle cx={mission.goal.x} cy={mission.goal.y} r="0.10" />
          <path
            d={`M ${mission.goal.x - 0.16} ${mission.goal.y} L ${mission.goal.x + 0.16} ${mission.goal.y}
                M ${mission.goal.x} ${mission.goal.y - 0.16} L ${mission.goal.x} ${mission.goal.y + 0.16}`}
          />
        </g>

        {/* robot arm */}
        <g className="robot-stage__arm" filter="url(#jointGlow)">
          <path
            d={`M ${frame.pose.base.x} ${frame.pose.base.y} L ${frame.pose.elbow.x} ${frame.pose.elbow.y} L ${frame.pose.effector.x} ${frame.pose.effector.y}`}
          />
          <circle cx={frame.pose.base.x} cy={frame.pose.base.y} r="0.045" />
          <circle cx={frame.pose.elbow.x} cy={frame.pose.elbow.y} r="0.036" />
          <circle cx={frame.pose.effector.x} cy={frame.pose.effector.y} r="0.045" />
        </g>
      </svg>

      <div className="robot-stage__footer">
        <div>
          <span>action</span>
          <strong>
            {formatSigned(frame.selected_action[0])}&ensp;{formatSigned(frame.selected_action[1])}
          </strong>
        </div>
        <div>
          <span>goal dist</span>
          <strong>{frame.goal_distance.toFixed(3)}</strong>
        </div>
        <div>
          <span>clearance</span>
          <strong>{frame.min_clearance.toFixed(3)}</strong>
        </div>
        <div>
          <span>forecast err</span>
          <strong>{frame.world_model_error.toFixed(3)}</strong>
        </div>
      </div>
    </section>
  )
}
