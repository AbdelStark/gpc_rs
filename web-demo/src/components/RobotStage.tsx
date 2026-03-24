import type { CSSProperties } from 'react'

import type { MissionSpec, PlannerMode, PlanningFrame, Vec2 } from '../types'

interface RobotStageProps {
  mission: MissionSpec
  frame: PlanningFrame
  mode: PlannerMode
}

const VIEW_BOX = '-1.22 -0.32 2.64 1.86'

function pathFromPoints(points: Vec2[]): string {
  if (points.length === 0) {
    return ''
  }

  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(3)} ${point.y.toFixed(3)}`)
    .join(' ')
}

function formatSigned(value: number) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`
}

export function RobotStage({ mission, frame, mode }: RobotStageProps) {
  const strategyPath = mode === 'opt' ? frame.optimized_path : frame.ranked_path
  const rootStyle = {
    '--mission-accent': mission.accent,
  } as CSSProperties

  return (
    <section className="robot-stage" style={rootStyle}>
      <div className="robot-stage__glass">
        <div className="robot-stage__head">
          <div>
            <p>Live planner frame</p>
            <strong>
              step {frame.step + 1} / {mission.max_steps}
            </strong>
          </div>
          <div className="robot-stage__chips">
            <span>policy prior</span>
            <span>world-model rank</span>
            <span>GPC-OPT</span>
          </div>
        </div>

        <svg
          viewBox={VIEW_BOX}
          aria-label={`${mission.title} mission stage`}
          className="robot-stage__svg"
          role="img"
        >
          <defs>
            <radialGradient id="stageGlow" cx="50%" cy="42%" r="66%">
              <stop offset="0%" stopColor="rgba(255,255,255,0.18)" />
              <stop offset="100%" stopColor="rgba(255,255,255,0)" />
            </radialGradient>
            <pattern
              id="gridPattern"
              width="0.14"
              height="0.14"
              patternUnits="userSpaceOnUse"
            >
              <path d="M 0 0.14 L 0 0 0.14 0" className="robot-stage__grid-line" />
            </pattern>
            <filter id="softGlow">
              <feGaussianBlur stdDeviation="0.02" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <rect x="-1.25" y="-0.35" width="2.7" height="1.9" className="robot-stage__backplate" />
          <rect x="-1.25" y="-0.35" width="2.7" height="1.9" fill="url(#gridPattern)" />
          <circle cx="0" cy="-0.02" r="1.36" className="robot-stage__reach" />
          <rect x="-1.22" y="-0.32" width="2.64" height="1.86" fill="url(#stageGlow)" />

          {frame.candidates
            .slice()
            .reverse()
            .map((candidate) => (
              <path
                key={`candidate-${candidate.rank}`}
                d={pathFromPoints(candidate.effector_path)}
                className="robot-stage__candidate"
                style={
                  {
                    opacity: `${Math.max(0.12, 0.44 - candidate.rank * 0.045)}`,
                  } as CSSProperties
                }
              />
            ))}

          <path d={pathFromPoints(frame.policy_path)} className="robot-stage__policy" />
          <path d={pathFromPoints(frame.ranked_path)} className="robot-stage__ranked" />
          <path d={pathFromPoints(frame.optimized_path)} className="robot-stage__optimized" />
          <path d={pathFromPoints(strategyPath)} className="robot-stage__selected" filter="url(#softGlow)" />
          <path d={pathFromPoints(frame.executed_path)} className="robot-stage__executed" />

          {mission.obstacles.map((obstacle, index) => (
            <g key={`obstacle-${index}`}>
              <circle
                cx={obstacle.x}
                cy={obstacle.y}
                r={obstacle.radius}
                className="robot-stage__obstacle"
              />
              <circle
                cx={obstacle.x}
                cy={obstacle.y}
                r={obstacle.radius + 0.05}
                className="robot-stage__obstacle-ring"
              />
            </g>
          ))}

          <g className="robot-stage__goal">
            <circle cx={mission.goal.x} cy={mission.goal.y} r="0.06" />
            <circle cx={mission.goal.x} cy={mission.goal.y} r="0.12" />
            <path
              d={`M ${mission.goal.x - 0.18} ${mission.goal.y} L ${mission.goal.x + 0.18} ${mission.goal.y}
                  M ${mission.goal.x} ${mission.goal.y - 0.18} L ${mission.goal.x} ${mission.goal.y + 0.18}`}
            />
          </g>

          <g className="robot-stage__arm">
            <path
              d={`M ${frame.pose.base.x} ${frame.pose.base.y} L ${frame.pose.elbow.x} ${frame.pose.elbow.y} L ${frame.pose.effector.x} ${frame.pose.effector.y}`}
            />
            <circle cx={frame.pose.base.x} cy={frame.pose.base.y} r="0.05" />
            <circle cx={frame.pose.elbow.x} cy={frame.pose.elbow.y} r="0.042" />
            <circle cx={frame.pose.effector.x} cy={frame.pose.effector.y} r="0.05" />
          </g>
        </svg>

        <div className="robot-stage__footer">
          <div>
            <span>selected action</span>
            <strong>
              {formatSigned(frame.selected_action[0])} / {formatSigned(frame.selected_action[1])}
            </strong>
          </div>
          <div>
            <span>goal distance</span>
            <strong>{frame.goal_distance.toFixed(3)}</strong>
          </div>
          <div>
            <span>clearance</span>
            <strong>{frame.min_clearance.toFixed(3)}</strong>
          </div>
          <div>
            <span>forecast error</span>
            <strong>{frame.world_model_error.toFixed(3)}</strong>
          </div>
        </div>
      </div>
    </section>
  )
}
