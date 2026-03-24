import type { RuntimeOverview } from '../types'

interface SignalChartProps {
  title: string
  accent: 'world' | 'policy'
  points: RuntimeOverview['world_loss_curve']
}

function buildPath(points: number[]): string {
  if (points.length === 0) {
    return ''
  }

  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = Math.max(max - min, 1e-4)

  return points
    .map((value, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * 100
      const y = 34 - ((value - min) / span) * 28
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}

export function SignalChart({ title, accent, points }: SignalChartProps) {
  const last = points.at(-1) ?? 0

  return (
    <section className={`signal-card signal-card--${accent}`}>
      <div className="signal-card__meta">
        <p>{title}</p>
        <strong>{last.toFixed(4)}</strong>
      </div>
      <svg viewBox="0 0 100 36" aria-hidden="true" className="signal-card__plot">
        <defs>
          <linearGradient id={`gradient-${accent}`} x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.18" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0.95" />
          </linearGradient>
        </defs>
        <path d="M 0 34 L 100 34" className="signal-card__baseline" />
        <path d={buildPath(points)} className="signal-card__line" />
      </svg>
    </section>
  )
}
