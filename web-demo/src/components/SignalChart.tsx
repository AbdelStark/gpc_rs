import type { RuntimeOverview } from '../types'

interface SignalChartProps {
  title: string
  accent: 'world' | 'policy'
  points: RuntimeOverview['world_loss_curve']
}

function buildPath(points: number[]): string {
  if (points.length === 0) return ''
  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = Math.max(max - min, 1e-4)

  return points
    .map((value, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * 100
      const y = 32 - ((value - min) / span) * 26
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}

function buildArea(points: number[]): string {
  if (points.length === 0) return ''
  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = Math.max(max - min, 1e-4)

  const coords = points.map((value, index) => {
    const x = (index / Math.max(points.length - 1, 1)) * 100
    const y = 32 - ((value - min) / span) * 26
    return `${x.toFixed(2)} ${y.toFixed(2)}`
  })

  return `M 0 34 L ${coords.map((c, i) => `${i === 0 ? '' : 'L '}${c}`).join(' ')} L 100 34 Z`
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
        <path d="M 0 34 L 100 34" className="signal-card__baseline" />
        <path d={buildArea(points)} className="signal-card__area" />
        <path d={buildPath(points)} className="signal-card__line" />
      </svg>
    </section>
  )
}
