import type { MissionPlayback, PlannerMode, RuntimeSnapshot } from './types'

const BASE = '/api'

export async function fetchSnapshot(): Promise<RuntimeSnapshot> {
  const response = await fetch(`${BASE}/snapshot`)
  if (!response.ok) {
    throw new Error(`snapshot failed: ${response.status}`)
  }
  return response.json()
}

export async function fetchSimulation(
  missionId: string,
  mode: PlannerMode,
  numCandidates: number,
): Promise<MissionPlayback> {
  const response = await fetch(`${BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mission_id: missionId,
      mode,
      num_candidates: numCandidates,
    }),
  })
  if (!response.ok) {
    throw new Error(`simulate failed: ${response.status}`)
  }
  return response.json()
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${BASE}/health`)
    return response.ok
  } catch {
    return false
  }
}
