import { fetchSimulation, fetchSnapshot, healthCheck } from './api'
import type { MissionPlayback, PlannerMode, RuntimeSnapshot } from './types'

export type PlannerRuntimeSource = 'browser' | 'server'

export interface PlannerStatus {
  source: PlannerRuntimeSource
  phase: 'booting' | 'ready' | 'planning' | 'fallback'
  message: string
}

export interface PlannerClient {
  readonly source: PlannerRuntimeSource
  snapshot(): Promise<RuntimeSnapshot>
  simulate(
    missionId: string,
    mode: PlannerMode,
    numCandidates: number,
  ): Promise<MissionPlayback>
  close(): void
}

export interface PlannerSession {
  client: PlannerClient
  snapshot: RuntimeSnapshot
}

type WorkerStatus = Extract<
  import('./types').WorkerResponse,
  { type: 'status' }
>

const RETRY_DELAY_MS = 800

function delay(ms: number) {
  return new Promise((resolve) => globalThis.setTimeout(resolve, ms))
}

function describeRuntimeSource(source: PlannerRuntimeSource | null): string {
  switch (source) {
    case 'browser':
      return 'browser / wasm'
    case 'server':
      return 'server / rest'
    default:
      return 'connecting…'
  }
}

export function formatPlannerSource(source: PlannerRuntimeSource | null): string {
  return describeRuntimeSource(source)
}

export async function createPlannerSession(
  onStatus?: (status: PlannerStatus) => void,
): Promise<PlannerSession> {
  if (canUseBrowserWorker()) {
    try {
      onStatus?.({
        source: 'browser',
        phase: 'booting',
        message: 'Starting browser / WASM planner…',
      })
      const browserSession = await createBrowserPlannerSession(onStatus)
      onStatus?.({
        source: 'browser',
        phase: 'ready',
        message: 'Browser / WASM planner ready.',
      })
      return browserSession
    } catch (error) {
      onStatus?.({
        source: 'server',
        phase: 'fallback',
        message: `Browser planner unavailable; using REST fallback. ${
          error instanceof Error ? error.message : 'Unknown worker failure.'
        }`,
      })
    }
  } else {
    onStatus?.({
      source: 'server',
      phase: 'fallback',
      message: 'Browser workers are unavailable; using REST fallback.',
    })
  }

  await waitForServerReady(onStatus)

  const client = new RestPlannerClient()
  const snapshot = await client.snapshot()

  onStatus?.({
    source: 'server',
    phase: 'ready',
    message: 'Server / REST planner ready.',
  })

  return { client, snapshot }
}

function canUseBrowserWorker(): boolean {
  return typeof window !== 'undefined' && typeof Worker !== 'undefined'
}

async function waitForServerReady(onStatus?: (status: PlannerStatus) => void) {
  onStatus?.({
    source: 'server',
    phase: 'booting',
    message: 'Waiting for the Rust planner server to come online…',
  })

  while (!(await healthCheck())) {
    await delay(RETRY_DELAY_MS)
  }
}

async function createBrowserPlannerSession(
  onStatus?: (status: PlannerStatus) => void,
): Promise<PlannerSession> {
  const worker = new Worker(new URL('./workers/planner.worker.ts', import.meta.url), {
    type: 'module',
  })
  const client = new BrowserPlannerClient(worker, onStatus)

  try {
    const snapshot = await client.snapshot()
    return { client, snapshot }
  } catch (error) {
    client.close()
    throw error
  }
}

class BrowserPlannerClient implements PlannerClient {
  readonly source = 'browser' as const
  private readonly worker: Worker
  private readonly onStatus?: (status: PlannerStatus) => void
  private currentOperation:
    | {
        resolve: (response: import('./types').WorkerResponse) => void
        reject: (error: Error) => void
      }
    | null = null
  private snapshotCache: RuntimeSnapshot | null = null

  constructor(worker: Worker, onStatus?: (status: PlannerStatus) => void) {
    this.worker = worker
    this.onStatus = onStatus
    this.worker.addEventListener('message', this.handleMessage)
    this.worker.addEventListener('error', this.handleError)
    this.worker.addEventListener('messageerror', this.handleMessageError)
  }

  async snapshot(): Promise<RuntimeSnapshot> {
    if (this.snapshotCache) {
      return this.snapshotCache
    }

    const response = await this.request({ type: 'bootstrap' })
    if (response.type !== 'snapshot') {
      throw new Error(`Unexpected worker response during bootstrap: ${response.type}`)
    }

    this.snapshotCache = response.snapshot
    return response.snapshot
  }

  async simulate(
    missionId: string,
    mode: PlannerMode,
    numCandidates: number,
  ): Promise<MissionPlayback> {
    const response = await this.request({
      type: 'simulate',
      missionId,
      mode,
      numCandidates,
    })

    if (response.type !== 'playback') {
      throw new Error(`Unexpected worker response during simulation: ${response.type}`)
    }

    return response.playback
  }

  close() {
    this.worker.removeEventListener('message', this.handleMessage)
    this.worker.removeEventListener('error', this.handleError)
    this.worker.removeEventListener('messageerror', this.handleMessageError)
    this.currentOperation = null
    this.worker.terminate()
  }

  private request(
    message: import('./types').WorkerRequest,
  ): Promise<import('./types').WorkerResponse> {
    if (this.currentOperation) {
      return Promise.reject(new Error('planner worker is already handling a request'))
    }

    return new Promise((resolve, reject) => {
      this.currentOperation = { resolve, reject }
      this.worker.postMessage(message)
    })
  }

  private handleMessage = (event: MessageEvent<import('./types').WorkerResponse>) => {
    const response = event.data

    if (response.type === 'status') {
      this.onStatus?.({
        source: this.source,
        phase: mapWorkerPhase(response.phase),
        message: response.message,
      })
      return
    }

    const operation = this.currentOperation
    this.currentOperation = null

    if (!operation) {
      return
    }

    if (response.type === 'error') {
      operation.reject(new Error(response.message))
      return
    }

    operation.resolve(response)
  }

  private handleError = () => {
    const operation = this.currentOperation
    this.currentOperation = null

    if (!operation) {
      return
    }

    operation.reject(new Error('browser planner worker failed'))
  }

  private handleMessageError = () => {
    this.handleError()
  }
}

class RestPlannerClient implements PlannerClient {
  readonly source = 'server' as const

  async snapshot(): Promise<RuntimeSnapshot> {
    return fetchSnapshot()
  }

  async simulate(
    missionId: string,
    mode: PlannerMode,
    numCandidates: number,
  ): Promise<MissionPlayback> {
    return fetchSimulation(missionId, mode, numCandidates)
  }

  close() {}
}

function mapWorkerPhase(phase: WorkerStatus['phase']): PlannerStatus['phase'] {
  switch (phase) {
    case 'initializing':
      return 'booting'
    case 'planning':
      return 'planning'
    default:
      return 'booting'
  }
}
