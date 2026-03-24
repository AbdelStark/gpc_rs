# GPC Web Demo

Interactive browser-based visualization of the GPC (Generative Policy Control) pipeline. A 2-link robot arm navigates around obstacles using a diffusion policy, a learned world model, and closed-loop replanning.

The frontend now prefers a browser-side planner runtime backed by `gpc-wasm` and a Web Worker. If the worker or WASM bootstrap is unavailable, it falls back to the existing Rust REST server automatically.

## Architecture

```
┌──────────────────┐         ┌─────────────────┐
│  gpc-demo-server │  REST   │   React + Vite  │
│  (native Rust)   │ ──────→ │   (browser)     │
│                  │  :3100  │                  │  :5174
│  trains models   │         │  visualizes      │
│  serves API      │         │  planning frames │
└──────────────────┘         └─────────────────┘
```

The server trains a diffusion policy and world model from synthetic expert demonstrations on startup (~1 second in release mode), then serves two endpoints:

- `GET /api/snapshot` — runtime overview, training loss curves, mission presets
- `POST /api/simulate` — runs closed-loop replanning for a mission, returns all planning frames

When the app falls back to the native server, it polls for readiness, fetches the snapshot, then triggers simulations. Vite proxies `/api` requests to the server.

When the browser runtime is available, the app skips the server poll entirely and boots the planner in-process. The status badge in the header shows whether the current session is using `browser / wasm` or `server / rest`.

## Running

```bash
# Terminal 1 — planner server
cd .. && cargo run --release -p gpc-demo-server

# Terminal 2 — frontend
npm install
npm run dev

# Open http://localhost:5174
```

## What It Shows

The demo visualizes the complete GPC inference loop at each planning step:

| Visual | What it represents |
|--------|--------------------|
| Faint ghost paths | K candidate trajectories from the diffusion policy, rolled through the world model |
| Orange dashed path | Raw policy sample (no evaluation) |
| Green path | GPC-RANK winner (highest-scoring candidate) |
| Blue dashed path | GPC-OPT result (gradient-refined trajectory) |
| Glowing path | Active strategy's selected trajectory |
| White trail | Executed path (composite of first-actions from each replan cycle) |
| Circular obstacles | Hazard zones penalized by the reward function |
| Crosshair | Goal target |

The control panel lets you switch between GPC-RANK and GPC-OPT, adjust the number of candidate rollouts (K), scrub through planning frames, and select different missions.

See [docs/demo-explained.md](../docs/demo-explained.md) for the full explanation.

## Stack

- **Server:** Rust, axum, Burn (NdArray backend), tokio
- **Frontend:** React 19, TypeScript, Vite, pure CSS (no UI library)
- **Typography:** Syne (headings), Instrument Sans (body), JetBrains Mono (data)

## Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server with API proxy to :3100 |
| `npm run build` | Production build |
| `npm run lint` | ESLint |
| `npm run preview` | Preview production build |
