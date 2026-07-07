# Cardinal Court — Voice Receptionist

A hosted **LiveKit voice agent** that acts as the front-desk receptionist for
**Cardinal Court**, a (fictional) 10-storey office building at 120 Southwark Street,
London SE1 0SW. Open the link, allow your mic, and talk to it — it answers building
questions (floors, tenants, amenities, access, transport, hours) grounded in the
building fact pack, and says "I don't have that" instead of making things up.

Built for the Venaglass AI Product Engineer take-home.

---

## Live demo

- **Live URL:** https://zaeemansari70.github.io/cardinal-court-voice-receptionist/
- **Before you dial in:** best in **Chrome**, **allow the microphone**, tap the mic
  button in the bottom-right corner, and give it **~10–15 seconds** to connect on the
  first call (free-tier cold start). Speak naturally; you can interrupt it.

## What it does

It's a real **speech-in / speech-out** agent (not a text bot with a mic), so you just
talk to it. It greets you once, answers concisely like a front desk, and handles the
awkward cases gracefully:

- **Grounded answers** — "Which floor is Loom on?" → *"Loom is on floors 3 and 4; their
  front desk is on 3."* Directions, café hours, showers, step-free access, parking,
  prayer room, meeting rooms, roof terrace, etc.
- **Honest refusals** — an unknown company, a tenant's direct number, or a named
  employee's mobile → it says it doesn't have that and offers reception, rather than
  inventing an answer.
- **Safety carve-out** — for a genuine medical/fire emergency it tells you to call 999
  and points you to the nearest defibrillator / assembly point, even though that isn't
  strictly a "building fact."

## Stack

- **Language:** Python (the most mature LiveKit Agents SDK — quickest path to a reliable
  voice agent in the time box).
- **Framework:** [LiveKit Agents](https://docs.livekit.io/agents/) — `AgentSession` voice
  pipeline with semantic turn detection and background voice cancellation.
- **Models:** [LiveKit Inference](https://docs.livekit.io/agents/models/inference/) — a
  single key path for STT (Deepgram Nova-3), LLM (OpenAI GPT), and TTS (Cartesia Sonic-3),
  so there's no juggling separate provider API keys.
- **Hosting:** deployed to **LiveKit Cloud** (`lk agent deploy`); the public URL is a
  tiny static page ([`docs/index.html`](docs/index.html)) that mounts the **LiveKit
  Agent Embed Widget**, served on **GitHub Pages** (no custom voice frontend to build).
- **Scaffold:** the `agent-starter-python` template (via `lk agent init`).

## Grounding approach (and why no RAG / MCP)

The whole fact pack is small (**under 2k tokens**) and static, so it lives **in the
system prompt in full** — see `INSTRUCTIONS` in [`src/agent.py`](src/agent.py). This is a
deliberate scoping decision:

- The hard cases here — **Loom on floors 3 *and* 4**, **couriers → parcel room, not the
  tenant's floor**, refusing an **unknown company** or a **tenant's direct number** — are
  **instruction-following** problems, not retrieval problems. Retrieval wouldn't make them
  more correct; a clear prompt does.
- For a pack this size, RAG/embeddings would add chunking, an index, retrieval latency,
  and a new failure mode (missing the right chunk) for **zero accuracy gain**, and voice
  agents are latency-sensitive.

The prompt is structured as: **role & tone → voice output rules → grounding rules →
safety carve-out → "things people get wrong" (the traps) → the full fact pack**. A short
greeting fires once on connect via `session.generate_reply(...)`.

### Hallucination & false-refusal policy

- **Never invent** tenants, floors, numbers, emails, hours, amenities, transport, parking,
  names, or procedures. Missing fact → say so + offer reception on 020 7946 0120.
- **Never over-refuse**: if a fact *is* in the pack, answer it directly. (A grounded agent
  that refuses real building facts is as bad as one that hallucinates.)
- **999 safety guidance is exempt** from the grounding rule — it's never suppressed.

## Verifying behavior (evals)

Grounding is verified **headlessly** (no mic needed) with the LiveKit Agents testing
framework: each test sends one user turn to the agent's LLM and uses an LLM judge to
score the reply. The suite in [`tests/test_agent.py`](tests/test_agent.py) covers the
core facts, the traps, every awkward refusal, and the 999 carve-out.

```bash
uv run pytest          # 15 behavioral evals, all passing
```

This is also how I'd catch a prompt regression before redeploying.

## Run it locally

Requires [`uv`](https://docs.astral.sh/uv/) and the
[LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) (`lk`).

```bash
uv sync                                   # install deps (manages its own Python ≥3.11)
uv run -m livekit.agents download-files   # one-time: turn-detector / VBC model files

# Talk to it in your terminal (mic + speakers):
uv run python src/agent.py console

# Or run it as a worker for a frontend / the cloud:
uv run python src/agent.py dev
```

### Environment variables

Copy `.env.example` → `.env.local` and fill in your LiveKit project credentials:

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

`.env.local` is git-ignored and never committed. Because we use LiveKit Inference, these
three are the **only** secrets needed — no separate STT/LLM/TTS keys.

## Deploy

```bash
lk agent create --region eu-central --secrets-file .env.local   # first time
lk agent deploy --secrets-file .env.local                       # subsequent versions
lk agent status                                                 # confirm Running
```

### Public URL

LiveKit's hosted Sandbox is deprecated (creation is disabled), so the public,
login-free URL is a small static page that mounts the **Agent Embed Widget**:

1. `docs/index.html` embeds `https://cloud.livekit.io/embed-popup.js` with
   `data-lk-agent="<agent-id>"`, and is served on **GitHub Pages** (Settings → Pages →
   branch `main`, folder `/docs`).
2. In the Cloud dashboard: **Agents → cardinal-court → Embed → Install** → add the Pages
   origin (e.g. `https://<username>.github.io`) to **Allowed origins**, then toggle
   **Embed widget on** and **Save**.

The widget only loads on origins allow-listed in the dashboard, so the agent ID in the
page is not a secret.

## Known limitations

- Free-tier **cold start** on the first connection (~10–15s).
- The whole pack is in-context on every turn — fine at this size, but this doesn't scale
  to a large or frequently-changing knowledge base.
- No memory across calls; each session starts fresh.
- English-first (Deepgram is set to multi-language STT, but the prompt/facts are English).

## What I'd do next (out of scope for the time box)

- **Structured tools / a small MCP server** for the building data, so facts can change
  without editing the prompt (and to talk through prompt-vs-tool grounding).
- **RAG** only if the knowledge base grew large — with an eval set to prove retrieval
  actually helps before adding the complexity.
- **Telephony (LiveKit SIP)** so it answers a real phone number.
- **Multi-language** responses (STT already allows it).
- **Observability / call transcripts** and a larger regression eval set wired into CI.

## Rough time split (~2h target)

- Setup, scaffold, first Cloud deploy of the default agent — ~25 min
- Grounding: system prompt + full fact pack + traps + 999 carve-out + greeting — ~30 min
- Eval suite (15 cases) + iterating the prompt until green — ~30 min
- Redeploy, live sandbox check, README, repo — ~30 min
