# Sentinel — Agentic Multimodal Video Moderation

Video → **Perception** (frames + audio) → **Policy** (RAG over the rulebook) → **Action** (decision matrix, audit log, review queue), orchestrated with LangGraph, including a Policy→Perception recheck loop.

All inference runs on **Groq**, so no GPU or local model downloads are needed:

| Role | Model (default) | Was |
|---|---|---|
| Policy reasoning | `openai/gpt-oss-120b` | Qwen2.5 via vLLM |
| Frame vision | `qwen/qwen3.8-27b` | LLaVA-1.5-7B |
| Speech-to-text | `whisper-large-v3` (word timestamps) | local Whisper |
| RAG embeddings | `all-MiniLM-L6-v2` (local, CPU/MPS) | same |

## Project structure
```
.
├── agents/
│   ├── orchestrator.py      # LangGraph pipeline: Perception → Policy → Action (+ recheck loop)
│   ├── perception_agent.py  # frame descriptions + audio transcription per segment
│   ├── scene_detect.py      # scene-aware segmentation (PySceneDetect)
│   ├── policy_agent.py      # violation reasoning with gpt-oss-120b
│   ├── rag_engine.py        # ChromaDB retrieval over the guidelines
│   ├── action_agent.py      # decision matrix, audit log, review queue
│   ├── groq_client.py       # Groq API wrapper with retry/backoff
│   └── config.py            # paths + model names (overridable via .env)
├── guidelines/
│   └── moderation_guidelines.md   # the policy rulebook used for RAG
├── demo_clips/              # put local test videos here (not committed)
├── logs/                    # audit_log.jsonl / review_queue.jsonl at runtime
├── backend.py               # FastAPI server + streaming endpoints
├── sentinel_live.html       # live moderation UI
├── run.sh                   # one-command setup + launch
├── LICENSE
└── requirements.txt
```

## Run
```bash
cp .env.example .env      # paste your GROQ_API_KEY
./run.sh                  # first run creates .venv + installs deps
```
Open http://127.0.0.1:9000, pick a clip from `demo_clips/`, and press start.

Requires `ffmpeg` (`brew install ffmpeg`) and `uv`.

## CLI tests
```bash
cd agents
../.venv/bin/python orchestrator.py ../demo_clips/demo_villain_30s.mp4
../.venv/bin/python policy_agent.py        # two canned text cases
../.venv/bin/python rag_engine.py          # offline, no key needed
```

## Notes
- Every model name can be overridden in `.env` if Groq renames or retires one.
- Each frame gets one vision call (description + YES/NO concern flag together), and each segment gets one policy call. Rate limits (429) are retried with backoff.
- The `/gpu` endpoint now reports Groq API call and token counts. The UI uses it as the "online" indicator.

## License
Released under the [MIT License](LICENSE).
