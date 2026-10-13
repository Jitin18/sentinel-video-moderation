"""
FastAPI Backend — wraps the LangGraph moderation orchestrator.
All models run on Groq: gpt-oss-120b (policy), Qwen3.8 27B (vision), Whisper large-v3 (audio).

Endpoints:
  GET  /health        - readiness check
  GET  /clips         - list available demo clips
  POST /moderate      - batch: full video, all results at once
  GET  /stream        - live: SSE, emits each segment as processed
  GET  /gpu           - Groq API usage stats (kept at /gpu for the UI)
"""
import sys, os, json, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

print("[Backend] Importing orchestrator (Groq-backed agents + RAG index)...")
import config
import groq_client
import perception_agent as perception
import orchestrator as orch
print("[Backend] Orchestrator ready!")

app = FastAPI(title="Agentic Content Moderation API")

# Allow the browser UI to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIPS_DIR = config.CLIPS_DIR
UPLOAD_DIR = config.UPLOAD_DIR


# ---------- Models ----------
class ModerateRequest(BaseModel):
    video_path: str
    segment_len: int = 10


# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "models": f"{config.POLICY_MODEL} + {config.VISION_MODEL} + {config.WHISPER_MODEL} (Groq)"}


# ---------- List clips ----------
@app.get("/clips")
def list_clips():
    clips = []
    for d in [CLIPS_DIR, UPLOAD_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith((".mp4", ".mov", ".avi", ".mkv")):
                    full = os.path.join(d, f)
                    clips.append({
                        "name": f,
                        "path": full,
                        "size_mb": round(os.path.getsize(full) / 1e6, 1),
                        "duration_sec": round(perception.get_video_duration(full), 1),
                    })
    return {"clips": clips}


# ---------- Upload ----------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    dest = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"path": dest, "name": file.filename,
            "duration_sec": round(perception.get_video_duration(dest), 1)}


# ---------- Batch moderation ----------
@app.post("/moderate")
def moderate(req: ModerateRequest):
    if not os.path.exists(req.video_path):
        return JSONResponse(status_code=404, content={"error": "video not found"})
    result = orch.moderate_video(req.video_path, req.segment_len)
    return result


# ---------- Live streaming moderation (SSE) ----------
@app.get("/stream")
async def stream(video_path: str, segment_len: int = 10):
    """Server-Sent Events: emit each segment result as it's processed."""
    if not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "video not found"})

    async def event_generator():
        segments = perception.segment_video(video_path, segment_len)
        duration = perception.get_video_duration(video_path)

        # Initial event: metadata
        yield _sse("start", {
            "video": os.path.basename(video_path),
            "duration_sec": round(duration, 1),
            "total_segments": len(segments),
        })

        t0 = time.time()
        flagged = 0
        for idx, (s, e) in enumerate(segments):
            # Run one segment (blocking call in thread so we don't block the event loop)
            result = await asyncio.to_thread(orch.moderate_segment, video_path, s, e)
            if result["verdict"]["violation"]:
                flagged += 1

            yield _sse("segment", {
                "index": idx,
                "start_sec": result["start_sec"],
                "end_sec": result["end_sec"],
                "visual": result["visual"],
                "audio": result["audio"],
                "words": result.get("words", []),
                "frame_times": result.get("frame_times", []),
                "scene_mode": result.get("scene_mode", ""),
                "per_frame": result.get("per_frame", []),
                "flagged_frame_times": result.get("flagged_frame_times", []),
                "verdict": result["verdict"],
                "decision": result["decision"],
                "recheck_count": result["recheck_count"],
                "metrics": result["metrics"],
            })

        # Final event: summary
        yield _sse("done", {
            "total_segments": len(segments),
            "flagged_segments": flagged,
            "total_latency_sec": round(time.time() - t0, 2),
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------- Groq usage stats ----------
@app.get("/gpu")
def gpu_stats():
    """Groq API usage. The UI polls this to show the backend is online;
    vram_pct is repurposed as a tiny activity meter (calls in this session, capped at 100)."""
    u = groq_client.usage
    return {
        "provider": "groq",
        "calls": u["calls"],
        "tokens": u["tokens"],
        "rate_limited": u["rate_limited"],
        "vram_pct": min(100, u["calls"]),
        "vram_used_gb": 0,
        "vram_total_gb": 0,
    }


@app.get("/video")
def serve_video(path: str):
    """Serve a video file for playback in the UI."""
    if os.path.exists(path) and path.endswith((".mp4",".mov",".avi",".mkv",".webm")):
        return FileResponse(path, media_type="video/mp4")
    return JSONResponse(status_code=404, content={"error":"video not found"})


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if os.path.exists(config.UI_PATH):
        with open(config.UI_PATH) as f:
            return f.read()
    return f"<h1>UI not found at {config.UI_PATH}</h1>"


if __name__ == "__main__":
    import uvicorn
    print("[Backend] Starting on port 9000...")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 9000)))
