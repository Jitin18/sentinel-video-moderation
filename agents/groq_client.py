"""
Groq client — replaces the local vLLM (Qwen), LLaVA and Whisper models.
  chat()        -> policy reasoning (gpt-oss-120b)
  vision()      -> frame description (Qwen3.8 27B, multimodal)
  transcribe()  -> Whisper large-v3 with word-level timestamps
Retries on rate limits (429) so the free tier doesn't kill a demo run.
"""
import base64
import io
import os
import subprocess
import tempfile
import time

import config
from groq import Groq, RateLimitError, APIConnectionError, InternalServerError

if not config.GROQ_API_KEY or "your_key_here" in config.GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Put it in project/.env (see .env.example).")

client = Groq(api_key=config.GROQ_API_KEY)

# simple usage counters (shown by the backend /gpu -> /usage endpoint)
usage = {"calls": 0, "tokens": 0, "rate_limited": 0}


def _with_retry(fn, tries=5):
    delay = 2.0
    for attempt in range(tries):
        try:
            usage["calls"] += 1
            return fn()
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            if isinstance(e, RateLimitError):
                usage["rate_limited"] += 1
            if attempt == tries - 1:
                raise
            print(f"[Groq] {type(e).__name__}, retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2


def chat(messages, model=None, max_tokens=512, temperature=0.1, json_mode=False):
    """Returns the raw completion response (so callers can read .usage)."""
    kwargs = dict(model=model or config.POLICY_MODEL, messages=messages,
                  max_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _with_retry(lambda: client.chat.completions.create(**kwargs))
    if resp.usage:
        usage["tokens"] += resp.usage.total_tokens
    return resp


def _image_to_data_url(img, max_side=768):
    """PIL image -> base64 JPEG data URL (downscaled to keep requests small)."""
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def vision(img, prompt, max_tokens=200):
    """Describe one PIL image with the Groq vision model. Returns text."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _image_to_data_url(img)}},
    ]}]
    resp = chat(messages, model=config.VISION_MODEL, max_tokens=max_tokens, temperature=0.0)
    return resp.choices[0].message.content.strip()


def _extract_audio(video_path):
    """Extract mono 16kHz audio with ffmpeg (keeps upload well under Groq's 25MB limit).
    Returns a temp file path, or None if ffmpeg is missing / video has no audio."""
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
             "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", out],
            capture_output=True, timeout=120)
        if r.returncode == 0 and os.path.getsize(out) > 0:
            return out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    os.unlink(out)
    return None


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def transcribe(video_path):
    """Whisper via Groq. Returns {"text", "segments": [{start,end,text}], "words": [{word,start,end}]}."""
    audio_path = _extract_audio(video_path)
    if audio_path is None:
        # no ffmpeg or no audio track — try sending the video file directly
        audio_path, cleanup = video_path, False
    else:
        cleanup = True
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
        resp = _with_retry(lambda: client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), data),
            model=config.WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0.0,
        ))
    except Exception as e:
        print(f"[Groq] Transcription failed ({e}); treating as silent.")
        return {"text": "", "segments": [], "words": []}
    finally:
        if cleanup:
            os.unlink(audio_path)

    segments = [{"start": float(_get(s, "start")), "end": float(_get(s, "end")),
                 "text": str(_get(s, "text", "")).strip()}
                for s in (_get(resp, "segments") or [])]
    words = [{"word": str(_get(w, "word", "")).strip(), "start": float(_get(w, "start")),
              "end": float(_get(w, "end"))}
             for w in (_get(resp, "words") or [])]
    return {"text": str(_get(resp, "text", "")).strip(), "segments": segments, "words": words}
