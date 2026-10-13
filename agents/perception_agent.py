"""
Perception Agent — unified multimodal perception engine (Groq-powered).
Vision: Qwen3.8 27B on Groq (replaces local LLaVA).
Audio:  Whisper large-v3 on Groq (replaces local Whisper).
Transcribes each video ONCE, processes segments efficiently.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import scene_detect
import groq_client
from PIL import Image

print("[Perception] Using Groq for vision + Whisper (no local models to load).")

# Cache for transcripts (transcribe each video only once)
_transcript_cache = {}

DESCRIBE_PROMPT = ("Describe what is happening in this image, including people, actions, objects, "
                   "visible text, and any potentially concerning or harmful content. Keep it under 80 words.")


def transcribe_full(video_path: str):
    """Transcribe the entire video ONCE, cache the segmented result."""
    if video_path in _transcript_cache:
        return _transcript_cache[video_path]
    t0 = time.time()
    result = groq_client.transcribe(video_path)
    elapsed = time.time() - t0

    # Groq returns words at the top level — nest them under their segment
    # (same shape the local Whisper produced, so the rest of the code is unchanged)
    words = result["words"]
    segments = []
    for s in result["segments"]:
        seg_words = [w for w in words if w["start"] >= s["start"] - 0.05 and w["start"] < s["end"]]
        segments.append({"start": s["start"], "end": s["end"], "text": s["text"], "words": seg_words})
    if not segments and words:
        segments = [{"start": words[0]["start"], "end": words[-1]["end"],
                     "text": result["text"], "words": words}]

    data = {
        "full_text": result["text"],
        "segments": segments,
        "transcribe_latency": round(elapsed, 2),
    }
    _transcript_cache[video_path] = data
    return data


def get_audio_for_segment(video_path: str, start_sec: float, end_sec: float):
    """Slice cached transcript to a time segment."""
    data = transcribe_full(video_path)
    segs = [s["text"] for s in data["segments"]
            if s["start"] < end_sec and s["end"] > start_sec]
    return " ".join(segs).strip()


def get_words_for_segment(video_path: str, start_sec: float, end_sec: float):
    """Return word-level timestamps that fall within this segment's time-range."""
    data = transcribe_full(video_path)
    words = []
    for s in data["segments"]:
        for w in s.get("words", []):
            if w["start"] < end_sec and w["end"] > start_sec:
                words.append({"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)})
    return words


def extract_frames(video_path: str, start_sec: float, end_sec: float, n_frames: int = 2):
    """Extract n representative frames from a segment."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fracs = [0.3, 0.7] if n_frames == 2 else [0.2, 0.5, 0.8][:n_frames]
    for frac in fracs:
        t = start_sec + (end_sec - start_sec) * frac
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def describe_frames(frames, prompt=DESCRIBE_PROMPT):
    """Run the vision model on frames, return combined description."""
    return " | ".join(groq_client.vision(img, prompt, max_tokens=150) for img in frames)


def perceive_segment(video_path: str, start_sec: float, end_sec: float):
    """Full perception for one segment: visual + audio."""
    t0 = time.time()
    frames = extract_frames(video_path, start_sec, end_sec)
    visual = describe_frames(frames)
    audio = get_audio_for_segment(video_path, start_sec, end_sec)
    return {
        "segment": f"{start_sec:.0f}-{end_sec:.0f}s",
        "start": start_sec,
        "end": end_sec,
        "visual": visual,
        "audio": audio,
        "latency_sec": round(time.time() - t0, 2),
    }


_FLAGGED_PROMPT = DESCRIBE_PROMPT + (
    "\n\nThen, on a final separate line, write exactly 'CONCERNING: YES' if the image contains any "
    "violent, weapon, gore, explicit, or harmful content, otherwise 'CONCERNING: NO'.")


def _describe_and_flag(img):
    """One vision call -> (description, concerning flag)."""
    resp = groq_client.vision(img, _FLAGGED_PROMPT, max_tokens=180)
    m = re.search(r"CONCERNING:\s*(YES|NO)", resp, re.IGNORECASE)
    concerning = bool(m) and m.group(1).upper() == "YES"
    desc = re.sub(r"\**CONCERNING:\s*\**\s*(YES|NO)\**", "", resp, flags=re.IGNORECASE).strip()
    return desc, concerning


def describe_frames_flagged(frames, frame_times):
    """Per-frame description + concerning flag. Returns (combined_text, per_frame list).
    per_frame = [{"time": t, "desc": d, "concerning": bool}, ...]"""
    # frames are independent API calls — run them in parallel (Groq latency varies a lot per call)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_describe_and_flag, frames))
    per_frame = [{"time": round(t, 2), "desc": d, "concerning": c}
                 for (d, c), t in zip(results, frame_times)]
    return " | ".join(d for d, _ in results), per_frame


def perceive_visual_scene_aware(video_path: str, start_sec: float, end_sec: float, max_scenes: int = 4):
    """Scene-aware visual perception with per-frame flagging.
    Returns (visual_text, frame_times, scene_mode, per_frame, flagged_times)."""
    frames, frame_times, mode = scene_detect.extract_frames_scene_aware(
        video_path, start_sec, end_sec, max_scenes=max_scenes)
    visual, per_frame = describe_frames_flagged(frames, frame_times)
    flagged_times = [f["time"] for f in per_frame if f["concerning"]]
    return visual, frame_times, mode, per_frame, flagged_times


def get_video_duration(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps else 0


def segment_video(video_path: str, segment_len: int = 10):
    """Yield (start, end) tuples for the whole video."""
    dur = get_video_duration(video_path)
    segs = []
    s = 0
    while s < dur:
        segs.append((s, min(s + segment_len, dur)))
        s += segment_len
    return segs


if __name__ == "__main__":
    import os, sys, json
    import config
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.CLIPS_DIR, "demo_tedtalk_40s.mp4")
    dur = get_video_duration(path)
    print(f"Video: {path}")
    print(f"Duration: {dur:.1f}s")
    print(f"Segments: {segment_video(path)}")
    print("\nPre-transcribing full video (once)...")
    t = transcribe_full(path)
    print(f"Transcribe latency: {t['transcribe_latency']}s")
    print(f"Full transcript: {t['full_text'][:200]}")
    print("\nPerceiving segment 0-10s...")
    out = perceive_segment(path, 0, 10)
    print(json.dumps(out, indent=2))
