"""
Scene-aware frame sampling.

Instead of grabbing fixed-fraction frames, detect actual shot changes
within a segment and sample one frame per distinct shot — so violent
content can't slip *between* sampled frames. Each returned frame carries
its TIMESTAMP (this is what later enables precise per-frame blurring).

Robust by design: if a segment has no detectable cuts (one continuous
shot — common in short clips), fall back to even sampling. So it only
ever IMPROVES on fixed sampling, never breaks.
"""
import cv2
from PIL import Image

# cache full scene list per video so we detect ONCE, not per segment
_scene_cache = {}


def _all_scenes(video_path, threshold=27.0):
    """Detect (and cache) the full scene list for a video. Returns list of
    (start_sec, end_sec) tuples, or None if detection fails."""
    if video_path in _scene_cache:
        return _scene_cache[video_path]
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.detect_scenes(video, show_progress=False)
        scenes = [(s.seconds, e.seconds) for (s, e) in sm.get_scene_list()]
        _scene_cache[video_path] = scenes
        return scenes
    except Exception:
        _scene_cache[video_path] = None
        return None


def _even_timestamps(start_sec, end_sec, n):
    """Fallback: evenly spaced timestamps across the segment."""
    if n == 1:
        return [start_sec + (end_sec - start_sec) * 0.5]
    fracs = [0.3, 0.7] if n == 2 else [i / (n - 1) for i in range(n)]
    return [start_sec + (end_sec - start_sec) * f for f in fracs]


def detect_scene_times(video_path, start_sec, end_sec, threshold=27.0, max_scenes=4):
    """
    Return a list of timestamps (seconds) to sample — one per detected
    shot within [start_sec, end_sec]. Falls back to even sampling if no
    cuts are found or detection fails.
    """
    scenes = _all_scenes(video_path, threshold)
    if scenes is None:
        # detection failed → safe fallback
        return _even_timestamps(start_sec, end_sec, 2), "even_fallback(detect_failed)"

    # keep scene midpoints that overlap our segment window [start_sec, end_sec]
    times = []
    for (s_sec, e_sec) in scenes:
        ov_s = max(s_sec, start_sec)
        ov_e = min(e_sec, end_sec)
        if ov_e > ov_s:
            times.append(round((ov_s + ov_e) / 2.0, 2))

    if not times:
        # no cuts inside this window → even sampling
        return _even_timestamps(start_sec, end_sec, 2), "even_fallback"

    # cap and de-duplicate (keep spread)
    times = sorted(set(times))
    if len(times) > max_scenes:
        step = len(times) / max_scenes
        times = [times[int(i * step)] for i in range(max_scenes)]
    return times, "scene_aware"


def extract_frames_scene_aware(video_path, start_sec, end_sec, threshold=27.0, max_scenes=4):
    """
    Returns (frames, frame_times, mode):
      frames: list of PIL.Image
      frame_times: list of timestamps (seconds) matching each frame
      mode: 'scene_aware' or 'even_fallback...' (for telemetry/UI)
    """
    times, mode = detect_scene_times(video_path, start_sec, end_sec, threshold, max_scenes)
    cap = cv2.VideoCapture(video_path)
    frames, kept_times = [], []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            kept_times.append(round(t, 2))
    cap.release()

    # safety: if nothing read, hard fallback to one middle frame
    if not frames:
        cap = cv2.VideoCapture(video_path)
        t = start_sec + (end_sec - start_sec) * 0.5
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            kept_times.append(round(t, 2))
        cap.release()
        mode = "even_fallback(no_read)"

    return frames, kept_times, mode


if __name__ == "__main__":
    import os, sys
    clip = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_clips", "demo_villain_30s.mp4")
    print(f"Testing scene detection on {clip}\n")
    for (s, e) in [(0, 10), (10, 20), (20, 28)]:
        frames, times, mode = extract_frames_scene_aware(clip, s, e)
        print(f"[{s}-{e}s] mode={mode} | {len(frames)} frames @ times={times}")