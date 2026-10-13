"""
Orchestrator — connects Perception, Policy, Action agents as a LangGraph state machine.
Includes the Policy->Perception recheck feedback loop and full metrics capture.
"""
import os, sys, time, json
from typing import TypedDict, Optional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import perception_agent as perception
import policy_agent as policy
import action_agent as action
from langgraph.graph import StateGraph, END


# ---------- Shared State ----------
class ModerationState(TypedDict):
    video_path: str
    start_sec: float
    end_sec: float
    visual: Optional[str]
    audio: Optional[str]
    words: list
    frame_times: list
    scene_mode: str
    per_frame: list
    flagged_frame_times: list
    verdict: Optional[dict]
    decision: Optional[dict]
    recheck_count: int
    metrics: dict


# ---------- Nodes ----------
def perception_node(state: ModerationState) -> ModerationState:
    t0 = time.time()
    # Scene-aware sampling: detect distinct shots, sample one frame each
    # (more frames on recheck for a closer look)
    max_scenes = 6 if state["recheck_count"] > 0 else 4
    visual, frame_times, scene_mode, per_frame, flagged_times = perception.perceive_visual_scene_aware(
        state["video_path"], state["start_sec"], state["end_sec"], max_scenes=max_scenes)
    state["visual"] = visual
    state["frame_times"] = frame_times
    state["scene_mode"] = scene_mode
    state["per_frame"] = per_frame
    state["flagged_frame_times"] = flagged_times
    state["audio"] = perception.get_audio_for_segment(state["video_path"], state["start_sec"], state["end_sec"])
    state["words"] = perception.get_words_for_segment(state["video_path"], state["start_sec"], state["end_sec"])
    state["metrics"]["perception_sec"] = round(time.time() - t0, 2)
    return state


def policy_node(state: ModerationState) -> ModerationState:
    t0 = time.time()
    # summarize per-frame concerning flags for the policy agent
    _pf = state.get("per_frame", []) or []
    _concerning = [f for f in _pf if f.get("concerning")]
    if _concerning:
        _frame_note = "An automated per-frame visual scan flagged these frame(s) as potentially concerning: " + \
            "; ".join([f"at {f['time']}s ({f['desc'][:80]})" for f in _concerning]) + \
            ". Consider whether the VISUAL itself independently violates a policy."
    else:
        _frame_note = ""
    verdict = policy.evaluate_content(state["visual"], state["audio"], frame_flags=_frame_note)
    state["verdict"] = verdict
    state["metrics"]["policy_sec"] = round(time.time() - t0, 2)
    state["metrics"]["tokens"] = verdict.get("_tokens", {})
    return state


def action_node(state: ModerationState) -> ModerationState:
    seg_info = {"segment": f"{state['start_sec']:.0f}-{state['end_sec']:.0f}s", "video": state["video_path"]}
    state["decision"] = action.execute_action(seg_info, state["verdict"])
    return state


# ---------- Conditional edge: recheck loop ----------
def should_recheck(state: ModerationState) -> str:
    decision = state["decision"]
    if decision.get("needs_recheck") and state["recheck_count"] < 2:
        state["recheck_count"] += 1
        return "recheck"
    return "done"


# ---------- Build the graph ----------
def build_graph():
    g = StateGraph(ModerationState)
    g.add_node("perception", perception_node)
    g.add_node("policy", policy_node)
    g.add_node("action", action_node)

    g.set_entry_point("perception")
    g.add_edge("perception", "policy")
    g.add_edge("policy", "action")
    g.add_conditional_edges("action", should_recheck, {
        "recheck": "perception",  # loop back for closer look
        "done": END,
    })
    return g.compile()


_graph = build_graph()


def moderate_segment(video_path: str, start_sec: float, end_sec: float) -> dict:
    """Run one segment through the full agent graph."""
    initial = {
        "video_path": video_path, "start_sec": start_sec, "end_sec": end_sec,
        "visual": None, "audio": None, "words": [], "frame_times": [], "scene_mode": "", "per_frame": [], "flagged_frame_times": [], "verdict": None, "decision": None,
        "recheck_count": 0, "metrics": {},
    }
    t0 = time.time()
    final = _graph.invoke(initial)
    final["metrics"]["total_sec"] = round(time.time() - t0, 2)
    return final


def moderate_video(video_path: str, segment_len: int = 10) -> dict:
    """Run the full video through the pipeline."""
    segments = perception.segment_video(video_path, segment_len)
    results = []
    t0 = time.time()
    for (s, e) in segments:
        results.append(moderate_segment(video_path, s, e))
    total = round(time.time() - t0, 2)

    flagged = [r for r in results if r["verdict"]["violation"]]
    return {
        "video": video_path,
        "total_segments": len(segments),
        "flagged_segments": len(flagged),
        "total_latency_sec": total,
        "results": results,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_clips", "demo_villain_30s.mp4")
    print(f"=== AGENTIC MODERATION (LangGraph): {path} ===\n")
    out = moderate_video(path)

    for r in out["results"]:
        v = r["verdict"]
        d = r["decision"]
        m = r["metrics"]
        print(f"[{r['start_sec']:.0f}-{r['end_sec']:.0f}s] "
              f"{v['severity']:6} conf={v.get('confidence')} -> {d['action']:20} "
              f"| rechecks={r['recheck_count']} | {m.get('total_sec')}s")
        if v["violation"]:
            print(f"         {v.get('policy_id')}: {v.get('reasoning','')[:90]}")

    print(f"\n=== {out['flagged_segments']}/{out['total_segments']} flagged "
          f"in {out['total_latency_sec']}s ===")
