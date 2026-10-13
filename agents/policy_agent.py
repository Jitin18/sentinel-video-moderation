"""
Policy Agent — reasons about content violations using gpt-oss-120b on Groq + guidelines.
Part of the Agentic Multi-modal Content Moderation system.
"""
import json
import re
import config
import groq_client  # Groq-hosted gpt-oss-120b (replaces local vLLM Qwen)

import rag_engine  # RAG: retrieves relevant policies per segment

# Load moderation guidelines once at import
with open(config.GUIDELINES_PATH) as f:
    GUIDELINES = f.read()

# Extract the STRUCTURAL sections that must ALWAYS be in the prompt
# (interplay rules, output contract, decision matrix) — these are not
# "retrievable knowledge", they're fixed instructions.
def _extract_section(text, header):
    import re
    m = re.search(r"(## " + re.escape(header) + r".*?)(?=\n## |\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""

_INTERPLAY = _extract_section(GUIDELINES, "MULTI-MODAL INTERPLAY RULES (CORE INTELLIGENCE)")
_SEVERITY = _extract_section(GUIDELINES, "SEVERITY TIERS")
_STRUCTURAL = _SEVERITY + "\n\n" + _INTERPLAY

POLICY_SYSTEM_PROMPT = """You are a content moderation Policy Agent. You evaluate content against community guidelines and return a structured verdict.

You are given:
1. The community guidelines (policy rulebook)
2. A description of what was SEEN in the video (visual)
3. A transcript of what was HEARD in the audio

Your job: determine if the content violates any policy. Pay special attention to the INTERPLAY RULES — the combination of visual and audio matters, not just each alone.

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{
  "violation": true or false,
  "policy_id": "POLICY_XX_XX" or null,
  "policy_name": "name" or null,
  "category": "category name" or null,
  "severity": "HIGH" or "MEDIUM" or "LOW" or "NONE",
  "confidence": 0.0 to 1.0,
  "interplay_rule_applied": "INTERPLAY_XX" or null,
  "reasoning": "brief explanation citing what was seen, heard, and which rule applies",
  "violating_phrase": "the EXACT verbatim words from the audio transcript that triggered the violation (copy them word-for-word so they can be matched and censored), or null if the violation is visual-only or there is no violation",
  "visual_violation": true or false,
  "audio_violation": true or false,
  "recommended_action": "auto-takedown" or "human review" or "log only" or "pass"
}

MODALITY FLAGS — set these precisely, they control what gets censored:
- "visual_violation": true ONLY if what was SEEN (the visual) independently contains a violation (e.g. a weapon, gore, explicit imagery). If the visual is neutral/benign and only the audio is harmful, set this to false.
- "audio_violation": true ONLY if what was HEARD (the audio transcript) contains a violation (e.g. threats, slurs, profanity). If the audio is benign and only the visual is harmful, set this to false.
- If both modalities independently violate, set both true. If there is no violation, set both false.

IMPORTANT for violating_phrase: copy the offending words EXACTLY as they appear in the AUDIO transcript provided — same words, same spelling. Do not paraphrase. If the violation is purely visual, set it to null."""


def evaluate_content(visual_description: str, audio_transcript: str, k: int = 5, frame_flags: str = "") -> dict:
    """Evaluate content against guidelines, return a structured verdict dict."""
    # RAG: retrieve the most relevant policies for THIS segment.
    # Query each modality separately and union the results — a long visual
    # description would otherwise drown out a short spoken threat (and vice versa).
    retrieval = {"ids": [], "docs": [], "n_total": 0}
    for query in [audio_transcript, visual_description, f"VISUAL: {visual_description}\nAUDIO: {audio_transcript}"]:
        if not query or not query.strip():
            continue
        r = rag_engine.retrieve(query, k=k)
        retrieval["n_total"] = r["n_total"]
        for pid, doc in zip(r["ids"], r["docs"]):
            if pid not in retrieval["ids"]:
                retrieval["ids"].append(pid)
                retrieval["docs"].append(doc)
    retrieved_policy_text = "\n\n".join(retrieval["docs"])

    user_prompt = f"""SEVERITY TIERS & INTERPLAY RULES:
{_STRUCTURAL}

RELEVANT POLICIES (retrieved for this content from the full rulebook of {retrieval['n_total']} policies):
{retrieved_policy_text}

CONTENT TO EVALUATE:
VISUAL (what was seen): {visual_description}
FRAME-LEVEL VISUAL SCAN: {frame_flags if frame_flags else "no specific frames flagged"}
AUDIO (what was heard): {audio_transcript}

Evaluate this content against the retrieved policies and interplay rules. If none of the retrieved policies are violated, return violation=false. Return ONLY the JSON verdict."""

    response = groq_client.chat(
        messages=[
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,  # gpt-oss is a reasoning model; leave room for its thinking
        temperature=0.1,
        json_mode=True,
    )

    raw = response.choices[0].message.content.strip()

    # Extract JSON even if wrapped in markdown fences
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {
            "violation": False, "policy_id": None, "severity": "NONE",
            "confidence": 0.0,
            "reasoning": "Failed to parse verdict: " + raw[:200],
            "recommended_action": "pass",
        }

    verdict["_tokens"] = {
        "prompt": response.usage.prompt_tokens,
        "completion": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
    }
    verdict["_retrieved_policies"] = retrieval["ids"]
    verdict["_n_policies_total"] = retrieval["n_total"]
    return verdict


if __name__ == "__main__":
    print("=== TEST 1: Neutral visual + threatening audio ===")
    v1 = evaluate_content(
        visual_description="A person sitting calmly in a room speaking to the camera",
        audio_transcript="I know where you live and I am going to come find you and hurt you",
    )
    print(json.dumps(v1, indent=2))

    print("\n=== TEST 2: Safe cooking video ===")
    v2 = evaluate_content(
        visual_description="A person chopping vegetables in a kitchen",
        audio_transcript="Today we are making a delicious vegetable stir fry, first add some oil",
    )
    print(json.dumps(v2, indent=2))