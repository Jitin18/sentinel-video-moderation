# Community Content Moderation Guidelines — Policy Rulebook v1.0

## SEVERITY TIERS
- HIGH: Imminent harm/illegal. Action: auto-takedown + audit log.
- MEDIUM: Clear violation, no imminent harm. Action: flag for human review.
- LOW: Borderline/minor. Action: log only.
- NONE: Compliant. Action: pass.

## CATEGORY 1 — HATE SPEECH & DISCRIMINATION

### POLICY_HS_01 — Direct Hate Speech
Severity: HIGH
Definition: Attacks/dehumanizes/promotes hatred against individuals or groups based on protected attributes (race, ethnicity, religion, gender, sexual orientation, disability, nationality).
Visual: hate symbols, discriminatory gestures. Audio: slurs, dehumanizing language, calls for exclusion.
Example: Spoken slur at a protected group -> VIOLATION (HIGH). Visual hate symbol -> VIOLATION (HIGH).
Action: Auto-takedown.

### POLICY_HS_02 — Implicit / Coded Hate
Severity: MEDIUM
Definition: Hate via dog whistles, coded language, or context-dependent imagery targeting protected groups indirectly.
Note: requires visual+audio interplay. Neutral words + hateful imagery may qualify.
Action: Flag for human review.

## CATEGORY 2 — VIOLENCE & THREATS

### POLICY_VL_01 — Graphic Violence
Severity: HIGH
Definition: Real-world violence, gore, or physical harm to shock or glorify.
Visual: blood, weapons against people, injury/death. Audio: sounds of assault/distress.
Note: clearly-labeled fictional/SFX violence is context-dependent (LOW/NONE).
Action: Auto-takedown.

### POLICY_VL_02 — Credible Threats
Severity: HIGH
Definition: Statements expressing intent to harm a specific person or group.
Note: audio interplay critical — neutral visuals + threatening speech still qualifies.
Example: Spoken threat to harm a named individual -> VIOLATION (HIGH).
Action: Auto-takedown + escalate.

### POLICY_VL_03 — Incitement to Violence
Severity: HIGH
Definition: Content encouraging/instructing others to commit violence.
Action: Auto-takedown.

## CATEGORY 3 — HARASSMENT & BULLYING

### POLICY_HR_01 — Targeted Harassment
Severity: MEDIUM
Definition: Repeated/severe attacks to degrade or intimidate a specific individual.
Action: Flag for human review.

### POLICY_HR_02 — Doxxing / Privacy Violation
Severity: HIGH
Definition: Sharing private/identifying info (address, phone, workplace) without consent.
Visual: on-screen documents, IDs, addresses, private screenshots.
Action: Auto-takedown.

## CATEGORY 4 — ADULT & SEXUAL CONTENT

### POLICY_AD_01 — Explicit Sexual Content
Severity: HIGH
Definition: Pornographic or sexually explicit material.
Action: Auto-takedown.

### POLICY_AD_02 — Suggestive Content
Severity: LOW
Definition: Sexually suggestive but non-explicit, may be inappropriate for general audiences.
Action: Log only / age-gate recommendation.

## CATEGORY 5 — DANGEROUS & HARMFUL ACTS

### POLICY_DG_01 — Self-Harm Promotion
Severity: HIGH
Definition: Promotes/glorifies/instructs self-harm or suicide.
Note: supportive educational/recovery content is NOT a violation.
Action: Auto-takedown + surface support resources.

### POLICY_DG_02 — Dangerous Challenges / Instructions
Severity: MEDIUM
Definition: Depicts/instructs dangerous activities likely to cause injury.
Action: Flag for human review.

## CATEGORY 6 — MISINFORMATION & REGULATED GOODS

### POLICY_MI_01 — Harmful Misinformation
Severity: MEDIUM
Definition: Demonstrably false claims risking public harm (health, safety, electoral).
Action: Flag for human review.

### POLICY_RG_01 — Regulated Goods
Severity: HIGH
Definition: Promotion/sale of illegal drugs, weapons, or regulated items.
Visual: drugs, firearms for sale, contraband.
Action: Auto-takedown.

## CATEGORY 7 — PROFANITY & SPAM (LOW TIER)

### POLICY_PF_01 — Excessive Profanity
Severity: LOW
Action: Log only.

### POLICY_SP_01 — Spam / Misleading
Severity: LOW
Definition: Repetitive, deceptive, or scam content.
Action: Log only.

## MULTI-MODAL INTERPLAY RULES (CORE INTELLIGENCE)
INTERPLAY_01 — Neutral visual + harmful audio: evaluate under the relevant audio policy at FULL severity. Benign visuals do NOT reduce severity.
INTERPLAY_02 — Harmful visual + neutral audio: evaluate under the relevant visual policy. Audio neutrality does NOT excuse the visual.
INTERPLAY_03 — Contextual reframing: if audio reframes visuals as educational/news/documentary/fiction, downgrade severity ONE tier, UNLESS HIGH-tier illegal (real graphic violence, credible threats, CSAM — never downgraded).
INTERPLAY_04 — Sarcasm/satire: if tone/context indicates satire not endorsement, downgrade ONE tier for MEDIUM/LOW only. HIGH never downgraded on satire grounds.

## POLICY AGENT OUTPUT CONTRACT (RETURN EXACTLY THIS JSON)
{
  "violation": true,
  "policy_id": "POLICY_VL_02",
  "policy_name": "Credible Threats",
  "category": "Violence & Threats",
  "severity": "HIGH",
  "confidence": 0.92,
  "interplay_rule_applied": "INTERPLAY_01",
  "reasoning": "Brief explanation citing what was seen, heard, and which rule applies.",
  "evidence": {"visual": "...", "audio": "...", "timestamp": "MM:SS-MM:SS"},
  "recommended_action": "auto-takedown"
}
If no violation: {"violation": false, "policy_id": null, "severity": "NONE", "confidence": 0.0, "reasoning": "...", "recommended_action": "pass"}

## ACTION DECISION MATRIX
HIGH + confidence >= 0.75 -> auto-takedown + audit log + escalate
HIGH + confidence < 0.75 -> urgent human review (set needs_recheck)
MEDIUM + confidence >= 0.70 -> human review queue
MEDIUM + confidence < 0.70 -> log + low-priority review (set needs_recheck)
LOW (any) -> log only
NONE (any) -> pass
