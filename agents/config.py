"""
Central config — paths + Groq model names.
All paths are relative to the project folder so it runs anywhere.
Model names can be overridden via environment variables / .env.
"""
import os

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(AGENTS_DIR)

# load .env from the project folder if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_DIR, ".env"))
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq models (override in .env if Groq renames/deprecates one)
POLICY_MODEL = os.environ.get("GROQ_POLICY_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")

GUIDELINES_PATH = os.path.join(PROJECT_DIR, "guidelines", "moderation_guidelines.md")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
AUDIT_LOG_PATH = os.path.join(LOGS_DIR, "audit_log.jsonl")
REVIEW_QUEUE_PATH = os.path.join(LOGS_DIR, "review_queue.jsonl")
CLIPS_DIR = os.path.join(PROJECT_DIR, "demo_clips")
UPLOAD_DIR = os.path.join(PROJECT_DIR, "dataset", "uploads")
UI_PATH = os.path.join(PROJECT_DIR, "sentinel_live.html")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
