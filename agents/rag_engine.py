"""
RAG Engine — retrieves the most relevant policies for a given segment.

Instead of stuffing the entire rulebook into every policy-agent call, we:
  1. chunk the rulebook into individual policies + interplay rules
  2. embed each chunk with a local sentence-transformers model (MiniLM)
  3. store vectors in ChromaDB (in-memory; rebuilt each session, fast)
  4. retrieve(query, k) -> the top-k most semantically relevant policy chunks

The interplay rules, output contract, and decision matrix are STRUCTURAL
(always needed) and are NOT retrieved — they're kept fixed in the prompt.
Only POLICY definitions are RAG-driven.
"""
import re
import time

from config import GUIDELINES_PATH as _GUIDELINES_PATH

_model = None
_collection = None
_n_policies = 0


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("[RAG] Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=_device())
        print("[RAG] Embedding model loaded.")
    return _model


def _chunk_rulebook(md_text: str):
    """Split the rulebook into one chunk per POLICY_XX_XX block."""
    chunks = []
    # Each policy starts with '### POLICY_' ; capture until the next ### or ##
    pattern = re.compile(r"(### (POLICY_[A-Z]+_\d+) — .*?)(?=\n### |\n## |\Z)", re.DOTALL)
    for m in pattern.finditer(md_text):
        block = m.group(1).strip()
        pid = m.group(2)
        # pull the human name from the header line "### POLICY_XX — Name"
        name_match = re.match(r"### POLICY_[A-Z]+_\d+ — (.*)", block)
        name = name_match.group(1).strip() if name_match else pid
        chunks.append({"id": pid, "name": name, "text": block})
    return chunks


def build_index():
    """Chunk + embed + store. Called once at import. Fast (~1-2s)."""
    global _collection, _n_policies
    import chromadb

    t0 = time.time()
    with open(_GUIDELINES_PATH) as f:
        md = f.read()
    chunks = _chunk_rulebook(md)
    _n_policies = len(chunks)

    model = _load_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    # in-memory chroma client (rebuilt each session; cheap)
    client = chromadb.Client()
    # fresh collection each build
    try:
        client.delete_collection("policies")
    except Exception:
        pass
    _collection = client.create_collection("policies", metadata={"hnsw:space": "cosine"})
    _collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"id": c["id"], "name": c["name"]} for c in chunks],
    )
    print(f"[RAG] Indexed {_n_policies} policies in {round(time.time()-t0,2)}s.")
    return _n_policies


def retrieve(query: str, k: int = 4):
    """Return the top-k relevant policy chunks for a query string.
    Returns: {"ids": [...], "docs": [...], "names": [...], "n_total": int}
    """
    if _collection is None:
        build_index()
    model = _load_model()
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = _collection.query(query_embeddings=q_emb, n_results=min(k, _n_policies))
    ids = res["ids"][0]
    docs = res["documents"][0]
    names = [m["name"] for m in res["metadatas"][0]]
    return {"ids": ids, "docs": docs, "names": names, "n_total": _n_policies}


# build at import so it's ready
build_index()

if __name__ == "__main__":
    print("\n=== TEST: gun + threat segment ===")
    r = retrieve("two men, one holding a gun. Audio: I want him dead, I want his family dead", k=4)
    print(f"Retrieved {len(r['ids'])} of {r['n_total']} policies:")
    for i, n in zip(r["ids"], r["names"]):
        print(f"  - {i}: {n}")

    print("\n=== TEST: safe cooking segment ===")
    r = retrieve("person chopping vegetables in a kitchen. Audio: today we make a vegetable stir fry", k=4)
    print(f"Retrieved {len(r['ids'])} of {r['n_total']} policies:")
    for i, n in zip(r["ids"], r["names"]):
        print(f"  - {i}: {n}")