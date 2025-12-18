import re
import numpy as np
from typing import List, Dict, Tuple
from llm_client import chat as llm_chat
from prompts import (
    BASE_SYSTEM_INSTRUCTION,
    USER_PROMPT_TEMPLATE,
    LANG_RULE_EN,
    LANG_RULE_URDU,
)


def is_urdu_text(text: str) -> bool:
    """Detect Urdu via Unicode range 0600–06FF."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def strip_markdown_for_preview(text: str) -> str:
    """
    Clean text for display in previews:
    - remove image markdown: ![alt](url)
    - convert link markdown [text](url) -> text
    - collapse extra whitespace
    """
    # Remove image markdown: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Replace links [text](url) with just "text"
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse whitespace/newlines
    text = " ".join(text.split())
    return text

def retrieve(
    query: str,
    embed_model,
    collection,
    k: int = 5,
) -> List[Dict]:
    """
    Retrieve top-k chunks from Chroma for a given query.
    """
    # 1) Embed query using the same model as ingestion (BGE-M3)
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")  # Chroma expects float32

    # 2) Query Chroma using query_embeddings (NOT query_texts, because we pre-embedded docs)
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
    )

    docs = results.get("documents", [[]])[0]    # list[str]
    metas = results.get("metadatas", [[]])[0]   # list[dict]
    dists = results.get("distances", [[]])[0]   # list[float] (similarity metric)

    retrieved = []
    for doc, meta, dist in zip(docs, metas, dists):

        # Build a clean preview only (do not modify the stored content)
        clean = strip_markdown_for_preview(doc)
        preview = clean[:200]
        if len(clean) > 200:
            preview += "…"
        
        retrieved.append(
            {
                "content": doc,
                "title": meta.get("title", ""),
                "source_url": meta.get("source_url"),
                "path": meta.get("path"),
                "scraped_at": meta.get("scraped_at"),
                "score": float(dist),
                "text_preview": preview,
            }
        )
    return retrieved


def build_context_for_prompt(
    retrieval_results: List[Dict],
    all_chunks: List[str],
) -> str:
    """
    Build a long context string from retrieved chunks, tagged by source.
    """
    blocks = []
    for i, r in enumerate(retrieval_results, start=1):
        idx = r["chunk_index"]
        lines = [f"Source {i}: {r['article_title']}"]
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
        lines.append(all_chunks[idx])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def answer_question(
    query: str,
    embed_model,
    collection,
    chat_history: List[Dict] | None = None,
    k: int = 5,
) -> Tuple[str, List[Dict]]:
    """
    End-to-end RAG answer: retrieve from Chroma and call the LLM.
    """
    # 1) Retrieve relevant chunks
    retrieved = retrieve(query, embed_model, collection, k=k)

    # 2) Build context text from retrieved chunks
    context_parts = []
    for item in retrieved:
        title = item.get("title") or "Source"
        source_url = item.get("source_url")
        header = f"### {title}"
        if source_url:
            header += f" ({source_url})"
        context_parts.append(f"{header}\n\n{item['content']}")

    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context retrieved."

    # 3) Language rule: detect Urdu vs English
    lang_rule = LANG_RULE_URDU if is_urdu_text(query) else LANG_RULE_EN

    system_msg = {
        "role": "system",
        "content": BASE_SYSTEM_INSTRUCTION.format(language_rule=lang_rule),
    }

    messages: List[Dict] = [system_msg]

    # optionally add filtered chat_history here...

    user_message = {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
            context=context_text,
            question=query,
        ),
    }
    messages.append(user_message)

    # 4) Call LLM with error handling
    try:
        reply = llm_chat(messages)
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {str(e)}")

    return reply, retrieved
