"""Prompt construction: the same context bundle feeds every generator backend."""

from __future__ import annotations

from typing import List, Sequence

from .models import REFUSAL_PREFIX, Prompt, RetrievedChunk

SYSTEM_PROMPT = f"""You are a customer support assistant. Answer using ONLY the context blocks provided.

Rules:
1. Every sentence of your answer must end with the ID of the block that supports it, in square brackets, e.g. [doc#section-0].
2. Cite only IDs that appear in the context. Never invent IDs, facts, numbers or policies.
3. If the context does not fully answer the question, reply with exactly this first sentence and nothing else:
   "{REFUSAL_PREFIX}"
4. Keep the answer short and factual; quote figures exactly as written in the context."""


def format_context_block(item: RetrievedChunk) -> str:
    chunk = item.chunk
    return f"[{chunk.chunk_id}] ({chunk.source} > {chunk.section})\n{chunk.text}"


def build_prompt(question: str, retrieved: Sequence[RetrievedChunk], max_context_chars: int = 6000) -> Prompt:
    """Keep above-threshold chunks in rank order until the context budget is spent."""
    included: List[RetrievedChunk] = []
    blocks: List[str] = []
    used = 0
    for item in retrieved:
        if not item.above_threshold:
            continue
        block = format_context_block(item)
        if included and used + len(block) > max_context_chars:
            break
        included.append(item)
        blocks.append(block)
        used += len(block)

    context_text = "\n\n".join(blocks) if blocks else "(no relevant context found)"
    user = f"Context:\n{context_text}\n\nQuestion: {question.strip()}\n\nAnswer with citations:"
    return Prompt(question=question.strip(), system=SYSTEM_PROMPT, user=user, context=included)
