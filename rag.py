import os
import json
import re
from typing import Any

from critic import Critic, GROUNDNESS_TOKENS, RETRIEVAL_TOKENS, UTILITY_TOKENS
from retrieve import retrieve
from yandex_client import yandex_complete

MODEL_UTILITY_FLOOR = int(os.getenv("CRITIC_UTILITY_FLOOR", "3"))
MODEL_GROUNDNESS_FLOOR = float(os.getenv("CRITIC_GROUNDNESS_FLOOR", "0.5"))
CRITIC_FILTER_CONTEXTS = os.getenv("CRITIC_FILTER_CONTEXTS", "0").strip() == "1"
CRITIC_POSTCHECK = os.getenv("CRITIC_POSTCHECK", "0").strip() == "1"
CRITIC_RETRY = os.getenv("CRITIC_RETRY", "0").strip() == "1"

# Self-RAG style controls.
SELF_RAG_MODE = os.getenv("SELF_RAG_MODE", "adaptive_retrieval").strip()
SELF_RAG_THRESHOLD = float(os.getenv("SELF_RAG_THRESHOLD", "0.5"))
SELF_RAG_W_REL = float(os.getenv("SELF_RAG_W_REL", "1.0"))
SELF_RAG_W_SUP = float(os.getenv("SELF_RAG_W_SUP", "1.0"))
SELF_RAG_W_USE = float(os.getenv("SELF_RAG_W_USE", "0.5"))
SELF_RAG_MAX_SCORE_CONTEXTS = int(os.getenv("SELF_RAG_MAX_SCORE_CONTEXTS", "5"))
SELF_RAG_USE_BEAM = os.getenv("SELF_RAG_USE_BEAM", "1").strip() == "1"
SELF_RAG_BEAM_WIDTH = int(os.getenv("SELF_RAG_BEAM_WIDTH", "2"))
SELF_RAG_MAX_DEPTH = int(os.getenv("SELF_RAG_MAX_DEPTH", "2"))
SELF_RAG_BEAM_CTX_PER_STEP = int(os.getenv("SELF_RAG_BEAM_CTX_PER_STEP", "2"))
SELF_RAG_SEGMENT_MAX_TOKENS = int(os.getenv("SELF_RAG_SEGMENT_MAX_TOKENS", "220"))


def _build_citation(c: dict) -> str:
    petitioner = (c.get("pet") or "").strip()
    respondent = (c.get("res") or "").strip()
    case_no = (c.get("case_no") or "").strip()
    file_name = (c.get("file_name") or "Unknown Source").strip()

    if petitioner and respondent:
        citation = f"{petitioner} vs {respondent}"
        if case_no:
            citation += f" (Case No: {case_no})"
        return citation

    return file_name


def _format_contexts(contexts: list[dict]) -> str:
    context_text = ""
    for c in contexts:
        citation = _build_citation(c)
        chunk_id = c.get("chunk_id")
        url = (c.get("url") or "").strip()
        link_part = f", Link: {url}" if url else ""

        text = (c.get("text") or "").strip()
        if not text:
            continue

        context_text += (
            f"[Source: {citation}{link_part}, Chunk: {chunk_id}]\n"
            f"{text}\n\n"
        )
    return context_text


def _build_prompt(question: str, context_text: str) -> str:
    return f"""
You are a legal assistant.

Use ONLY the context below to answer the question.

Rules:
- First give a clear answer in 3-8 bullet points (or a short paragraph).
- Then add a section titled "Citations:" with 1-5 citations.
- Every bullet/claim must be supported by at least one citation.
- Do NOT output only citations.
- If the context is insufficient, say so.

Citations format (exact):
[Source: PETITIONER vs RESPONDENT (Case No: ...), Link: URL, Chunk: n]

Context:
{context_text}

Question:
{question}
""".strip()


def _generate_with_context(question: str, contexts: list[dict]) -> str:
    context_text = _format_contexts(contexts)
    if not context_text.strip():
        context_text = "[No relevant context found in vector DB]\n"
    prompt = _build_prompt(question, context_text)
    return yandex_complete(prompt, temperature=0.2, max_tokens=1000)


def _generate_without_context(question: str) -> str:
    prompt = f"""
You are a legal assistant.

The user asked:
{question}

Rules:
- If this requires case-specific facts, explicitly say retrieval/context is required.
- Otherwise provide a concise answer.
""".strip()
    return yandex_complete(prompt, temperature=0.2, max_tokens=600)


def _extract_json_object(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _generate_segment(question: str, draft: str, contexts: list[dict]) -> tuple[str, bool]:
    context_text = _format_contexts(contexts).strip() if contexts else ""
    prompt = f"""
You are writing a grounded legal answer in multiple steps.
Return JSON only with this schema:
{{
  "segment": "next short segment (1-3 sentences)",
  "done": true or false
}}

Question:
{question}

Current draft:
{draft if draft.strip() else "[EMPTY]"}

Context for this step:
{context_text if context_text else "[NO CONTEXT]"}
""".strip()
    raw = yandex_complete(prompt, temperature=0.1, max_tokens=SELF_RAG_SEGMENT_MAX_TOKENS)
    parsed = _extract_json_object(raw) or {}
    segment = str(parsed.get("segment", "")).strip()
    done = bool(parsed.get("done", False))
    if not segment:
        segment = raw.strip()
    if not segment:
        segment = ""
    return segment, done


def _filter_contexts_with_critic(critic: Critic, question: str, contexts: list[dict]) -> list[dict]:
    filtered = []
    for ctx in contexts:
        evidence = (ctx.get("text") or "").strip()
        if not evidence:
            continue
        token = critic.relevance(question, evidence)
        if token == "[Relevant]":
            filtered.append(ctx)
    return filtered


def _retrieval_probability(critic: Critic, question: str) -> tuple[float, dict[str, float]]:
    dist = critic.retrieval_distribution(question)
    retrieve_p = dist.get("[Retrieval]", 0.0)
    no_retrieve_p = dist.get("[No Retrieval]", 0.0)
    denom = retrieve_p + no_retrieve_p
    if denom <= 0:
        return 1.0, dist
    return retrieve_p / denom, dist


def _utility_score_from_dist(ut_dist: dict[str, float]) -> float:
    # Self-RAG weighting in short-form scorer: [-1, -0.5, 0, 0.5, 1]
    token_weights = {
        "[Utility:1]": -1.0,
        "[Utility:2]": -0.5,
        "[Utility:3]": 0.0,
        "[Utility:4]": 0.5,
        "[Utility:5]": 1.0,
    }
    return sum(token_weights[tok] * ut_dist.get(tok, 0.0) for tok in UTILITY_TOKENS)


def _ground_score_from_dist(grd_dist: dict[str, float]) -> float:
    return (
        grd_dist.get("[Fully supported]", 0.0)
        + 0.5 * grd_dist.get("[Partially supported]", 0.0)
    )


def _score_candidate(
    critic: Critic,
    question: str,
    answer: str,
    contexts: list[dict],
) -> dict[str, Any]:
    evidence_text = "\n\n".join(
        [(ctx.get("text") or "").strip() for ctx in contexts if (ctx.get("text") or "").strip()]
    )[:12000]

    rel_dist = critic.relevance_distribution(question, evidence_text if evidence_text else "[No evidence]")
    grd_dist = critic.groundness_distribution(question, answer, evidence_text if evidence_text else "[No evidence]")
    ut_dist = critic.utility_distribution(question, answer)

    relevance_score = rel_dist.get("[Relevant]", 0.0)
    ground_score = _ground_score_from_dist(grd_dist)
    utility_score = _utility_score_from_dist(ut_dist)
    final_score = (
        SELF_RAG_W_REL * relevance_score
        + SELF_RAG_W_SUP * ground_score
        + SELF_RAG_W_USE * utility_score
    )

    best_grd = max(GROUNDNESS_TOKENS, key=lambda t: grd_dist.get(t, 0.0))
    best_ut = max(UTILITY_TOKENS, key=lambda t: ut_dist.get(t, 0.0))
    return {
        "final_score": final_score,
        "relevance_score": relevance_score,
        "ground_score": ground_score,
        "utility_score": utility_score,
        "groundness_token": best_grd,
        "utility_token": best_ut,
        "rel_dist": rel_dist,
        "grd_dist": grd_dist,
        "ut_dist": ut_dist,
    }


def _run_beam_search_api(
    critic: Critic,
    question: str,
    contexts: list[dict],
    mode: str,
) -> tuple[str, list[dict], list[dict[str, Any]]]:
    initial_node = {
        "draft": "",
        "score": 0.0,
        "contexts": [],
        "done": False,
        "trace": [],
    }
    frontier = [initial_node]
    all_nodes = []
    per_step_ctxs = contexts[: max(1, min(len(contexts), SELF_RAG_BEAM_CTX_PER_STEP))]

    for _ in range(max(1, SELF_RAG_MAX_DEPTH)):
        expanded = []
        for node in frontier:
            if node["done"]:
                expanded.append(node)
                continue

            probe_q = question
            if node["draft"].strip():
                probe_q += f"\nCurrent answer draft: {node['draft'][:1200]}"
            retrieve_prob, _ = _retrieval_probability(critic, probe_q)
            no_retrieve_prob = 1.0 - retrieve_prob

            allow_retrieval = mode == "always_retrieve" or (
                mode == "adaptive_retrieval" and retrieve_prob > SELF_RAG_THRESHOLD
            )
            allow_no_retrieval = mode != "always_retrieve" or not allow_retrieval

            if allow_no_retrieval:
                seg, done = _generate_segment(question, node["draft"], [])
                draft = (node["draft"] + "\n" + seg).strip() if seg else node["draft"]
                score = _score_candidate(critic, question, draft, node["contexts"])
                expanded.append(
                    {
                        "draft": draft,
                        "score": node["score"] + score["final_score"] + no_retrieve_prob,
                        "contexts": node["contexts"],
                        "done": done,
                        "trace": node["trace"] + [{"decision": "[No Retrieval]", "score": score["final_score"]}],
                    }
                )

            if allow_retrieval:
                for ctx in per_step_ctxs:
                    seg, done = _generate_segment(question, node["draft"], [ctx])
                    draft = (node["draft"] + "\n" + seg).strip() if seg else node["draft"]
                    new_contexts = node["contexts"] + [ctx]
                    score = _score_candidate(critic, question, draft, new_contexts)
                    expanded.append(
                        {
                            "draft": draft,
                            "score": node["score"] + score["final_score"] + retrieve_prob,
                            "contexts": new_contexts,
                            "done": done,
                            "trace": node["trace"] + [{"decision": "[Retrieval]", "score": score["final_score"]}],
                        }
                    )

        expanded = sorted(expanded, key=lambda n: n["score"], reverse=True)
        frontier = expanded[: max(1, SELF_RAG_BEAM_WIDTH)]
        all_nodes.extend(frontier)
        if frontier and all(n["done"] for n in frontier):
            break

    best = sorted(frontier, key=lambda n: n["score"], reverse=True)[0]
    # Finalize in your existing answer format for consistency.
    dedup_contexts = []
    seen_ids = set()
    for c in best["contexts"]:
        cid = c.get("id") or f"{c.get('file_name')}::{c.get('chunk_id')}"
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        dedup_contexts.append(c)

    if dedup_contexts:
        final_answer = _generate_with_context(question, dedup_contexts)
    else:
        final_answer = _generate_without_context(question)

    beam_meta = [
        {
            "path_score": n["score"],
            "used_context_count": len(n["contexts"]),
            "done": n["done"],
            "trace": n["trace"],
        }
        for n in sorted(all_nodes, key=lambda n: n["score"], reverse=True)[: max(1, SELF_RAG_BEAM_WIDTH * 2)]
    ]
    return final_answer, dedup_contexts, beam_meta


def _self_rag_mode() -> str:
    valid = {"adaptive_retrieval", "no_retrieval", "always_retrieve"}
    return SELF_RAG_MODE if SELF_RAG_MODE in valid else "adaptive_retrieval"


def _utility_bucket(token: str) -> int:
    mapping = {
        "[Utility:1]": 1,
        "[Utility:2]": 2,
        "[Utility:3]": 3,
        "[Utility:4]": 4,
        "[Utility:5]": 5,
    }
    return mapping.get(token, 3)


def _ground_bucket(token: str) -> float:
    mapping = {
        "[Fully supported]": 1.0,
        "[Partially supported]": 0.5,
        "[No support / Contradictory]": 0.0,
    }
    return mapping.get(token, 0.5)


def generate_answer(
    question: str,
    top_k: int = 5,
    case_no: str | None = None,
    file_name: str | None = None,
):
    """
    Returns: (answer_text, contexts, meta)
    Optional filters restrict retrieval to a case/file.
    """
    critic_enabled = os.getenv("USE_CRITIC", "1").strip() == "1"
    critic = Critic() if critic_enabled else None

    mode = _self_rag_mode()
    meta: dict[str, Any] = {
        "critic_enabled": critic_enabled,
        "mode": mode,
        "threshold": SELF_RAG_THRESHOLD,
        "retrieval_decision": None,
        "retrieval_distribution": None,
        "retrieval_probability": None,
        "initial_context_count": 0,
        "used_context_count": 0,
        "relevance_filtered_count": 0,
        "groundness_token": None,
        "groundness_score": None,
        "utility_token": None,
        "utility_score": None,
        "retry_performed": False,
        "candidate_scores": [],
        "beam_search_used": False,
        "beam_width": SELF_RAG_BEAM_WIDTH,
        "max_depth": SELF_RAG_MAX_DEPTH,
        "beam_paths": [],
    }

    do_retrieve = True
    if mode == "always_retrieve":
        do_retrieve = True
    elif mode == "no_retrieval":
        do_retrieve = False
    elif critic is not None:
        retrieve_prob, dist = _retrieval_probability(critic, question)
        do_retrieve = retrieve_prob > SELF_RAG_THRESHOLD
        meta["retrieval_distribution"] = dist
        meta["retrieval_probability"] = retrieve_prob

    meta["retrieval_decision"] = RETRIEVAL_TOKENS[0] if do_retrieve else RETRIEVAL_TOKENS[1]
    if not do_retrieve:
        return _generate_without_context(question), [], meta

    contexts = retrieve(question, top_k=top_k, case_no=case_no, file_name=file_name)
    meta["initial_context_count"] = len(contexts)

    used_contexts = contexts
    if critic is not None and contexts and CRITIC_FILTER_CONTEXTS:
        relevant_contexts = _filter_contexts_with_critic(critic, question, contexts)
        if relevant_contexts:
            used_contexts = relevant_contexts
        meta["relevance_filtered_count"] = len(relevant_contexts)

    if critic is not None and SELF_RAG_USE_BEAM:
        beam_answer, beam_contexts, beam_paths = _run_beam_search_api(critic, question, used_contexts, mode)
        answer_text = beam_answer
        used_contexts = beam_contexts
        meta["beam_search_used"] = True
        meta["beam_paths"] = beam_paths
        if used_contexts:
            best_score = _score_candidate(critic, question, answer_text, used_contexts)
            meta["groundness_token"] = best_score["groundness_token"]
            meta["groundness_score"] = _ground_bucket(best_score["groundness_token"])
            meta["utility_token"] = best_score["utility_token"]
            meta["utility_score"] = _utility_bucket(best_score["utility_token"])
        meta["used_context_count"] = len(used_contexts)
        return answer_text, used_contexts, meta

    answer_text = _generate_with_context(question, used_contexts)

    if critic is not None and used_contexts:
        candidates = used_contexts[: max(1, min(len(used_contexts), SELF_RAG_MAX_SCORE_CONTEXTS))]
        scored: list[tuple[str, dict, dict[str, Any]]] = []
        for ctx in candidates:
            per_ctx = [ctx]
            cand_answer = _generate_with_context(question, per_ctx)
            score = _score_candidate(critic, question, cand_answer, per_ctx)
            scored.append((cand_answer, per_ctx[0], score))

        # Also score the combined-context answer.
        base_score = _score_candidate(critic, question, answer_text, used_contexts)
        scored.append((answer_text, {"id": "combined"}, base_score))

        ranked = sorted(scored, key=lambda x: x[2]["final_score"], reverse=True)
        best_answer, best_ctx, best_score = ranked[0]
        answer_text = best_answer
        if best_ctx.get("id") != "combined":
            used_contexts = [best_ctx]

        meta["candidate_scores"] = [
            {
                "final_score": item[2]["final_score"],
                "relevance_score": item[2]["relevance_score"],
                "ground_score": item[2]["ground_score"],
                "utility_score": item[2]["utility_score"],
                "groundness_token": item[2]["groundness_token"],
                "utility_token": item[2]["utility_token"],
                "candidate_type": "combined" if item[1].get("id") == "combined" else "single_context",
            }
            for item in ranked
        ]
        meta["groundness_token"] = best_score["groundness_token"]
        meta["groundness_score"] = _ground_bucket(best_score["groundness_token"])
        meta["utility_token"] = best_score["utility_token"]
        meta["utility_score"] = _utility_bucket(best_score["utility_token"])

    meta["used_context_count"] = len(used_contexts)

    if critic is not None and used_contexts and CRITIC_POSTCHECK:
        evidence_text = "\n\n".join(
            [(ctx.get("text") or "").strip() for ctx in used_contexts if (ctx.get("text") or "").strip()]
        )[:12000]
        ground_tok = critic.groundness(question, answer_text, evidence_text)
        utility_tok = critic.utility(question, answer_text)
        ground_score = _ground_bucket(ground_tok)
        utility_score = _utility_bucket(utility_tok)

        meta["groundness_token"] = ground_tok
        meta["groundness_score"] = ground_score
        meta["utility_token"] = utility_tok
        meta["utility_score"] = utility_score

        low_quality = (ground_score < MODEL_GROUNDNESS_FLOOR) or (utility_score < MODEL_UTILITY_FLOOR)
        if low_quality and CRITIC_RETRY:
            retry_contexts = retrieve(question, top_k=max(8, top_k + 3), case_no=case_no, file_name=file_name)
            retry_used = retry_contexts
            if retry_contexts and CRITIC_FILTER_CONTEXTS:
                relevant_retry = _filter_contexts_with_critic(critic, question, retry_contexts)
                if relevant_retry:
                    retry_used = relevant_retry
            retry_answer = _generate_with_context(question, retry_used)
            answer_text = retry_answer
            used_contexts = retry_used
            meta["retry_performed"] = True
            meta["used_context_count"] = len(used_contexts)

    return answer_text, used_contexts, meta
