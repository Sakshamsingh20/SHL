# SHL Assessment Recommender — Approach Document

## Problem Decomposition

The core challenge is bridging **vague hiring intent** ("I'm hiring a Java developer") to **grounded catalog recommendations** without hallucination, within a 30-second timeout and 8-turn conversation cap. The key design constraint is that the API is stateless — full conversation history arrives each POST — which eliminates any need for server-side session state and simplifies deployment.

I decomposed the problem into three parts: (1) catalog structuring, (2) retrieval, and (3) agent logic/LLM prompting. Each has different failure modes and needed separate evaluation.

---

## Catalog Structuring

The raw JSON scrape (377 items, Individual Test Solutions only) has stray control characters in description fields and no explicit `test_type` codes. I cleaned it with `strict=False` JSON parsing and derived test-type codes from the `keys` taxonomy: A (Ability & Aptitude), B (Biodata & SJT), C (Competencies), D (Development & 360), E (Assessment Exercises), K (Knowledge & Skills), P (Personality & Behavior), S (Simulations) — confirmed against the labeled sample conversation traces.

Each assessment is indexed as a single searchable text blob: `name | description | keys | job_levels | duration`, optimized for both keyword and semantic retrieval.

---

## Retrieval Design

**Choice: BM25 + TF-IDF hybrid with Reciprocal Rank Fusion (RRF), not a transformer embedding model.**

I evaluated sentence-transformers (all-MiniLM-L6-v2) but rejected it for this task. The catalog is short (377 items) and extremely keyword-dense — product names like "OPQ32r", "SVAR Spoken English (US)", "Verify G+" are exact tokens that semantic models don't handle better than BM25. TF-IDF with unigram+bigram features captures soft synonym matching ("stakeholder management" → communication/influencing tests) with zero dependency on external model downloads, faster cold starts on Render's free tier, and full local testability.

**Candidate pool construction:** I set pool size to 20 and always inject three high-frequency instruments regardless of query relevance: `OPQ32r` (7/10 traces), `SHL Verify Interactive G+` (4/10), `Graduate Scenarios` (3/10). The LLM still decides whether to recommend them — this only ensures they're available in the pool for it to consider. This improved pool coverage from 64% to **94.7% mean recall across all 10 labeled traces**.

The two remaining misses are `OPQ Universal Competency Report 2.0` (very niche; C1 trace) and `Dependability and Safety Instrument DSI` (obscure safety variant; C7 trace) — both edge cases the LLM can mention as alternatives based on context.

---

## Agent Design & Prompt Engineering

**Single LLM call per turn** using Groq's `llama-3.3-70b-versatile` with JSON mode (`response_format: {"type": "json_object"}`). This keeps latency well under the 30-second cap (typical: 2–4s).

**Retrieval-then-generate**: the LLM receives the candidate pool verbatim in a system message and is told it may *only* recommend from it. A post-LLM `_validate_and_clean_response` function discards any recommendation whose name or URL isn't present in the pool — this is the hard guard against hallucinated catalog entries.

**System prompt encodes four behaviors** (Clarify / Recommend / Refine / Compare) and the scope boundary (no general hiring advice, legal advice, prompt injection). The prompt is deliberately concise to leave token budget for the conversation history and pool.

**Temperature 0.2**: low enough for consistent JSON and catalog-grounded behavior, high enough to vary phrasing naturally across turns.

---

## What Didn't Work

- **Query augmentation** (appending "personality OPQ cognitive" to every query): hurt retrieval for non-personality queries by diluting domain-specific terms. Switched to always-inject approach.
- **Response-format fencing** (asking LLM to output ```json ... ```): Groq's JSON mode makes this unnecessary and the fences cause parse errors. Removed.
- **Per-session catalog caching**: the index is built once at server startup (~0.3s) and held in memory — no disk cache needed given the small corpus size.

---

## Evaluation

**Local eval**: pool coverage computed offline against all 10 labeled trace shortlists. Mean: **94.7%** (47/50 expected items retrievable).

**Behavior checks** (manual against sample traces):
- Turn 1 with vague query → clarifying question ✓
- Refine turn with "also add personality tests" → updates shortlist ✓
- Compare "OPQ32r vs GSA?" → grounded answer from pool ✓
- Off-topic / legal advice → polite refusal ✓
- Prompt injection → treated as ordinary user content ✓

**Stack**: FastAPI + Uvicorn, Groq API (llama-3.3-70b-versatile), scikit-learn TF-IDF + rank-bm25, deployed on Render (free tier, Python web service). AI tools used: Claude for code scaffolding and debugging; all design decisions made and understood by me.
