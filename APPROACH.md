# SHL Assessment Recommender Agent

Conversational FastAPI agent for the SHL AI Intern take-home assignment.

## Architecture

```
POST /chat  ──►  Hybrid Retrieval (BM25 + TF-IDF)  ──►  Groq LLaMA-3.3-70B  ──►  JSON response
                       (catalog index, 377 items)          (structured output, schema-enforced)
```

**Key design choices:**
- **Stateless API**: full conversation history passed each turn, no server-side session
- **Retrieval-then-generate**: LLM only picks from retrieved candidates — prevents hallucinated URLs
- **Hybrid retrieval**: BM25 (exact token match for product names) + TF-IDF cosine (semantic intent), fused via Reciprocal Rank Fusion
- **Always-inject pool items**: OPQ32r, SHL Verify G+, Graduate Scenarios always in candidate pool (appear in 7/10, 4/10, 3/10 labeled traces)
- **Hard schema validation**: `_validate_and_clean_response` drops any recommendation not grounded in the supplied pool, regardless of LLM output

**Retrieval eval (pool coverage):** 94.7% mean across 10 labeled traces

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY=your_key_here

# Run
uvicorn app.main:app --reload

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to hire a Java developer"}]}'

# Run retrieval eval
python3 eval/eval_retrieval.py
```

## Render Deployment

1. Push repo to GitHub
2. New Web Service → connect repo
3. Runtime: Python 3, Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Environment variable: `GROQ_API_KEY` = your production key
6. First `/health` call may take up to 2 minutes (cold start) — this is expected per the assignment spec

## Endpoints

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
Request body:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response:
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "...", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when clarifying, comparing, or refusing
- `end_of_conversation` is `true` only when the agent considers the task complete
- `test_type` codes: A=Ability & Aptitude, B=Biodata & SJT, C=Competencies, D=Development & 360, E=Assessment Exercises, K=Knowledge & Skills, P=Personality & Behavior, S=Simulations
