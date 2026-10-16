# Ledger Agent Server

A FastAPI service for the Acme Appliances ledger assignment. It accepts a plain-English question and returns `{"answer": ...}`. The implementation discovers common API documentation and collection paths through the personalized `questionData` proxy, follows common pagination styles, loads returned records into SQLite, and uses SQL for arithmetic.

## 1. Configure environment variables

Set these in your hosting provider's environment settings (do not commit secrets):

- `QUESTION_DATA_URL`: the *full* personalized URL supplied in the assignment, including `email`, `quizSign`, `questionId`, and `path=/`.
- `AIPROXY_TOKEN`: your course AI Proxy token. Alternatively set `OPENAI_API_KEY` for an OpenAI-compatible provider.
- `OPENAI_BASE_URL`: defaults to `https://aiproxy.sanand.workers.dev/openai/v1`.
- `OPENAI_MODEL`: defaults to `gpt-4o-mini`.

Do not include the `#/default/...` fragment from Swagger docs in the endpoint URL submitted to the grader.

## 2. Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Set the environment variables, then run:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 3. Test the exact POST contract

```bash
curl -X POST http://127.0.0.1:8000/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What was the total revenue in USD from the North region in March 2026?"}'
```

Expected response shape:

```json
{"answer": 12345.67}
```

The number above is only an example of the response shape, not the answer to the assignment.

Useful diagnostics:

- `GET /health` shows which tables were discovered.
- `GET /docs` opens Swagger UI for local testing. The grader endpoint is the deployed service root, for example `https://YOUR-SERVICE.onrender.com/`, not the `/docs` URL.
- `POST /answer` is an alias for `POST /`.

## 4. Deploy on Render

1. Put these files in a GitHub repository.
2. Create a Render **Web Service** from that repository (or use the included `render.yaml`).
3. Add `QUESTION_DATA_URL` and `AIPROXY_TOKEN` as private environment variables in Render.
4. Deploy, then open `/health`. Confirm that tables are loaded.
5. Test a POST to the service root with JSON before submitting the root URL as the assignment endpoint.

## Notes

- API discovery handles OpenAPI GET paths, common ledger collection paths, and common `next`, cursor, page, or offset pagination metadata. An undocumented API with a nonstandard pagination scheme may need its specific pagination field added in `server.py` after inspecting the root response.
- The questionData URL includes a quiz signature. Keep it out of public repositories, screenshots, and logs.
- A free web-service tier may sleep when idle; cold starts can be too slow for a strict 15-second grader. Warm the deployment before testing and use a hosting tier that meets the assignment's latency requirement.
