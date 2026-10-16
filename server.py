"""Personalized Ledger Agent API for the TDS assignment.

POST / or POST /answer with {"question": "..."} -> {"answer": ...}

The source API is reached through the questionData proxy URL. Configure that URL
and an OpenAI-compatible API token via environment variables; do not commit them.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

QUESTION_DATA_URL = os.getenv("QUESTION_DATA_URL", "").strip()
AI_TOKEN = os.getenv("AIPROXY_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
AI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://aiproxy.sanand.workers.dev/openai/v1").strip()
AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
REQUEST_TIMEOUT = float(os.getenv("SOURCE_TIMEOUT_SECONDS", "4.5"))
MAX_API_CALLS = int(os.getenv("MAX_API_CALLS_PER_QUESTION", "8"))

app = FastAPI(title="Ledger Agent", version="1.0.0")

# The source API and its response cache.
_http = requests.Session()
_http.headers.update({"Accept": "application/json", "User-Agent": "ledger-agent/1.0"})
_http_lock = threading.Lock()
_cache_lock = threading.RLock()
_api_cache: OrderedDict[str, Any] = OrderedDict()
_CACHE_LIMIT = 300

# One in-memory SQLite DB holds the collections discovered from the source API.
_db = sqlite3.connect(":memory:", check_same_thread=False)
_db.row_factory = sqlite3.Row
_db_lock = threading.RLock()
_table_columns: dict[str, list[str]] = {}
_table_counts: dict[str, int] = {}
_table_records: dict[str, list[dict[str, Any]]] = {}
_bootstrap_done = False
_bootstrap_lock = threading.Lock()

COMMON_PATHS = [
    "/", "/openapi.json", "/api/openapi.json", "/swagger.json",
    "/orders", "/products", "/customers", "/refunds", "/transactions",
    "/ledger", "/api/orders", "/api/products", "/api/customers",
    "/api/refunds", "/api/transactions", "/api/ledger",
]


def _safe_name(value: str, default: str = "records") -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().strip("/").lower()).strip("_")
    if not name:
        name = default
    if name[0].isdigit():
        name = "t_" + name
    return name[:60]


def _normalize_path(path: str) -> str:
    """Normalize an API path while preserving its query string."""
    path = (path or "/").strip()
    if path.startswith("http://") or path.startswith("https://"):
        parsed = urlsplit(path)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
    if not path.startswith("/"):
        path = "/" + path
    return path


def _call_source_once(path: str) -> Any:
    if not QUESTION_DATA_URL:
        raise RuntimeError("QUESTION_DATA_URL is not configured")
    path = _normalize_path(path)
    parsed = urlsplit(QUESTION_DATA_URL)
    source_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    source_params["path"] = path
    base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    # Serializes the signed query correctly; no token is printed in logs.
    with _http_lock:
        response = _http.get(base, params=source_params, timeout=REQUEST_TIMEOUT)
    text = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = {"_http_status": response.status_code, "_content_type": response.headers.get("content-type", ""), "_text": text[:12000]}
    if response.status_code >= 400:
        return {"_http_status": response.status_code, "_error": payload}
    return payload


def get_api(path: str, use_cache: bool = True) -> Any:
    path = _normalize_path(path)
    if use_cache:
        with _cache_lock:
            if path in _api_cache:
                _api_cache.move_to_end(path)
                return _api_cache[path]
    payload = _call_source_once(path)
    with _cache_lock:
        _api_cache[path] = payload
        _api_cache.move_to_end(path)
        while len(_api_cache) > _CACHE_LIMIT:
            _api_cache.popitem(last=False)
    return payload


def _is_record_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(item, dict) for item in value)


def _primary_records(payload: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Find the main collection and its semantic key in a common JSON API shape."""
    if _is_record_list(payload):
        return payload, None
    if isinstance(payload, dict):
        preferred = ("data", "items", "results", "records", "orders", "customers", "products", "refunds", "transactions", "ledger", "rows")
        for key in preferred:
            value = payload.get(key)
            if _is_record_list(value):
                return value, key
            if isinstance(value, dict):
                found, child_key = _primary_records(value)
                if found is not None:
                    return found, child_key or key
        for key, value in payload.items():
            if _is_record_list(value):
                return value, key
        for value in payload.values():
            if isinstance(value, dict):
                found, key = _primary_records(value)
                if found is not None:
                    return found, key
    return None, None


def _append_query(path: str, key: str, value: Any) -> str:
    parsed = urlsplit(_normalize_path(path))
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    pairs = [(k, v) for k, v in pairs if k != key]
    pairs.append((key, str(value)))
    return urlunsplit(("", "", parsed.path, urlencode(pairs), ""))


def _next_page(payload: Any, current_path: str, current_count: int) -> str | None:
    """Recognize common next-link, cursor, page-number and offset pagination forms."""
    if not isinstance(payload, dict):
        return None
    candidate = None
    for key in ("next", "next_url", "nextUrl", "next_page_url", "nextPageUrl"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            break
    for container_key in ("links", "pagination", "meta", "page_info", "pageInfo"):
        container = payload.get(container_key)
        if candidate is None and isinstance(container, dict):
            for key in ("next", "next_url", "nextUrl", "next_page_url", "nextPageUrl"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    candidate = value.strip()
                    break
    if candidate:
        if candidate.startswith("http://") or candidate.startswith("https://"):
            parsed = urlsplit(candidate)
            return (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
        return _normalize_path(candidate)

    containers = [payload]
    for key in ("pagination", "meta", "page_info", "pageInfo"):
        if isinstance(payload.get(key), dict):
            containers.append(payload[key])
    flat = {}
    for c in containers:
        flat.update(c)

    cursor = flat.get("next_cursor") or flat.get("nextCursor") or flat.get("cursor_next")
    if cursor:
        return _append_query(current_path, "cursor", cursor)
    if flat.get("has_more") is True or flat.get("hasMore") is True:
        cur_cursor = flat.get("cursor") or flat.get("end_cursor") or flat.get("endCursor")
        if cur_cursor:
            return _append_query(current_path, "cursor", cur_cursor)

    current = flat.get("current_page", flat.get("currentPage", flat.get("page")))
    total_pages = flat.get("total_pages", flat.get("totalPages", flat.get("last_page", flat.get("lastPage"))))
    if isinstance(current, int) and isinstance(total_pages, int) and current < total_pages:
        return _append_query(current_path, "page", current + 1)

    total = flat.get("total", flat.get("total_count", flat.get("totalCount")))
    limit = flat.get("limit", flat.get("per_page", flat.get("perPage", flat.get("page_size", flat.get("pageSize")))))
    offset = flat.get("offset", 0)
    if isinstance(total, int) and isinstance(limit, int) and limit > 0 and isinstance(offset, int) and offset + current_count < total:
        return _append_query(current_path, "offset", offset + current_count)
    return None


def _fetch_pages(path: str, max_pages: int = 100) -> tuple[list[dict[str, Any]], list[Any]]:
    all_records: list[dict[str, Any]] = []
    payloads: list[Any] = []
    current = _normalize_path(path)
    visited: set[str] = set()
    for _ in range(max_pages):
        if current in visited:
            break
        visited.add(current)
        payload = get_api(current)
        payloads.append(payload)
        recs, _key = _primary_records(payload)
        if recs:
            all_records.extend(recs)
        next_path = _next_page(payload, current, len(recs or []))
        if not next_path or next_path in visited:
            break
        current = next_path
    # Remove exact duplicate records occasionally repeated at page boundaries.
    if len(all_records) > 1:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in all_records:
            fingerprint = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(record)
        all_records = unique
    return all_records, payloads


def _flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key, value in record.items():
        key = _safe_name(str(raw_key), "field")
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_record(value, full_key))
        elif isinstance(value, (list, tuple)):
            out[full_key] = json.dumps(value, ensure_ascii=False, default=str)
        elif isinstance(value, bool):
            out[full_key] = int(value)
        elif value is None or isinstance(value, (str, int, float)):
            out[full_key] = value
        else:
            out[full_key] = str(value)
    return out


def _sqlite_type(values: list[Any]) -> str:
    nonnull = [v for v in values if v is not None]
    if nonnull and all(isinstance(v, int) and not isinstance(v, bool) for v in nonnull):
        return "INTEGER"
    if nonnull and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nonnull):
        return "REAL"
    return "TEXT"


def _insert_records(table: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    table = _safe_name(table)
    flattened = [_flatten_record(record) for record in records]
    columns: list[str] = []
    for row in flattened:
        for col in row:
            if col not in columns:
                columns.append(col)
    if not columns:
        return
    with _db_lock:
        existing_cols = _table_columns.get(table)
        if existing_cols is None:
            definitions = []
            for col in columns:
                values = [row.get(col) for row in flattened]
                definitions.append(f'"{col}" {_sqlite_type(values)}')
            _db.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(definitions)})')
            _table_columns[table] = list(columns)
            _table_records[table] = []
            existing_cols = _table_columns[table]
        else:
            for col in columns:
                if col not in existing_cols:
                    _db.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" TEXT')
                    existing_cols.append(col)
        old_rows = _table_records.setdefault(table, [])
        known = {json.dumps(row, sort_keys=True, default=str, separators=(",", ":")) for row in old_rows}
        to_add = []
        for row in flattened:
            fingerprint = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
            if fingerprint not in known:
                known.add(fingerprint)
                to_add.append(row)
                old_rows.append(row)
        if to_add:
            cols_sql = ", ".join(f'"{col}"' for col in existing_cols)
            marks = ", ".join("?" for _ in existing_cols)
            values = [[row.get(col) for col in existing_cols] for row in to_add]
            _db.executemany(f'INSERT INTO "{table}" ({cols_sql}) VALUES ({marks})', values)
        _db.commit()
        _table_counts[table] = len(old_rows)


def _ingest_payload(path: str, payload: Any, full_records: list[dict[str, Any]] | None = None) -> None:
    # The path name is the fallback. A semantic top-level array key is preferred.
    recs, key = _primary_records(payload)
    records = full_records if full_records is not None else recs
    if records:
        path_name = urlsplit(_normalize_path(path)).path.rstrip("/").split("/")[-1] or "root_records"
        table_name = key if key and key not in ("data", "items", "results", "records", "rows") else path_name
        if table_name in ("api", "v1", "v2") or not table_name:
            table_name = path_name or "root_records"
        _insert_records(table_name, records)
    # Also ingest separately named nested arrays if they carry object records.
    if isinstance(payload, dict):
        for key2, value in payload.items():
            if _is_record_list(value):
                _insert_records(key2, value)
            elif isinstance(value, dict):
                for key3, value3 in value.items():
                    if _is_record_list(value3):
                        _insert_records(key3, value3)


def _extract_openapi_paths(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        # Sometimes the API returns an envelope around the spec.
        for key in ("data", "openapi", "spec"):
            if isinstance(payload.get(key), dict) and isinstance(payload[key].get("paths"), dict):
                paths = payload[key]["paths"]
                break
    if not isinstance(paths, dict):
        return []
    result = []
    for path, methods in paths.items():
        if not isinstance(path, str) or "{" in path or not isinstance(methods, dict):
            continue
        if any(method.lower() == "get" for method in methods):
            result.append(path)
    return result[:80]


def bootstrap() -> None:
    """Discover likely docs and collection routes; cache data before graded requests."""
    global _bootstrap_done
    if _bootstrap_done:
        return
    with _bootstrap_lock:
        if _bootstrap_done:
            return
        if not QUESTION_DATA_URL:
            _bootstrap_done = True
            return
        paths_to_try = list(COMMON_PATHS)
        # First wave: root plus docs and likely collection paths.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        first_payloads: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(get_api, path): path for path in paths_to_try}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    first_payloads[path] = future.result()
                except Exception:
                    continue
        discovered = []
        for path, payload in first_payloads.items():
            discovered.extend(_extract_openapi_paths(payload))
            _ingest_payload(path, payload)
        all_paths = list(dict.fromkeys(paths_to_try + discovered))
        # Second wave: GET discovered collection paths, following recognized pagination.
        def fetch_and_ingest(path: str) -> None:
            try:
                records, payloads = _fetch_pages(path)
                for payload in payloads:
                    _ingest_payload(path, payload)
                if records:
                    _ingest_payload(path, payloads[0] if payloads else records, records)
            except Exception:
                return
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch_and_ingest, all_paths))
        _bootstrap_done = True


def _schema_context() -> str:
    with _db_lock:
        if not _table_columns:
            return "No SQL tables are loaded yet. Use api_get to inspect the API root, documentation, and collections."
        parts = []
        for table in sorted(_table_columns):
            cols = _table_columns[table]
            rows = _table_records.get(table, [])[:2]
            sample = json.dumps(rows, ensure_ascii=False, default=str)[:1100]
            parts.append(f"TABLE {table} ({_table_counts.get(table, 0)} rows)\nCOLUMNS: {', '.join(cols)}\nSAMPLE: {sample}")
        return "\n\n".join(parts)[:16000]


def _list_tables() -> dict[str, Any]:
    with _db_lock:
        return {"tables": [{"name": table, "row_count": _table_counts.get(table, 0), "columns": cols} for table, cols in sorted(_table_columns.items())]}


def _run_sql(sql: str) -> dict[str, Any]:
    sql = (sql or "").strip()
    if not sql or ";" in sql.rstrip(";"):
        return {"error": "Use one read-only SQL query without multiple statements."}
    sql = sql.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.I):
        return {"error": "Only SELECT queries are allowed."}

    def authorizer(action: int, arg1: str | None, arg2: str | None, dbname: str | None, source: str | None) -> int:
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    with _db_lock:
        _db.set_authorizer(authorizer)
        try:
            cursor = _db.execute(sql)
            columns = [d[0] for d in (cursor.description or [])]
            rows = [dict(row) for row in cursor.fetchmany(100)]
            return {"columns": columns, "rows": rows, "row_count_returned": len(rows)}
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            _db.set_authorizer(None)


def _tool_api_get(path: str) -> dict[str, Any]:
    try:
        records, payloads = _fetch_pages(path, max_pages=100)
        for payload in payloads:
            _ingest_payload(path, payload)
        if records and payloads:
            _ingest_payload(path, payloads[0], records)
        payload = payloads[0] if payloads else get_api(path)
        # Avoid flooding the model; include first few records and pagination metadata.
        recs, key = _primary_records(payload)
        output: dict[str, Any] = {"requested_path": _normalize_path(path), "tables": _list_tables().get("tables", [])}
        if records:
            output["record_count_fetched"] = len(records)
            output["record_sample"] = records[:3]
            output["collection_key"] = key
        else:
            output["response"] = payload
        return output
    except Exception as exc:
        return {"error": f"GET failed for {_normalize_path(path)}: {exc}"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all discovered order-ledger tables, row counts, and columns.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sql",
            "description": "Run a read-only SQL SELECT against discovered source data. Use this for exact filtering, joins, sums, counts, averages, min/max, and rounding.",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string", "description": "One SQLite SELECT query."}}, "required": ["sql"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_get",
            "description": "Explore a source API GET path (for example '/', '/openapi.json', '/orders?page=2'). Follows common pagination, caches results, and ingests record collections into SQL tables.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative API path beginning with /, including query string if needed."}}, "required": ["path"]},
        },
    },
]

SYSTEM_PROMPT = """You answer questions about Acme Appliances' order ledger using only the discovered API data.
You have SQLite tables from the API. Inspect their schema and samples, then use query_sql for all arithmetic and filtering. If the schema is incomplete, use api_get to inspect the API root, OpenAPI schema and relevant routes; follow pagination fully before concluding.

Rules:
- Treat user wording flexibly; identify the requested metric, date/time range, geography, product, customer, currency, and status filters.
- Use exact date boundaries. For a calendar month use date >= first day and date < first day of next month. For phrases such as "orders placed in March", apply the date filter to the order placement/created date, NOT the refund date.
- Join tables only using IDs or relations supported by the data/schema. Match product names exactly as the API spells them.
- IMPORTANT REFUND SEMANTICS: If the user asks for "refunded orders", inspect BOTH the order table and any refunds/refund-transactions table. A refund may exist as a separate record even if the order's status is not literally "refunded". When a refunds table exists and links to orders, count an order as refunded when it has at least one qualifying refund record; use EXISTS or COUNT(DISTINCT order primary key) so multiple refund records do not duplicate an order. Use the order table for order date and region filters. Only rely exclusively on an order status field if the schema/data shows that is the actual source of refund truth or no refund table exists. If refund status is present, distinguish completed/successful refunds from pending/failed/cancelled ones; inspect actual status values instead of guessing.
- For a count that unexpectedly returns zero, do not immediately answer zero. Inspect relevant tables, distinct status/region values, date formats/ranges, and join keys. Check whether the date is on the order or refund record, whether strings need case-insensitive matching, and whether the join actually matches records. Only return zero after validating these points.
- Use case-insensitive comparisons for categorical values when appropriate (for example lower(trim(region)) = 'north'), while preserving exact product names as the API spells them.
- Do not confuse gross revenue, net revenue, order value, and refunds. Subtract refunds only when the question asks for net revenue / revenue after refunds; report refunds separately when asked.
- Use USD amounts and round money to two decimal places. Counts are integers. Use SQL for calculations instead of mental arithmetic.
- Do not invent missing fields or guess. If there are several candidate columns, inspect samples and use the field whose meaning fits the question.
- Final output MUST be a JSON object with one key named answer. Use a number for numeric answers, or a string for names/text. Do not include explanation or Markdown in the final answer.
"""


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


def _client() -> OpenAI:
    if not AI_TOKEN:
        raise RuntimeError("Set AIPROXY_TOKEN or OPENAI_API_KEY in the service environment")
    return OpenAI(api_key=AI_TOKEN, base_url=AI_BASE_URL, timeout=3.5, max_retries=0)


def _decode_final(content: str) -> Any:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]
        if isinstance(parsed, (str, int, float, bool)) or parsed is None:
            return parsed
        return parsed
    except Exception:
        # The system prompt asks for JSON. Preserve a plain-text response if provider
        # does not honor the format, instead of returning an HTML/error page.
        return content


def answer_question(question: str) -> Any:
    if not QUESTION_DATA_URL:
        raise RuntimeError("Set QUESTION_DATA_URL to the full personalized questionData URL")
    if not _bootstrap_done:
        bootstrap()
    client = _client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nCURRENT DATA SCHEMA AND SAMPLES:\n" + _schema_context()},
        {"role": "user", "content": question},
    ]
    api_calls = 0
    for _ in range(3):
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return _decode_final(message.content or "")
        messages.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                name = tool_call.function.name
                if name == "list_tables":
                    result = _list_tables()
                elif name == "query_sql":
                    result = _run_sql(str(args.get("sql", "")))
                elif name == "api_get":
                    api_calls += 1
                    if api_calls > MAX_API_CALLS:
                        result = {"error": f"API exploration cap ({MAX_API_CALLS}) reached. Use the discovered schema and tables."}
                    else:
                        result = _tool_api_get(str(args.get("path", "/")))
                else:
                    result = {"error": "Unknown tool"}
            except Exception as exc:
                result = {"error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, ensure_ascii=False, default=str)[:16000]})
    # One final no-tools response after the bounded tool loop.
    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=messages + [{"role": "system", "content": "Now provide the final JSON object only: {\"answer\": ...}."}],
        temperature=0,
    )
    return _decode_final(response.choices[0].message.content or "")


@app.get("/")
def health_get() -> dict[str, Any]:
    return {"status": "ok", "service": "Ledger Agent", "post_endpoint": "/", "tables_loaded": len(_table_columns)}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "tables_loaded": len(_table_columns), "tables": _list_tables()["tables"]}


@app.post("/")
def answer_root(body: QuestionRequest) -> dict[str, Any]:
    try:
        answer = answer_question(body.question)
        return {"answer": answer}
    except Exception as exc:
        # Keep errors structured for debugging. Grader should receive answer after
        # the environment variables and discovery setup are correct.
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/answer")
def answer_alias(body: QuestionRequest) -> dict[str, Any]:
    return answer_root(body)


@app.on_event("startup")
def startup() -> None:
    # Preload data before test requests arrive. Errors are handled lazily on the first
    # question as well, so a transient source failure does not prevent server startup.
    try:
        bootstrap()
    except Exception:
        pass
