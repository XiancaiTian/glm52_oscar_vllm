# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import itertools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
                "long_prefill_semaphore": asyncio.Semaphore(_long_prefill_limit()),
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def _long_prefill_limit() -> int:
    return max(1, int(os.environ.get("PREFILL_PROXY_LONG_PREFILL_LIMIT", "1")))


def _prefill_route_policy() -> str:
    return os.environ.get(
        "PREFILL_PROXY_LONG_PREFILL_ROUTE_POLICY", "least_available"
    ).strip().lower()


def _decode_first_chunk_timeout_s() -> float:
    raw = os.environ.get("PREFILL_PROXY_DECODE_FIRST_CHUNK_TIMEOUT_S", "900")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 900.0


class PrefillUnavailable(Exception):

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 503,
        prefill: str | None = None,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.prefill = prefill
        self.details = details or {}


def _context_length_guard_enabled() -> bool:
    return os.environ.get(
        "PREFILL_PROXY_CONTEXT_LENGTH_GUARD", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _max_model_len() -> int:
    raw = os.environ.get("PREFILL_PROXY_MAX_MODEL_LEN", "202752")
    try:
        return max(1, int(raw))
    except ValueError:
        return 202752


def _context_length_safety_margin() -> int:
    raw = os.environ.get("PREFILL_PROXY_CONTEXT_LENGTH_SAFETY_MARGIN", "1024")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1024


def _coerce_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _output_token_fields(req_data: dict) -> list[tuple[str, int]]:
    fields: list[tuple[str, int]] = []
    for key in ("max_tokens", "max_completion_tokens"):
        value = _coerce_int(req_data.get(key))
        if value is not None:
            fields.append((key, value))
    return fields


def _system_content_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _normalize_anthropic_system_messages(
    req_data: dict,
    request_id: str,
    api: str,
) -> dict:
    messages = req_data.get("messages")
    if not isinstance(messages, list):
        return req_data

    normalized_messages = []
    moved_system_parts: list[str] = []
    moved = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            moved += 1
            text = _system_content_to_text(message.get("content"))
            if text:
                moved_system_parts.append(text)
            continue
        normalized_messages.append(message)

    if moved == 0:
        return req_data

    normalized = req_data.copy()
    normalized["messages"] = normalized_messages
    existing_system = _system_content_to_text(normalized.get("system"))
    system_parts = [part for part in [existing_system, *moved_system_parts] if part]
    if system_parts:
        normalized["system"] = "\n\n".join(system_parts)
    else:
        normalized.pop("system", None)
    print(
        "PROXY_ANTHROPIC_SYSTEM_NORMALIZED "
        f"request_id={request_id} api={api} moved_system_messages={moved}",
        flush=True,
    )
    return normalized


def _context_input_too_long_response(
    *,
    request_id: str,
    api: str,
    input_tokens: int,
    max_model_len: int,
    safety_margin: int,
) -> JSONResponse:
    allowed_output = max_model_len - input_tokens - safety_margin
    message = (
        f"This model's maximum context length is {max_model_len} tokens. "
        f"Your prompt contains {input_tokens} input tokens and the proxy "
        f"reserves {safety_margin} safety tokens, leaving no output budget. "
        "Please reduce the length of the input prompt."
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "input_tokens",
            },
            "details": {
                "request_id": request_id,
                "api": api,
                "input_tokens": input_tokens,
                "allowed_output": allowed_output,
                "max_model_len": max_model_len,
                "safety_margin": safety_margin,
                "guard": "proxy_pre_prefill",
            },
        },
    )


def _context_precheck_error_response(
    *,
    request_id: str,
    api: str,
    status_code: int,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    error_type = (
        "invalid_request_error"
        if 400 <= status_code < 500
        else "service_unavailable"
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": "context_length_precheck_failed",
            },
            "details": {
                "request_id": request_id,
                "api": api,
                **(details or {}),
            },
        },
    )


def _extract_upstream_error_message(body, raw_text: str) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("message", "detail", "reason", "code"):
                value = error.get(key)
                if value:
                    return str(value)
        elif error:
            return str(error)
        for key in ("message", "detail", "reason"):
            value = body.get(key)
            if value:
                return str(value)
    raw = (raw_text or "").strip()
    if raw:
        return raw[:1024]
    return "unknown decode error"


async def _count_messages_tokens_with_decode(
    request: Request,
    req_data: dict,
    request_id: str,
) -> tuple[int | None, JSONResponse | None]:
    decode_client_info = get_next_client(request.app, "decode")
    payload = req_data.copy()
    payload.pop("stream", None)
    payload.pop("stream_options", None)
    headers = {
        "Authorization": request.headers.get(
            "authorization", f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        ),
        "X-Request-Id": request_id,
    }
    response = await decode_client_info["client"].post(
        "/messages/count_tokens", json=payload, headers=headers
    )
    if response.status_code >= 400:
        text = response.text
        try:
            body = response.json()
        except Exception:
            body = {"raw": text[:4096]}
        upstream_message = _extract_upstream_error_message(body, text)
        print(
            "PROXY_CONTEXT_LENGTH_COUNT_ERROR "
            f"request_id={request_id} status_code={response.status_code} "
            f"decode={decode_client_info['host']}:{decode_client_info['port']} "
            f"body={text[:4096]!r}",
            flush=True,
        )
        return None, _context_precheck_error_response(
            request_id=request_id,
            api="/messages",
            status_code=400,
            message=(
                "context length precheck failed before prefill: "
                f"{upstream_message}"
            ),
            details={
                "upstream_status_code": response.status_code,
                "upstream_message": upstream_message,
                "upstream_body": body,
                "decode": f"{decode_client_info['host']}:{decode_client_info['port']}",
                "guard": "proxy_pre_prefill",
            },
        )
    try:
        body = response.json()
    except Exception as exc:
        return None, _context_precheck_error_response(
            request_id=request_id,
            api="/messages",
            status_code=503,
            message="context length precheck returned invalid JSON before prefill",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
    input_tokens = body.get("input_tokens")
    if not isinstance(input_tokens, int):
        return None, _context_precheck_error_response(
            request_id=request_id,
            api="/messages",
            status_code=503,
            message="context length precheck did not return input_tokens",
            details={"upstream_body": body, "guard": "proxy_pre_prefill"},
        )
    return input_tokens, None


async def _prevalidate_context_length(
    api: str,
    request: Request,
    req_data: dict,
    request_id: str,
) -> JSONResponse | None:
    if not _context_length_guard_enabled() or api != "/messages":
        return None
    output_fields = _output_token_fields(req_data)
    if not output_fields:
        return None
    input_tokens, error_response = await _count_messages_tokens_with_decode(
        request, req_data, request_id
    )
    if error_response is not None:
        return error_response
    assert input_tokens is not None
    max_model_len = _max_model_len()
    safety_margin = _context_length_safety_margin()
    allowed_output = max_model_len - input_tokens - safety_margin
    if allowed_output <= 0:
        print(
            "PROXY_CONTEXT_LENGTH_REJECT "
            f"request_id={request_id} api={api} input_tokens={input_tokens} "
            f"allowed_output={allowed_output} max_model_len={max_model_len} "
            f"safety_margin={safety_margin} guard=proxy_pre_prefill",
            flush=True,
        )
        return _context_input_too_long_response(
            request_id=request_id,
            api=api,
            input_tokens=input_tokens,
            max_model_len=max_model_len,
            safety_margin=safety_margin,
        )

    clipped_fields = []
    for field, requested_tokens in output_fields:
        if requested_tokens > allowed_output:
            req_data[field] = allowed_output
            clipped_fields.append((field, requested_tokens))

    if not clipped_fields:
        return None

    for field, requested_tokens in clipped_fields:
        print(
            "PROXY_CONTEXT_LENGTH_CLAMP "
            f"request_id={request_id} api={api} field={field} "
            f"input_tokens={input_tokens} requested_output={requested_tokens} "
            f"allowed_output={allowed_output} max_model_len={max_model_len} "
            f"safety_margin={safety_margin} guard=proxy_pre_prefill",
            flush=True,
        )
    return None


def _task_root() -> Path:
    configured = os.environ.get("TASK_ROOT")
    candidates = [
        configured,
        "/workspace/glm52_speed_up_v2_stable_8th",
        "/nfs/AE/zhanghong/workflow/vllm_a/glm52_speed_up_v2_stable_8th",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return Path(candidates[-1])


def _prefill_status_path() -> Path:
    configured = os.environ.get("PREFILL_STATUS_FILE")
    if configured:
        return Path(configured)
    return _task_root() / "state" / "watchdog" / "prefill_status.json"


def _internal_prefill_warmup_allowed(request: Request) -> bool:
    token = os.environ.get("PREFILL_PROXY_INTERNAL_TOKEN", "").strip()
    if not token:
        return False
    provided = request.headers.get("x-prefill-warmup-token", "").strip()
    return bool(provided) and provided == token


def _prefill_key(client_info: dict) -> str:
    return f"{client_info['host']}:{client_info['port']}"


def _load_prefill_status() -> dict:
    path = _prefill_status_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"prefills": {}, "source": str(path), "missing": True}
    except Exception as exc:
        print(
            "PROXY_PREFILL_STATUS_READ_ERROR "
            f"path={path} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return {"prefills": {}, "source": str(path), "read_error": str(exc)}
    if not isinstance(data, dict):
        return {"prefills": {}, "source": str(path), "invalid": True}
    data.setdefault("prefills", {})
    data["source"] = str(path)
    return data


def _status_for_prefill(client_info: dict, status_data: dict) -> dict:
    key = _prefill_key(client_info)
    prefills = status_data.get("prefills") or {}
    status = prefills.get(key) or prefills.get(client_info["host"]) or {}
    if not isinstance(status, dict):
        status = {"status": str(status)}
    status = status.copy()
    status.setdefault("status", "healthy")
    return status


def _prefill_is_unavailable(status: dict) -> bool:
    state = str(status.get("status", "healthy")).strip().lower()
    return state in {
        "unavailable",
        "quarantined",
        "restarting",
        "warming",
        "warmup",
        "recovering",
        "failed",
        "disabled",
    }


def _prefill_is_internal_warmup_state(status: dict) -> bool:
    state = str(status.get("status", "healthy")).strip().lower()
    return state in {"warming", "warmup", "recovering"}


def _status_error_code(status: dict) -> str:
    state = str(status.get("status", "")).strip().lower()
    code = str(status.get("error_code", "")).strip()
    if code:
        return code
    if state in {"restarting", "warming", "warmup", "recovering"}:
        return "prefill_node_restarting"
    if state in {"quarantined", "disabled"}:
        return "prefill_path_unavailable"
    return "current_prefill_node_unavailable"


def _prefill_error_response(exc: PrefillUnavailable) -> JSONResponse:
    body = {
        "error": {
            "message": exc.message,
            "type": "prefill_unavailable",
            "code": exc.code,
        }
    }
    if exc.prefill:
        body["error"]["prefill"] = exc.prefill
    if exc.details:
        body["details"] = exc.details
    return JSONResponse(status_code=exc.status_code, content=body)


def _service_error_response(
    code: str,
    message: str,
    *,
    status_code: int = 503,
    details: dict | None = None,
) -> JSONResponse:
    body = {"error": {"message": message, "type": "service_unavailable", "code": code}}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def _request_text_size(req_data: dict) -> int:
    messages = req_data.get("messages") or []
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", "")))
                else:
                    total += len(str(item))
        else:
            total += len(str(content))
    prompt = req_data.get("prompt")
    if isinstance(prompt, str):
        total += len(prompt)
    elif isinstance(prompt, list):
        total += sum(len(str(x)) for x in prompt)
    return total


def _is_long_prefill(req_data: dict) -> bool:
    threshold = int(os.environ.get("PREFILL_PROXY_MIN_DISAGG_CHARS", "2048"))
    return _request_text_size(req_data) >= threshold


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


def _semaphore_available_slots(semaphore: asyncio.Semaphore) -> int:
    # asyncio.Semaphore intentionally keeps this private; for this proxy it is
    # the least invasive way to avoid routing long prefills to a busy node.
    return max(0, int(getattr(semaphore, "_value", 0)))


def _requested_prefill_key(request: Request, req_data: dict) -> str | None:
    requested = (
        request.headers.get("x-prefill-target")
        or request.headers.get("x-prefill-node")
        or req_data.pop("_prefill_target", None)
        or req_data.pop("prefill_target", None)
    )
    if requested is None:
        return None
    return str(requested).strip()


def _healthy_prefill_clients(app, status_data: dict) -> tuple[list[dict], list[dict]]:
    healthy = []
    unavailable = []
    for client_info in app.state.prefill_clients:
        status = _status_for_prefill(client_info, status_data)
        row = {"client_info": client_info, "status": status}
        if _prefill_is_unavailable(status):
            unavailable.append(row)
        else:
            healthy.append(client_info)
    return healthy, unavailable


def _choose_round_robin_healthy(app, healthy_clients: list[dict]) -> dict:
    clients = app.state.prefill_clients
    healthy_keys = {_prefill_key(client) for client in healthy_clients}
    for _ in range(len(clients)):
        client_idx = next(app.state.prefill_iterator)
        candidate = clients[client_idx]
        if _prefill_key(candidate) in healthy_keys:
            return candidate
        print(
            "PROXY_PREFILL_SKIP_UNAVAILABLE "
            f"prefill={_prefill_key(candidate)} reason=status_file",
            flush=True,
        )
    return healthy_clients[0]


def get_prefill_client(
    request: Request, long_prefill: bool, requested_prefill: str | None = None
):
    status_data = _load_prefill_status()
    clients = request.app.state.prefill_clients
    healthy_clients, unavailable = _healthy_prefill_clients(request.app, status_data)

    if requested_prefill:
        normalized_requested = requested_prefill
        if ":" not in normalized_requested:
            for client_info in clients:
                if client_info["host"] == normalized_requested:
                    normalized_requested = _prefill_key(client_info)
                    break
        for client_info in clients:
            if _prefill_key(client_info) == normalized_requested:
                status = _status_for_prefill(client_info, status_data)
                if _prefill_is_unavailable(status):
                    if (
                        _internal_prefill_warmup_allowed(request)
                        and _prefill_is_internal_warmup_state(status)
                    ):
                        print(
                            "PROXY_PREFILL_INTERNAL_WARMUP_ALLOWED "
                            f"prefill={_prefill_key(client_info)} "
                            f"status={status.get('status')} "
                            f"reason={status.get('reason', '')}",
                            flush=True,
                        )
                        return client_info
                    key = _prefill_key(client_info)
                    code = _status_error_code(status)
                    if code == "prefill_path_unavailable":
                        code = "current_prefill_node_unavailable"
                    raise PrefillUnavailable(
                        code,
                        f"current prefill node is unavailable: {key}",
                        prefill=key,
                        details={"status": status, "status_file": status_data["source"]},
                    )
                return client_info
        raise PrefillUnavailable(
            "current_prefill_node_unavailable",
            f"requested prefill node is not registered: {requested_prefill}",
            prefill=requested_prefill,
            details={"status_file": status_data["source"]},
        )

    if not healthy_clients:
        details = {
            "status_file": status_data["source"],
            "unavailable_prefills": [
                {
                    "prefill": _prefill_key(row["client_info"]),
                    "status": row["status"],
                }
                for row in unavailable
            ],
        }
        raise PrefillUnavailable(
            "no_healthy_prefill_path",
            "no healthy prefill path is available",
            details=details,
        )

    if not long_prefill or _prefill_route_policy() not in {
        "least_available",
        "least_busy",
        "available",
    }:
        return _choose_round_robin_healthy(request.app, healthy_clients)

    if len(healthy_clients) <= 1:
        return healthy_clients[0]

    start_idx = next(request.app.state.prefill_iterator)
    ordered_clients = [
        clients[(start_idx + offset) % len(clients)] for offset in range(len(clients))
        if _prefill_key(clients[(start_idx + offset) % len(clients)])
        in {_prefill_key(client) for client in healthy_clients}
    ]
    return max(
        ordered_clients,
        key=lambda client_info: _semaphore_available_slots(
            client_info["long_prefill_semaphore"]
        ),
    )


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    # These args are not supported for P
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    # read/consume the response body to release the connection
    # otherwise, it would http.ReadError
    await response.aread()

    # Add back the min_tokens and min_completion_tokens so D can use them
    req_data["min_tokens"] = min_tokens
    req_data["min_completion_tokens"] = min_completion_tokens

    return response


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        chunk_iter = response.aiter_bytes().__aiter__()
        first_chunk = True
        timeout_s = _decode_first_chunk_timeout_s()
        while True:
            try:
                if first_chunk and timeout_s > 0:
                    chunk = await asyncio.wait_for(
                        chunk_iter.__anext__(), timeout=timeout_s
                    )
                else:
                    chunk = await chunk_iter.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                print(
                    "PROXY_STREAM_TIMEOUT "
                    f"request_id={request_id} endpoint={endpoint} "
                    f"timeout_s={timeout_s} "
                    f"decode={client_info['host']}:{client_info['port']}",
                    flush=True,
                )
                raise
            first_chunk = False
            yield chunk


def _passthrough_headers(request: Request) -> dict:
    headers = {
        "Authorization": request.headers.get(
            "authorization", f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        )
    }
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    request_id = request.headers.get("x-request-id")
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


async def _decode_v1_passthrough(endpoint: str, request: Request):
    decode_client_info = get_next_client(request.app, "decode")
    headers = _passthrough_headers(request)
    method = request.method

    if method in ("GET", "DELETE"):
        response = await decode_client_info["client"].request(
            method, endpoint, headers=headers
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    req_data = await request.json()
    if endpoint == "/messages/count_tokens" and isinstance(req_data, dict):
        request_id = str(uuid.uuid4())
        req_data = _normalize_anthropic_system_messages(
            req_data, request_id, endpoint
        )
    if req_data.get("stream"):

        async def generate_stream():
            async with decode_client_info["client"].stream(
                method, endpoint, json=req_data, headers=headers
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    response = await decode_client_info["client"].request(
        method, endpoint, json=req_data, headers=headers
    )
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _decode_root_passthrough(endpoint: str, request: Request):
    decode_client_info = get_next_client(request.app, "decode")
    headers = _passthrough_headers(request)
    url = f"http://{decode_client_info['host']}:{decode_client_info['port']}{endpoint}"
    body = await request.body()
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.request(
            request.method,
            url,
            headers=headers,
            content=body if body else None,
        )
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _decode_prefixed_v1_passthrough(endpoint: str, request: Request):
    return await _decode_v1_passthrough(endpoint, request)


async def _handle_completions(api: str, request: Request):
    try:
        t0 = time.perf_counter()
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        if api == "/messages":
            req_data = _normalize_anthropic_system_messages(
                req_data, request_id, api
            )
        max_tokens = req_data.get("max_tokens", req_data.get("max_completion_tokens"))
        text_chars = _request_text_size(req_data)
        long_prefill = _is_long_prefill(req_data)
        requested_prefill = _requested_prefill_key(request, req_data)
        context_error = await _prevalidate_context_length(
            api, request, req_data, request_id
        )
        if context_error is not None:
            return context_error

        # Prefer an idle Prefill node for long prompts. Plain round-robin can
        # queue a request behind an already-running long prefill while another
        # Prefill node is idle, and that wait used to be reported as prefill_ms.
        prefill_client_info = get_prefill_client(
            request, long_prefill, requested_prefill
        )

        # Send request to prefill service
        prefill_queue_ms = 0.0
        t_prefill_rpc_start = time.perf_counter()
        if long_prefill:
            semaphore = prefill_client_info["long_prefill_semaphore"]
            t_prefill_wait_start = time.perf_counter()
            await semaphore.acquire()
            t_prefill_rpc_start = time.perf_counter()
            prefill_queue_ms = (t_prefill_rpc_start - t_prefill_wait_start) * 1000.0
            try:
                response = await send_request_to_service(
                    prefill_client_info, api, req_data, request_id
                )
            finally:
                semaphore.release()
        else:
            response = await send_request_to_service(
                prefill_client_info, api, req_data, request_id
            )
        t_prefill_done = time.perf_counter()
        prefill_rpc_ms = (t_prefill_done - t_prefill_rpc_start) * 1000.0

        # Extract the needed fields
        response_json = response.json()
        await response.aclose()  # CRITICAL: Release connection back to pool
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            t_decode_start = time.perf_counter()
            first_chunk = True
            chunk_count = 0
            try:
                async for chunk in stream_service_response(
                    decode_client_info, api, req_data, request_id=request_id
                ):
                    chunk_count += 1
                    if first_chunk:
                        first_chunk = False
                        t_first_chunk = time.perf_counter()
                        print(
                            "PROXY_TIMING "
                            f"request_id={request_id} api={api} "
                            f"max_tokens={max_tokens} "
                            f"text_chars={text_chars} "
                            f"long_prefill={long_prefill} "
                            f"prefill_ms={(t_prefill_done - t0) * 1000:.3f} "
                            f"prefill_queue_ms={prefill_queue_ms:.3f} "
                            f"prefill_rpc_ms={prefill_rpc_ms:.3f} "
                            f"decode_first_ms={(t_first_chunk - t_decode_start) * 1000:.3f} "
                            f"ttft_proxy_ms={(t_first_chunk - t0) * 1000:.3f} "
                            f"prefill_route_policy={_prefill_route_policy()} "
                            f"prefill_id={prefill_client_info['id']} "
                            f"prefill={prefill_client_info['host']}:{prefill_client_info['port']} "
                            f"decode={decode_client_info['host']}:{decode_client_info['port']}",
                            flush=True,
                        )
                    yield chunk
            except Exception as e:
                t_error = time.perf_counter()
                print(
                    "PROXY_STREAM_ERROR "
                    f"request_id={request_id} api={api} "
                    f"elapsed_ms={(t_error - t0) * 1000:.3f} "
                    f"chunks={chunk_count} error={type(e).__name__}: {e}",
                    flush=True,
                )
                raise
            if chunk_count == 0:
                t_empty = time.perf_counter()
                print(
                    "PROXY_STREAM_EMPTY "
                    f"request_id={request_id} api={api} "
                    f"elapsed_ms={(t_empty - t0) * 1000:.3f}",
                    flush=True,
                )
            t_done = time.perf_counter()
            print(
                "PROXY_DONE "
                f"request_id={request_id} api={api} "
                f"total_ms={(t_done - t0) * 1000:.3f} chunks={chunk_count}",
                flush=True,
            )

        return StreamingResponse(generate_stream(), media_type="application/json")

    except PrefillUnavailable as e:
        print(
            "PROXY_PREFILL_UNAVAILABLE "
            f"api={api} code={e.code} prefill={e.prefill} message={e.message}",
            flush=True,
        )
        return _prefill_error_response(e)
    except httpx.TimeoutException as e:
        print(
            "PROXY_UPSTREAM_TIMEOUT "
            f"api={api} error={type(e).__name__}: {e}",
            flush=True,
        )
        return _service_error_response(
            "prefill_path_unavailable",
            "prefill path timed out",
            details={"error_type": type(e).__name__, "error": str(e)},
        )
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else None
        print(
            "PROXY_UPSTREAM_HTTP_ERROR "
            f"api={api} status_code={status_code} error={type(e).__name__}: {e}",
            flush=True,
        )
        return _service_error_response(
            "p2d_transfer_failed" if status_code and status_code >= 500 else "kv_transfer_failed",
            "prefill to decode transfer failed",
            status_code=503,
            details={"upstream_status_code": status_code, "error": str(e)},
        )
    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/v1/models")
async def handle_models(request: Request):
    return await _decode_v1_passthrough("/models", request)


@app.post("/v1/messages")
async def handle_messages(request: Request):
    return await _handle_completions("/messages", request)


@app.post("/v1/messages/count_tokens")
async def handle_messages_count_tokens(request: Request):
    return await _decode_v1_passthrough("/messages/count_tokens", request)


@app.post("/v1/chat/completions/batch")
async def handle_chat_completions_batch(request: Request):
    return await _decode_prefixed_v1_passthrough("/chat/completions/batch", request)


@app.post("/v1/chat/completions/render")
async def handle_chat_completions_render(request: Request):
    return await _decode_prefixed_v1_passthrough("/chat/completions/render", request)


@app.post("/v1/completions/render")
async def handle_completions_render(request: Request):
    return await _decode_prefixed_v1_passthrough("/completions/render", request)


@app.post("/v1/responses")
async def handle_responses(request: Request):
    return await _decode_prefixed_v1_passthrough("/responses", request)


@app.get("/v1/responses/{response_id}")
async def handle_response_get(response_id: str, request: Request):
    return await _decode_prefixed_v1_passthrough(f"/responses/{response_id}", request)


@app.post("/v1/responses/{response_id}/cancel")
async def handle_response_cancel(response_id: str, request: Request):
    return await _decode_prefixed_v1_passthrough(f"/responses/{response_id}/cancel", request)


@app.get("/health")
async def handle_health(request: Request):
    return await _decode_root_passthrough("/health", request)


@app.get("/metrics")
async def handle_metrics(request: Request):
    return await _decode_root_passthrough("/metrics", request)


@app.get("/version")
async def handle_version(request: Request):
    return await _decode_root_passthrough("/version", request)


@app.get("/ping")
async def handle_ping_get(request: Request):
    return await _decode_root_passthrough("/ping", request)


@app.post("/ping")
async def handle_ping_post(request: Request):
    return await _decode_root_passthrough("/ping", request)


@app.get("/load")
async def handle_load(request: Request):
    return await _decode_root_passthrough("/load", request)


@app.post("/tokenize")
async def handle_tokenize(request: Request):
    return await _decode_root_passthrough("/tokenize", request)


@app.post("/detokenize")
async def handle_detokenize(request: Request):
    return await _decode_root_passthrough("/detokenize", request)


@app.post("/generative_scoring")
async def handle_generative_scoring(request: Request):
    return await _decode_root_passthrough("/generative_scoring", request)


@app.post("/is_scaling_elastic_ep")
async def handle_is_scaling_elastic_ep(request: Request):
    return await _decode_root_passthrough("/is_scaling_elastic_ep", request)


@app.post("/scale_elastic_ep")
async def handle_scale_elastic_ep(request: Request):
    return await _decode_root_passthrough("/scale_elastic_ep", request)


@app.post("/invocations")
async def handle_invocations(request: Request):
    return await _decode_root_passthrough("/invocations", request)


@app.post("/inference/v1/generate")
async def handle_inference_v1_generate(request: Request):
    return await _decode_root_passthrough("/inference/v1/generate", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    status_data = _load_prefill_status()
    prefill_status = {
        _prefill_key(client_info): _status_for_prefill(client_info, status_data)
        for client_info in app.state.prefill_clients
    }
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
        "prefill_status": prefill_status,
        "prefill_status_file": status_data.get("source"),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
