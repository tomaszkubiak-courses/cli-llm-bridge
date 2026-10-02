"""OpenAI- and Anthropic-compatible local HTTP API backed by the Claude Code CLI.

Every request starts a fresh, stateless `claude -p` process. The client sends the
full conversation each time (as with any chat completions API), so the bridge keeps
no state between requests. Anthropic Messages requests are translated to the OpenAI
shape, handled by the same core, and translated back.
"""

import argparse
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

__version__ = "0.1.0"

# Model aliases understood by the Claude Code CLI. Full model names also work.
MODELS = ["sonnet", "opus", "haiku", "fable"]
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
DEFAULT_SYSTEM = "You are a helpful assistant."
# Above this size the tool-call schema is sent in compact form. Windows caps a whole
# command line at 32,767 characters, and Linux caps a single argument at 128 KiB.
MAX_SCHEMA_CHARS = 16000
TRANSCRIPT_HEADER = (
    "Below is the conversation so far. Write only the next assistant reply, "
    "without any [role] prefix."
)
TOOLS_HEADER = (
    "The functions listed below are not tools you can call. Your only tool is the one that "
    "submits your reply (StructuredOutput); never call a function from this list directly.\n"
    "The caller can run these functions for you. To request one, put it in the tool_calls "
    "field of your reply with the function name and arguments, and keep content short or "
    "empty. The caller runs it and sends the result back in a later message. Never say that "
    "a function is unavailable: requesting it through tool_calls is how you use it.\n"
    "When you can answer without a function, leave tool_calls empty and write the answer in content.\n\n"
    "Functions:"
)


class BridgeError(Exception):
    def __init__(self, status, message, kind="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


def text_of(content):
    """Return the text of an OpenAI message `content` (a string or a list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if part.get("type") != "text":
            raise BridgeError(400, f"Unsupported content part type: {part.get('type')!r}. Only text is supported.")
        texts.append(part.get("text", ""))
    return "\n".join(texts)


def build_prompt(messages):
    """Split OpenAI messages into (system prompt, user prompt)."""
    system, turns = [], []
    for m in messages:
        role = m.get("role")
        text = text_of(m.get("content"))
        if role in ("system", "developer"):
            system.append(text)
        elif role == "user":
            turns.append(("user", text))
        elif role == "assistant":
            parts = [text] if text else []
            for call in m.get("tool_calls") or []:
                f = call.get("function", {})
                parts.append(f"[tool call {call.get('id', '')}] {f.get('name')}({f.get('arguments', '')})")
            turns.append(("assistant", "\n".join(parts)))
        elif role == "tool":
            turns.append((f"tool result {m.get('tool_call_id', '')}", text))
        else:
            raise BridgeError(400, f"Unsupported message role: {role!r}")
    if not turns:
        raise BridgeError(400, "At least one non-system message is required.")
    if len(turns) == 1 and turns[0][0] == "user":
        prompt = turns[0][1]
    else:
        prompt = TRANSCRIPT_HEADER + "\n\n" + "\n\n".join(f"[{who}]\n{text}" for who, text in turns)
    return "\n\n".join(system) or DEFAULT_SYSTEM, prompt


def select_tools(tools, tool_choice):
    """Apply `tool_choice` to the tool list. Returns (tools, at_least_one_call_required)."""
    if not tools or tool_choice == "none":
        return [], False
    if tool_choice == "required":
        return tools, True
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        chosen = [t for t in tools if t["function"]["name"] == name]
        if not chosen:
            raise BridgeError(400, f"tool_choice names an unknown tool: {name!r}")
        return chosen, True
    return tools, False


def tools_schema(tools, required):
    """JSON schema that forces the reply into {content, tool_calls} with valid tool arguments.

    The schema travels on the command line, so a large tool set falls back to a compact
    schema that only checks tool names; the parameters are still in the system prompt.
    """
    variants = [
        {
            "type": "object",
            "properties": {
                "name": {"const": t["function"]["name"]},
                "arguments": t["function"].get("parameters") or {"type": "object"},
            },
            "required": ["name", "arguments"],
        }
        for t in tools
    ]
    items = {"anyOf": variants}
    if len(json.dumps(items)) > MAX_SCHEMA_CHARS:
        items = {
            "type": "object",
            "properties": {"name": {"enum": [t["function"]["name"] for t in tools]}, "arguments": {"type": "object"}},
            "required": ["name", "arguments"],
        }
    calls = {"type": "array", "items": items}
    if required:
        calls["minItems"] = 1
    return {
        "type": "object",
        "properties": {"content": {"type": "string"}, "tool_calls": calls},
        "required": ["content", "tool_calls"],
    }


def tools_prompt(tools):
    lines = [TOOLS_HEADER]
    for t in tools:
        f = t["function"]
        lines.append(f"- {f['name']}: {f.get('description', '')}")
        lines.append(f"  parameters: {json.dumps(f.get('parameters') or {})}")
    return "\n".join(lines)


def child_env(config):
    """Environment for `claude`, minus an ANTHROPIC_BASE_URL that points back at this bridge."""
    env = dict(os.environ)
    url = env.get("ANTHROPIC_BASE_URL")
    if url:
        parts = urlsplit(url)
        if parts.port == config.port and parts.hostname in {config.host, "localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            del env["ANTHROPIC_BASE_URL"]
    return env


def run_claude(config, system, prompt, model, effort, schema):
    """Run one stateless `claude -p` call and return its parsed JSON result."""
    cmd = [
        config.claude, "-p",
        "--output-format", "json",
        "--no-session-persistence",
        # Keep the user's CLAUDE.md, hooks, plugins and MCP servers out of the request.
        "--safe-mode",
        # Plain model access: the model cannot run commands or touch files.
        "--tools", "",
    ]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    # An empty temp dir as cwd: nothing project-specific is picked up, and the
    # system prompt goes through a file to stay clear of command-line length limits.
    with tempfile.TemporaryDirectory(prefix="cli-llm-bridge-") as tmp:
        system_file = Path(tmp, "system.txt")
        system_file.write_text(system, encoding="utf-8")
        cmd += ["--system-prompt-file", str(system_file)]
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=tmp, env=child_env(config),
                timeout=config.timeout,
            )
        except subprocess.TimeoutExpired:
            raise BridgeError(504, f"Claude Code did not answer within {config.timeout} s.", "timeout")
        except OSError as e:
            raise BridgeError(502, f"Could not start Claude Code: {e}", "upstream_error")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout).strip()[-1000:]
        raise BridgeError(502, f"Claude Code failed (exit code {proc.returncode}): {detail}", "upstream_error")
    if result.get("is_error"):
        status = result.get("api_error_status")
        status = status if isinstance(status, int) and 400 <= status < 600 else 502
        raise BridgeError(status, f"Claude Code error: {result.get('result')}", "upstream_error")
    return result


def complete(config, body, run=run_claude):
    """Handle one chat completions request body and return the OpenAI response object."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise BridgeError(400, "'messages' must be a list.")
    system, prompt = build_prompt(messages)
    model = body.get("model")
    model = config.default_model if model in (None, "", "default") else model
    effort = body.get("reasoning_effort")
    effort = effort if effort in EFFORTS else None

    tools, required = select_tools(body.get("tools"), body.get("tool_choice"))
    fmt = body.get("response_format") or {}
    schema = None
    if tools:
        schema = tools_schema(tools, required)
        system += "\n\n" + tools_prompt(tools)
    elif fmt.get("type") == "json_schema":
        schema = fmt.get("json_schema", {}).get("schema") or {"type": "object"}
    elif fmt.get("type") == "json_object":
        schema = {"type": "object"}

    result = run(config, system, prompt, model, effort, schema)

    message = {"role": "assistant", "content": result.get("result", "")}
    finish = "stop"
    if schema is not None:
        output = result.get("structured_output")
        if output is None:
            raise BridgeError(502, "Claude Code returned no structured output.", "upstream_error")
        if tools:
            calls = output.get("tool_calls") or []
            message["content"] = output.get("content") or None
            if calls:
                message["tool_calls"] = [
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))},
                    }
                    for c in calls
                ]
                finish = "tool_calls"
        else:
            message["content"] = json.dumps(output)

    usage = result.get("usage") or {}
    prompt_tokens = sum(usage.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    completion_tokens = usage.get("output_tokens", 0)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "default",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream_chunks(response, include_usage):
    """Turn a finished response into chat.completion.chunk objects (one content chunk)."""
    choice = response["choices"][0]
    message = choice["message"]
    delta = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(message["tool_calls"])]
    base = {k: response[k] for k in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    yield dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])
    yield dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}])
    if include_usage:
        yield dict(base, choices=[], usage=response["usage"])


def anthropic_to_openai(body):
    """Translate an Anthropic Messages request body into a chat completions body."""
    if not isinstance(body.get("messages"), list):
        raise BridgeError(400, "'messages' must be a list.")
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": text_of(body["system"])})
    for m in body["messages"]:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        texts, calls = [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                texts.append(block.get("text", ""))
            elif kind == "tool_use":
                calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input", {}))},
                })
            elif kind == "tool_result":
                result = text_of(block.get("content"))
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": f"Error: {result}" if block.get("is_error") else result,
                })
            elif kind in ("thinking", "redacted_thinking"):
                continue  # earlier reasoning; the CLI produces its own
            else:
                raise BridgeError(400, f"Unsupported content block type: {kind!r}. Only text, tool_use and tool_result are supported.")
        if texts or calls:
            message = {"role": role, "content": "\n".join(texts)}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)

    tools = []
    for t in body.get("tools") or []:
        if t.get("type") not in (None, "custom"):
            raise BridgeError(400, f"Unsupported tool type: {t.get('type')!r}. Only client tools are supported.")
        tools.append({"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema")}})
    choice = body.get("tool_choice") or {}
    tool_choice = {
        "none": "none",
        "any": "required",
        "tool": {"type": "function", "function": {"name": choice.get("name")}},
    }.get(choice.get("type"), "auto")

    out = {"model": body.get("model"), "messages": messages, "tools": tools, "tool_choice": tool_choice}
    output_config = body.get("output_config") or {}
    fmt = output_config.get("format") or body.get("output_format") or {}
    if fmt.get("type") == "json_schema":
        out["response_format"] = {"type": "json_schema", "json_schema": {"schema": fmt.get("schema")}}
    if output_config.get("effort"):
        out["reasoning_effort"] = output_config["effort"]
    return out


def openai_to_anthropic(response):
    """Translate a chat completion response into an Anthropic Messages response."""
    message = response["choices"][0]["message"]
    content = [{"type": "text", "text": message["content"]}] if message.get("content") else []
    for call in message.get("tool_calls") or []:
        f = call["function"]
        content.append({"type": "tool_use", "id": call["id"], "name": f["name"], "input": json.loads(f["arguments"])})
    usage = response["usage"]
    return {
        "id": "msg_" + response["id"].removeprefix("chatcmpl-"),
        "type": "message",
        "role": "assistant",
        "model": response["model"],
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": "tool_use" if message.get("tool_calls") else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]},
    }


def anthropic_events(message):
    """Turn a finished Anthropic message into its stream events (one delta per content block)."""
    yield {"type": "message_start", "message": dict(
        message, content=[], stop_reason=None, usage=dict(message["usage"], output_tokens=0))}
    for i, block in enumerate(message["content"]):
        if block["type"] == "text":
            yield {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}}
        else:
            yield {"type": "content_block_start", "index": i, "content_block": dict(block, input={})}
            yield {"type": "content_block_delta", "index": i,
                   "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}
        yield {"type": "content_block_stop", "index": i}
    yield {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
           "usage": {"output_tokens": message["usage"]["output_tokens"]}}
    yield {"type": "message_stop"}


ROUTES = {
    "/v1/chat/completions": "openai",
    "/chat/completions": "openai",
    "/v1/messages": "anthropic",
    "/messages": "anthropic",
}
ANTHROPIC_ERROR_TYPES = {400: "invalid_request_error", 404: "not_found_error", 429: "rate_limit_error",
                         504: "timeout_error", 529: "overloaded_error"}


class Handler(BaseHTTPRequestHandler):
    server_version = f"cli-llm-bridge/{__version__}"
    config = None  # set in main()
    slots = None  # semaphore limiting concurrent claude processes

    def do_GET(self):
        if urlsplit(self.path).path.rstrip("/") in ("/v1/models", "/models"):
            # One list that satisfies both the OpenAI and the Anthropic model list shapes.
            data = [{"id": m, "object": "model", "type": "model", "display_name": m, "created": 0,
                     "created_at": "1970-01-01T00:00:00Z", "owned_by": "claude-code"} for m in MODELS]
            self.send_json(200, {"object": "list", "data": data, "has_more": False,
                                 "first_id": MODELS[0], "last_id": MODELS[-1]})
        else:
            self.send_error_json(BridgeError(404, f"Unknown path: {self.path}", "not_found"), "openai")

    def do_POST(self):
        # Anthropic SDKs add query strings such as ?beta=true, so match on the path only.
        api = ROUTES.get(urlsplit(self.path).path.rstrip("/"))
        if not api:
            self.send_error_json(BridgeError(404, f"Unknown path: {self.path}", "not_found"), "openai")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                raise BridgeError(400, "Request body is not valid JSON.")
            with self.slots:
                if api == "anthropic":
                    response = openai_to_anthropic(complete(self.config, anthropic_to_openai(body)))
                else:
                    response = complete(self.config, body)
        except BridgeError as e:
            self.send_error_json(e, api)
            return
        if not body.get("stream"):
            self.send_json(200, response)
        elif api == "anthropic":
            self.send_events(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in anthropic_events(response))
        else:
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            chunks = [f"data: {json.dumps(c)}\n\n" for c in stream_chunks(response, include_usage)]
            self.send_events(chunks + ["data: [DONE]\n\n"])

    def send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_events(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for event in events:
            self.wfile.write(event.encode())

    def send_error_json(self, e, api):
        if api == "anthropic":
            kind = ANTHROPIC_ERROR_TYPES.get(e.status, "api_error")
            self.send_json(e.status, {"type": "error", "error": {"type": kind, "message": e.message}})
        else:
            self.send_json(e.status, {"error": {"message": e.message, "type": e.kind, "code": e.status}})


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenAI- and Anthropic-compatible local HTTP API backed by the Claude Code CLI.")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: 127.0.0.1, local only).")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765).")
    parser.add_argument("--claude", default=shutil.which("claude"), help="Path to the claude executable (default: found on PATH).")
    parser.add_argument("--default-model", help="Model used when a request names none (default: the CLI's own default).")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Maximum parallel claude processes (default: 2).")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to wait for one answer (default: 600).")
    parser.add_argument("--version", action="version", version=__version__)
    config = parser.parse_args(argv)

    if not config.claude:
        sys.exit("Error: the claude executable was not found on PATH. Install Claude Code or pass --claude.")
    try:
        loopback = ipaddress.ip_address(config.host).is_loopback
    except ValueError:
        loopback = config.host == "localhost"
    if not loopback:
        print(
            f"Warning: listening on {config.host} exposes your Claude account to the network. "
            "Anyone who can reach this port can use it. See README.md.",
            file=sys.stderr,
        )

    Handler.config = config
    Handler.slots = threading.BoundedSemaphore(config.max_concurrency)
    server = ThreadingHTTPServer((config.host, config.port), Handler)
    base = f"http://{config.host}:{config.port}"
    print(
        f"cli-llm-bridge {__version__} listening\n"
        f"  OpenAI base URL:    {base}/v1\n"
        f"  Anthropic base URL: {base}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
