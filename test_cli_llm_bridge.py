"""Offline tests: the claude CLI is replaced by a fake, so no account is needed."""

import json
import unittest
import unittest.mock
from types import SimpleNamespace

from cli_llm_bridge import (
    DEFAULT_SYSTEM,
    BridgeError,
    anthropic_events,
    anthropic_to_openai,
    build_prompt,
    child_env,
    complete,
    openai_to_anthropic,
    stream_chunks,
)

CONFIG = SimpleNamespace(default_model=None, timeout=10)
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def fake_run(result):
    calls = []

    def run(config, system, prompt, model, effort, schema):
        calls.append(dict(system=system, prompt=prompt, model=model, effort=effort, schema=schema))
        return dict({"result": "", "usage": {"input_tokens": 10, "output_tokens": 2}}, **result)

    return run, calls


class BuildPromptTest(unittest.TestCase):
    def test_single_user_message_is_sent_as_is(self):
        system, prompt = build_prompt([{"role": "user", "content": "Hi"}])
        self.assertEqual((system, prompt), (DEFAULT_SYSTEM, "Hi"))

    def test_transcript_keeps_order_tool_calls_and_results(self):
        system, prompt = build_prompt([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
        ])
        self.assertEqual(system, "Be brief.")
        self.assertLess(prompt.index("[user]\nWeather?"), prompt.index('get_weather({"city":"Paris"})'))
        self.assertLess(prompt.index("get_weather"), prompt.index("[tool result c1]\nSunny"))

    def test_rejects_images_and_empty_conversations(self):
        with self.assertRaises(BridgeError):
            build_prompt([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])
        with self.assertRaises(BridgeError):
            build_prompt([{"role": "system", "content": "only system"}])


class CompleteTest(unittest.TestCase):
    def test_plain_chat(self):
        run, calls = fake_run({"result": "Hello."})
        r = complete(CONFIG, {"model": "haiku", "messages": [{"role": "user", "content": "Hi"}]}, run)
        self.assertEqual(r["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertEqual(r["usage"]["total_tokens"], 12)
        self.assertEqual(calls[0]["model"], "haiku")
        self.assertIsNone(calls[0]["schema"])

    def test_tool_calls_are_returned_in_openai_shape(self):
        run, calls = fake_run({"structured_output": {
            "content": "", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}})
        r = complete(CONFIG, {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER]}, run)
        choice = r["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        self.assertIn("get_weather", calls[0]["system"])
        variant = calls[0]["schema"]["properties"]["tool_calls"]["items"]["anyOf"][0]
        self.assertEqual(variant["properties"]["arguments"], WEATHER["function"]["parameters"])

    def test_tool_choice_none_and_required(self):
        run, calls = fake_run({"result": "No tools.", "structured_output": {"content": "x", "tool_calls": []}})
        messages = [{"role": "user", "content": "Hi"}]
        complete(CONFIG, {"messages": messages, "tools": [WEATHER], "tool_choice": "none"}, run)
        self.assertIsNone(calls[-1]["schema"])
        complete(CONFIG, {"messages": messages, "tools": [WEATHER], "tool_choice": "required"}, run)
        self.assertEqual(calls[-1]["schema"]["properties"]["tool_calls"]["minItems"], 1)

    def test_large_tool_sets_get_a_compact_schema(self):
        big = {"type": "object", "properties": {f"p{i}": {"type": "string", "description": "x" * 200} for i in range(10)}}
        tools = [{"type": "function", "function": {"name": f"tool_{i}", "parameters": big}} for i in range(20)]
        run, calls = fake_run({"structured_output": {"content": "", "tool_calls": []}})
        complete(CONFIG, {"messages": [{"role": "user", "content": "Hi"}], "tools": tools}, run)
        items = calls[0]["schema"]["properties"]["tool_calls"]["items"]
        self.assertEqual(items["properties"]["name"]["enum"], [f"tool_{i}" for i in range(20)])
        self.assertLess(len(json.dumps(calls[0]["schema"])), 2000)
        self.assertIn('"p9"', calls[0]["system"])  # parameters still reach the model

    def test_json_schema_response_format(self):
        schema = {"type": "object", "properties": {"rating": {"type": "integer"}}}
        run, calls = fake_run({"structured_output": {"rating": 4}})
        r = complete(CONFIG, {
            "messages": [{"role": "user", "content": "Rate it."}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}},
        }, run)
        self.assertEqual(json.loads(r["choices"][0]["message"]["content"]), {"rating": 4})
        self.assertEqual(calls[0]["schema"], schema)

    def test_stream_chunks_end_with_finish_reason_and_usage(self):
        run, _ = fake_run({"result": "Hello."})
        r = complete(CONFIG, {"messages": [{"role": "user", "content": "Hi"}]}, run)
        chunks = list(stream_chunks(r, include_usage=True))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Hello.")
        self.assertEqual(chunks[1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[2]["usage"], r["usage"])


class AnthropicTest(unittest.TestCase):
    TOOL = {"name": "get_weather", "description": "Current weather.", "input_schema": WEATHER["function"]["parameters"]}

    def test_request_translation_keeps_tool_round_trip(self):
        body = anthropic_to_openai({
            "model": "sonnet", "max_tokens": 1024,
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "Sunny"}]},
                    {"type": "text", "text": "And tomorrow?"}]},
            ],
            "tools": [self.TOOL],
            "tool_choice": {"type": "any"},
            "output_config": {"effort": "low"},
        })
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(body["messages"][2]["tool_calls"][0]["function"]["arguments"], '{"city": "Paris"}')
        self.assertEqual(body["messages"][3], {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny"})
        self.assertEqual(body["tools"][0]["function"]["parameters"], self.TOOL["input_schema"])
        self.assertEqual((body["tool_choice"], body["reasoning_effort"]), ("required", "low"))

    def test_rejects_server_tools_and_images(self):
        with self.assertRaises(BridgeError):
            anthropic_to_openai({"messages": [{"role": "user", "content": "Hi"}],
                                 "tools": [{"type": "web_search_20250305", "name": "web_search"}]})
        with self.assertRaises(BridgeError):
            anthropic_to_openai({"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]})

    def test_tool_use_response_and_stream_events(self):
        run, _ = fake_run({"structured_output": {
            "content": "Checking.", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}})
        body = anthropic_to_openai({"messages": [{"role": "user", "content": "Weather?"}], "tools": [self.TOOL]})
        message = openai_to_anthropic(complete(CONFIG, body, run))
        self.assertEqual(message["stop_reason"], "tool_use")
        self.assertEqual([b["type"] for b in message["content"]], ["text", "tool_use"])
        self.assertEqual(message["content"][1]["input"], {"city": "Paris"})
        events = list(anthropic_events(message))
        self.assertEqual(events[0]["type"], "message_start")
        self.assertEqual(events[-1]["type"], "message_stop")
        deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
        self.assertEqual(json.loads(deltas[1]["partial_json"]), {"city": "Paris"})

    def test_child_env_never_points_claude_back_at_the_bridge(self):
        config = SimpleNamespace(host="127.0.0.1", port=8765)
        with unittest.mock.patch.dict("os.environ", {"ANTHROPIC_BASE_URL": "http://localhost:8765"}):
            self.assertNotIn("ANTHROPIC_BASE_URL", child_env(config))
        with unittest.mock.patch.dict("os.environ", {"ANTHROPIC_BASE_URL": "https://gateway.example.com"}):
            self.assertIn("ANTHROPIC_BASE_URL", child_env(config))


if __name__ == "__main__":
    unittest.main()
