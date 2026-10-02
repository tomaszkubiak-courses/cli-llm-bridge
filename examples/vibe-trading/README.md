# Example: Vibe-Trading

[Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) is a natural-language finance research
agent with backtesting. It can use cli-llm-bridge with **no code changes**: its Anthropic
provider accepts a custom base URL ("Anthropic / Messages API Proxy").

Tested with Vibe-Trading v0.1.16.

Before you start, read the bridge's [terms of use](../../README.md#read-this-first-terms-of-use)
and [limitations](../../README.md#limitations). Vibe-Trading's agent makes many LLM calls per
task, and every one of them counts against your Claude plan's usage limits.

## Requirements

- **A recent bridge.** Vibe-Trading sends more than 100 tool definitions with every request.
  Bridge versions before the compact tool schema (commit `1ae6af8`) pass them on the command
  line and exceed the Windows limit of 32,767 characters, so every agent call fails there.
- **Vibe-Trading with the fix for issue
  [#1672](https://github.com/HKUDS/Vibe-Trading/issues/1672)**, if you run backtests. On
  `main` at v0.1.16, any run that writes a strategy file crashes with
  `AttributeError: 'AgentLoop' object has no attribute '_active_model_id'`, whatever the
  provider. The fix is in pull request
  [#1673](https://github.com/HKUDS/Vibe-Trading/pull/1673); until it is merged, check out that
  branch or skip backtests.

## 1. Install Vibe-Trading

Follow the local install path in the
[Vibe-Trading README](https://github.com/HKUDS/Vibe-Trading#path-b-local-install). It needs
Python 3.11 to 3.13. Install the Anthropic extra as well:

```bash
pip install -e ".[anthropic]"
```

## 2. Start the bridge

Start the bridge in its own terminal. That terminal must **not** have `ANTHROPIC_API_KEY` or
`ANTHROPIC_BASE_URL` set, because Claude Code reads them too (see
[Environment variables](../../README.md#environment-variables)).

```bash
# Installed with pip (inside the activated virtual environment) or with uv tool install
cli-llm-bridge --max-concurrency 3

# uv project environment (after uv sync)
uv run cli-llm-bridge --max-concurrency 3

# Not installed: run the file from the cli-llm-bridge folder
python cli_llm_bridge.py --max-concurrency 3
```

## 3. Configure Vibe-Trading

Copy `agent/.env.example` to `agent/.env`, comment out the provider block that is active there
(OpenRouter), and add:

```env
LANGCHAIN_PROVIDER=anthropic
LANGCHAIN_MODEL_NAME=sonnet
ANTHROPIC_BASE_URL=http://127.0.0.1:8765
ANTHROPIC_API_KEY=unused
ANTHROPIC_MAX_TOKENS=16384
```

- The Anthropic client refuses to start without a key, so it gets a placeholder.
- `ANTHROPIC_MAX_TOKENS` is Vibe-Trading's own advice for model names it does not recognise.
- Vibe-Trading loads `agent/.env` into its own process only, so these settings don't reach the
  bridge.

Also raise the request timeout in the same file, because bridge calls take longer than a direct
API:

```env
TIMEOUT_SECONDS=600
```

### OpenAI route

The OpenAI-compatible route works too:

```env
LANGCHAIN_PROVIDER=openai
LANGCHAIN_MODEL_NAME=sonnet
OPENAI_BASE_URL=http://127.0.0.1:8765/v1
OPENAI_API_KEY=unused
```

Leave `LANGCHAIN_USE_RESPONSES_API` unset: the bridge does not serve OpenAI's Responses API.
The Anthropic route is recommended because it is the one Vibe-Trading documents for proxies.

### Models

Use `sonnet` or `opus`. With more than 100 tools to choose from, smaller models such as `haiku`
pick tools less reliably.

## 4. Run

```bash
vibe-trading                      # interactive terminal UI
vibe-trading serve --port 8899    # web UI
vibe-trading run -p "Backtest a BTC-USDT 20/50 moving-average strategy for 2024 and summarize return and drawdown"
```

## What was verified

With the Anthropic route and `sonnet`:

| Task | Result |
| --- | --- |
| Daily prices for AAPL.US over 10 trading days, latest close and percentage change | Success in about 20 s, 2 bridge calls |
| The quick-start backtest above | Success in about 1 to 2 minutes, 6 to 14 bridge calls: the agent wrote the strategy code, ran it on OKX data and summarized the results |

The bridge was also tested with Vibe-Trading's real tool definitions: the right tools were picked
with correct arguments.

## What to expect

- **Runs are slower** than with a real API: each LLM call takes several seconds.
- **Streamed answers appear all at once** when they are finished.
- **Image analysis fails**: Vibe-Trading's vision tool sends images, which the bridge rejects
  with HTTP 400.
- **Vibe-Trading's figure check may reject drafts.** Its grounding gate checks every number in
  the answer against the session's tool data and can reject a few drafts before it releases
  one, occasionally cutting a number from a heading. This is Vibe-Trading's own behaviour, not
  the bridge's.
