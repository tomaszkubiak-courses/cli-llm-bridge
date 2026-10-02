# Example: TradingAgents

[TradingAgents](https://github.com/TauricResearch/TradingAgents) is a multi-agent LLM
trading research framework built on LangGraph. It can use cli-llm-bridge with **no code
changes**: it already supports a generic OpenAI-compatible provider and a custom Anthropic base
URL.

Tested with TradingAgents v0.5.2.

Before you start, read the bridge's [terms of use](../../README.md#read-this-first-terms-of-use)
and [limitations](../../README.md#limitations). A TradingAgents run makes many LLM calls, and
every one of them counts against your Claude plan's usage limits.

## 1. Install TradingAgents

Follow the installation steps in the
[TradingAgents README](https://github.com/TauricResearch/TradingAgents#installation). Set up
its data sources as described in its
[Required APIs](https://github.com/TauricResearch/TradingAgents#required-apis) section; only
the LLM settings change, and no LLM provider key is needed.

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

See [Installation](../../README.md#installation) for the setup each line assumes.
TradingAgents runs several analysts, so allowing a few parallel requests shortens a run.

## 3. Configure TradingAgents

TradingAgents reads `TRADINGAGENTS_*` settings from the environment or from a `.env` file in
the folder you run it from. It loads that file into its own process only, so these settings
don't reach the bridge.

### Option A: OpenAI-compatible provider (recommended)

```env
TRADINGAGENTS_LLM_PROVIDER=openai_compatible
TRADINGAGENTS_LLM_BACKEND_URL=http://127.0.0.1:8765/v1
TRADINGAGENTS_DEEP_THINK_LLM=sonnet
TRADINGAGENTS_QUICK_THINK_LLM=haiku
```

This uses TradingAgents' generic provider for custom endpoints. It needs no API key.

### Option B: Anthropic provider

```env
TRADINGAGENTS_LLM_PROVIDER=anthropic
TRADINGAGENTS_LLM_BACKEND_URL=http://127.0.0.1:8765
TRADINGAGENTS_DEEP_THINK_LLM=sonnet
TRADINGAGENTS_QUICK_THINK_LLM=haiku
ANTHROPIC_API_KEY=unused
```

The Anthropic client refuses to start without a key, so it gets a placeholder. TradingAgents
warns that `sonnet` and `haiku` are not in its known model list; the warning is harmless. Full
model names avoid it and also let TradingAgents send its `effort` setting.

### Models

`deep_think_llm` handles the research and portfolio decisions, `quick_think_llm` the analysts'
data gathering. Any model name your Claude Code version accepts works, for example `opus`
for the deep model if your plan allows it.

## 4. Run

With `TRADINGAGENTS_LLM_PROVIDER` set, the interactive CLI skips its provider menu:

```bash
tradingagents
```

Pick a shallow research depth for the first run.

Or from Python, as in TradingAgents' own `main.py` (`DEFAULT_CONFIG` already applies the
settings above):

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
_, decision = ta.propagate("NVDA", "2026-09-01")
print(decision)
```

## What was verified

Using TradingAgents' own client code against the bridge, with both options:

- analyst tool calls (`bind_tools`, for example `get_stock_data`) return correct tool calls,
- the structured outputs of the Trader (`TraderProposal`) and Portfolio Manager
  (`PortfolioDecision`) parse correctly, without falling back to free text.

## What to expect

- **Runs are slower** than with a real API: each LLM call takes several seconds, and a full run
  makes dozens of them.
- **Live output arrives in bursts**, because the bridge delivers each streamed answer in one
  piece.
- `temperature` and `max_tokens` from the TradingAgents config are ignored by the bridge.
