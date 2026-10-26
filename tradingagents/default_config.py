import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_DEEP_THINK_PROVIDER":       "deep_think_provider",
    "TRADINGAGENTS_QUICK_THINK_PROVIDER":      "quick_think_provider",
    "TRADINGAGENTS_DEEP_THINK_BACKEND_URL":    "deep_think_backend_url",
    "TRADINGAGENTS_QUICK_THINK_BACKEND_URL":   "quick_think_backend_url",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_MAX_TOOL_ROUNDS":      "max_tool_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES":      "llm_max_retries",
    "TRADINGAGENTS_MAX_TOKENS":           "max_tokens",
    "TRADINGAGENTS_AS_OF_TIME":           "as_of_time",
    "TRADINGAGENTS_MEMORY_LOG_ENABLED":   "memory_log_enabled",
    # A dotted key names a nested setting.
    "TRADINGAGENTS_COVERAGE_ENFORCE":     "data_coverage.enforce",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place.

    A dotted key (``data_coverage.enforce``) sets a value inside a nested dict.
    """
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        *parents, leaf = key.split(".")
        target = config
        for parent in parents:
            target = target.setdefault(parent, {})
        try:
            target[leaf] = _coerce(raw, target.get(leaf))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


def build_default_config() -> dict:
    """The built-in defaults with the TRADINGAGENTS_* environment folded in.

    Read when the package is imported, as DEFAULT_CONFIG; call it again to see
    the environment as it is now.
    """
    return _apply_env_overrides({
        "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR") or os.path.join(_TRADINGAGENTS_HOME, "logs"),
        "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR") or os.path.join(_TRADINGAGENTS_HOME, "cache"),
        "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH") or os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md"),
        # Optional cap on the number of resolved memory log entries. When set,
        # the oldest resolved entries are pruned once this limit is exceeded.
        # Pending entries are never pruned. None disables rotation entirely.
        "memory_log_max_entries": None,
        # False runs without touching the memory log: no decision is recorded
        # and no past one is settled (the lessons already there are still
        # read). For test and baseline runs, which are not decisions.
        "memory_log_enabled": True,
        # LLM settings
        "llm_provider": "openai",
        "deep_think_llm": "gpt-6-sol",
        "quick_think_llm": "gpt-6-luna",
        # When None, each provider's client falls back to its own default endpoint
        # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
        # The CLI overrides this per provider when the user picks one. Keeping a
        # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
        # being forwarded to Gemini, producing malformed request URLs).
        "backend_url": None,
        # A tier may name its own provider and endpoint (#1440): the quick tier serves
        # the analysts, researchers, debaters and trader, the deep tier the managers.
        # None means the tier uses llm_provider and backend_url.
        "quick_think_provider": None,
        "deep_think_provider": None,
        "quick_think_backend_url": None,
        "deep_think_backend_url": None,
        # Provider-specific thinking configuration
        "google_thinking_level": None,      # "high", "minimal", etc.
        "openai_reasoning_effort": None,    # "medium", "high", "low"
        "anthropic_effort": None,           # "high", "medium", "low"
        # Sampling temperature, forwarded to every provider when set. None leaves
        # each provider at its own default. Lower values reduce run-to-run
        # variation on models that honor it; reasoning models largely ignore it
        # and no setting makes LLM output bit-identical across runs (see README).
        "temperature": None,
        # SDK retry budget forwarded to every provider chat client. None leaves each
        # provider/SDK at its own default (usually 2). Raise it to ride out bursty
        # 429 throttling on rate-limited deployments instead of aborting a run (#1091).
        "llm_max_retries": None,
        # Cap on output tokens forwarded to every provider chat client. None leaves
        # each provider at its own default. Set it to bound a model that emits
        # unbounded reasoning/output and hangs or trips a gateway idle timeout
        # (e.g. some deepseek-v4-flash deployments, #1204).
        "max_tokens": None,
        # Checkpoint/resume: when True, LangGraph saves state after each node
        # so a crashed run can resume from the last successful step.
        "checkpoint_enabled": False,
        # Output language for analyst reports and final decision
        # Internal agent debate stays in English for reasoning quality
        "output_language": "English",
        # Debate and discussion settings
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        # Rounds of tool calls an analyst may make before it is asked for its report.
        "max_tool_rounds": 20,
        # News / data fetching parameters
        # Increase for longer lookback strategies or to broaden macro coverage;
        # decrease to reduce token usage in agent prompts.
        "news_article_limit": 20,             # max articles per ticker (ticker-news)
        # A Taiwan listing's sentiment rests on headlines alone: with fewer
        # independent stories than this (after de-duplication, agents/headlines.py)
        # the sentiment analyst reports Insufficient Data and calls no model.
        "sentiment_min_items": 5,
        "global_news_article_limit": 10,      # max articles for global/macro news
        "global_news_lookback_days": 7,       # macro news lookback window
        # For a Taiwan listing, dated items (news) count as known by the
        # analysis date up to this time of day, Asia/Taipei. "13:30:00" would
        # hold a run to what was public at the close.
        "as_of_time": "23:59:59",
        # Search queries used by get_global_news for macro headlines. Extend or
        # replace to broaden geographic / sector coverage.
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        # Used instead of global_news_queries for a Taiwan listing (2330, 2383.TW,
        # 8069.TWO): Taiwan's market, rates, currency and the supply chains that
        # move it. Yahoo serves recent articles only, so a past date has none.
        "global_news_queries_tw": [
            "台股 加權指數 外資",
            "Taiwan stock market TAIEX",
            "Taiwan central bank interest rate",
            "新台幣 匯率",
            "semiconductor supply chain TSMC AI server",
        ],
        # Data vendor configuration
        # Category-level configuration (default for all tools in category).
        # The configured value is the exact vendor chain — requests are NOT silently
        # routed to vendors you didn't choose. For ordered fallback, list several,
        # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
        #
        # Taiwan stocks (TWSE/TPEx). With the defaults below nothing needs
        # switching: set FINMIND_TOKEN (in .env) and a Taiwan symbol (2330,
        # 2330.TW, 8069.TWO) is served by FinMind while every other symbol skips
        # it and goes to the next vendor. Without the token, Taiwan symbols go
        # to Yahoo (2330.TW) too. Other set-ups, passed to set_config() or
        # TradingAgentsGraph(config=...):
        #
        #   Turn FinMind off, Yahoo for everything:
        #     "data_vendors": {"core_stock_apis": "yfinance",
        #                      "technical_indicators": "yfinance",
        #                      "fundamental_data": "sec_edgar,yfinance",
        #                      "news_data": "yfinance"}
        #
        #   FinMind for one tool only, e.g. Taiwan news, Yahoo for the rest:
        #     "data_vendors": {"news_data": "yfinance", ...as above...},
        #     "tool_vendors": {"get_news": "finmind,yfinance"}
        #
        #   FinMind with no Yahoo fallback (Taiwan-only runs; a US symbol then
        #   reports NO_DATA_AVAILABLE instead of being priced):
        #     "data_vendors": {"core_stock_apis": "finmind",
        #                      "technical_indicators": "finmind",
        #                      "fundamental_data": "finmind",
        #                      "news_data": "finmind"}
        #
        #   Taiwan-flavoured macro headlines and reports for a Taiwan run:
        #     "global_news_queries": ["台股 加權指數 外資", "Taiwan central bank interest rate",
        #                             "semiconductor supply chain TSMC"],
        #     "output_language": "Traditional Chinese"   # or TRADINGAGENTS_OUTPUT_LANGUAGE
        #
        # The alpha benchmark already follows the suffix: .TW / .TWO -> ^TWII.
        "data_vendors": {
            # FinMind serves Taiwan (TWSE/TPEx) symbols such as 2330, 2330.TW or
            # 8069.TWO and needs FINMIND_TOKEN; any other symbol, or a missing
            # token, goes straight to the next vendor in the chain.
            "core_stock_apis": "finmind,yfinance",       # Options: finmind, alpha_vantage, yfinance
            "technical_indicators": "finmind,yfinance",  # Options: finmind, alpha_vantage, yfinance
            # Statements come from SEC EDGAR as filed (US filers), then Yahoo; the
            # overview and insider tools, which SEC EDGAR does not serve, from Yahoo.
            "fundamental_data": "finmind,sec_edgar,yfinance",  # Options: finmind, sec_edgar, alpha_vantage, yfinance
            "news_data": "finmind,yfinance",             # Options: finmind, alpha_vantage, yfinance
            "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
            "prediction_markets": "polymarket",  # Options: polymarket (keyless)
            # Taiwan chip data (institutional flows, margin, foreign holding,
            # securities lending) for the chip analyst; FinMind only.
            "chip_data": "finmind",              # Options: finmind
        },
        # The chip analyst's look-back and its high-volume down day rule: volume at
        # least volume_spike_ratio times the prior 20-day average, close down at
        # least down_day_pct percent.
        "chip_analysis": {"lookback_days": 20, "volume_spike_ratio": 1.5, "down_day_pct": 3.0},
        # The Data Coverage step scores the run's tool results from 0 to 1
        # (agents/coverage.py); a score of at least "high" reads as high
        # coverage, at least "medium" as medium, anything lower as low. Add a
        # "critical" list of checks to replace coverage.CRITICAL_CHECKS: one of
        # them not fully served holds the level at medium at most.
        #
        # The coverage cap on the final decision (agents/rating.apply_coverage_cap):
        # coverage low (cap_on_low) or critical data missing (cap_on_critical)
        # moves a rating by rating_cap and holds confidence at low; medium
        # coverage holds confidence at medium. With enforce false (the default)
        # the decision only records what the cap would change; true applies it.
        "data_coverage": {
            "high": 0.8, "medium": 0.5,
            "enforce": False,
            "cap_on_low": True, "cap_on_critical": True,
            "rating_cap": {"Buy": "Overweight", "Sell": "Underweight"},
        },
        # Taiwan peer groups for the peer valuation table, by stock code:
        # {"2383": ["6274", "6213"]}. Peers are read from tradingagents/data/tw_peers.json,
        # then ~/.tradingagents/tw_peers.json, then this key; each replaces the
        # lists of the codes it names.
        "peer_groups": {},
        # Tool-level configuration (takes precedence over category-level)
        "tool_vendors": {
            # Example: "get_stock_data": "alpha_vantage",  # Override category default
        },
        # Benchmark for alpha calculation in the reflection layer.
        # ``benchmark_ticker`` (when set) overrides the suffix map for all
        # tickers; leave it None to use ``benchmark_map`` for auto-detection
        # based on the ticker's exchange suffix. SPY remains the US default
        # so the reflection label keeps reading "Alpha vs SPY" for US tickers
        # while non-US tickers get their regional index automatically.
        # Trading days from the first close after the analysis date over which a
        # decision's outcome is measured, for reflection and for the backtest figures.
        "holding_period_days": 5,
        "benchmark_ticker": None,
        "benchmark_map": {
            ".NS":  "^NSEI",       # NSE India (Nifty 50)
            ".BO":  "^BSESN",      # BSE India (Sensex)
            ".T":   "^N225",       # Tokyo (Nikkei 225)
            ".TW":  "^TWII",       # Taiwan (TAIEX)
            ".TWO": "^TWII",       # Taipei OTC (TPEx has no Yahoo index; TAIEX)
            ".KS":  "^KS11",       # Korea (KOSPI)
            ".KQ":  "^KQ11",       # Korea (KOSDAQ)
            ".HK":  "^HSI",        # Hong Kong (Hang Seng)
            ".SI":  "^STI",        # Singapore (Straits Times)
            ".L":   "^FTSE",       # London (FTSE 100)
            ".DE":  "^GDAXI",      # Germany (DAX)
            ".PA":  "^FCHI",       # Paris (CAC 40)
            ".AS":  "^AEX",        # Amsterdam (AEX)
            ".SW":  "^SSMI",       # Switzerland (SMI)
            ".MI":  "FTSEMIB.MI",  # Milan (FTSE MIB)
            ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
            ".AX":  "^AXJO",       # Australia (ASX 200)
            ".SS":  "000001.SS",   # Shanghai (SSE Composite)
            ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
            ".SA":  "^BVSP",       # B3 Brazil (Ibovespa)
            "":     "SPY",         # default for US-listed tickers (no suffix)
        },

    })


DEFAULT_CONFIG = build_default_config()
