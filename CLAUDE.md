# Working on this repository

TradingAgents with Taiwan (TWSE/TPEx) support through FinMind, on the `finmind-tw` branch. The user writes in
Traditional Chinese; reports for Taiwan runs are in Traditional Chinese (`TRADINGAGENTS_OUTPUT_LANGUAGE`).

## Rules

- **Publication dates live in one place:** `tradingagents/dataflows/vendors/finmind/publication.py` decides when
  monthly revenue and quarterly statements count as public (`revenue_visible`, `latest_due_revenue_month`,
  `statement_visible`, `latest_due_statement`, `is_financial`). The data layer trims with it and the coverage
  step's freshness check calls it. Never compute a deadline anywhere else: two copies of this rule once
  disagreed and capped healthy runs (2881.TW on 2026-10-14).
- **Real runs never touch the memory log:** every test or calibration run of the CLI takes `--no-memory-log`
  (or `TRADINGAGENTS_MEMORY_LOG_ENABLED=false`). Baseline entries already in the log are marked `RUN: baseline`
  and are ignored by every reader.
- **Each phase is its own commit**, and the full suite passes at that commit: `.venv/bin/python -m pytest tests -q`
  and `.venv/bin/ruff check tradingagents cli tests`. Tests never reach the network (`tests/conftest.py`); a test
  that leaves `DEFAULT_CONFIG` changed fails.
- **Labels the program parses stay in English** in every output language: `**Rating**:`, `**Recommendation**:`,
  `**Confidence**:`, `**Data Limitations**:`, `**Coverage Cap**`, `**Overall Sentiment:**`, `**News Items:**`,
  `FINAL TRANSACTION PROPOSAL:`. Text the program writes into a report goes through `tradingagents/agents/i18n.py`.
  Analysts, researchers and debaters ask `get_language_instruction(labelled_lines=False)`; only the research
  manager, trader and portfolio manager are told about the rating lines.
- Copy the default config with `copy.deepcopy`, never `.copy()` or `{**config}`: its settings nest.

## FinMind dataset traps

- **Monthly revenue** (`TaiwanStockMonthRevenue`): `date` is the month of *publication*; the month covered is
  `revenue_year` / `revenue_month`. `create_time` is empty on most rows before 2026, FinMind re-stamped every
  stock's February and March 2026 rows on 2026-04-20/21, and financial holding companies and insurers publish
  2–7 days after the 10th.
- **Cash-flow statement** (`TaiwanStockCashFlowsStatement`) is year-to-date: a quarter is the difference of two
  periods (Q2 = H1 − Q1). The income statement is per quarter; the balance sheet carries `<item>_per` rows to drop.
- **Net income:** `EquityAttributableToOwnersOfParent` in the income statement is net income attributable to
  the parent (歸母淨利); `IncomeAfterTaxes` includes non-controlling interests (稅後淨利). EPS is on the parent's.
- **Prices:** `TaiwanStockPrice` is traded prices, not dividend-adjusted (`TaiwanStockPriceAdj` is the adjusted
  one); Yahoo's are adjusted. Don't mix the two in one series.
- **Units:** institutional buys and sells (`TaiwanStockInstitutionalInvestorsBuySell`) are in shares (1 lot,
  張, = 1,000 shares); margin and short balances are in lots; market-wide institutional totals are in TWD.
- **News** (`TaiwanStockNews`) is one day per request, stamped in UTC; cut windows on the Taipei clock
  (`date_window.in_market_window`).
- Statements have no filing date: they count as public on their statutory deadline.
- The free plan allows 600 requests an hour; `client.fetch` caches every answer and counts requests against
  `finmind_requests_per_hour`. `TaiwanStockInfo` without a `data_id`, `TaiwanStockTradingDate` and the
  market-wide totals are free; `InterestRate` without a `data_id` and `TaiwanBusinessIndicator` are not.

## Where things are

- Data coverage, freshness checks and the score: `tradingagents/agents/coverage.py`; the cap on the final
  decision: `tradingagents/agents/rating.py` (`apply_coverage_cap`); the data-gap prompt rules:
  `tradingagents/agents/context.py` (`data_gap_rules`).
- Sentiment for a Taiwan listing counts independent headlines (`tradingagents/agents/headlines.py`) and reports
  Insufficient Data below `sentiment_min_items`.
- Output language check: `tradingagents/agents/language_check.py`.
- The working plan and deferred items: `~/.claude/plans/pasted-content-id-5714-tradingagents-giggly-hopper.md`.
