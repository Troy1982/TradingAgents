"""Backtest how well the final rating of a Taiwan run predicts what followed.

Unlike ``tradingagents.backtest``, whose results table is the memory log, this
never touches the memory log: every run is a CLI run with ``--no-memory-log``
into a results directory of its own, and what it decided is one line of
``runs.jsonl``. Returns are computed afterwards, without a model, from FinMind
prices and the TAIEX / TPEx total return indices.

The steps, each a command of ``python -m tradingagents.rating_backtest``:

- ``universe``: draw the stocks from what was listed on a date, before any run
  (``universe.py``), and write the plan;
- ``run``: run every (ticker, date, run) cell of the plan not yet in
  ``runs.jsonl`` (``runner.py``, ``store.py``);
- ``settle``: the excess return of each cell over 5, 20 and 60 trading days
  (``settle.py``);
- ``report``: the metrics and the baselines (``metrics.py``).
"""
