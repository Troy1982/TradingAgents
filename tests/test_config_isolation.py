"""Each run works on its own copy of the settings: nested ones never reach DEFAULT_CONFIG or the next run."""

import pytest

import tradingagents.default_config as default_config
from cli import run as cli_run
from tradingagents.graph import trading_graph

_SELECTIONS = {"research_depth": 1, "quick_think_llm": "q", "deep_think_llm": "d", "backend_url": None,
               "llm_provider": "openai", "output_language": "English"}


@pytest.mark.unit
def test_a_cli_run_changing_a_nested_setting_leaves_the_defaults_and_the_next_run_alone():
    first = cli_run._build_run_config(_SELECTIONS, checkpoint=None)
    first["data_coverage"]["enforce"] = True
    first["data_vendors"]["news_data"] = "yfinance"

    assert default_config.DEFAULT_CONFIG["data_coverage"]["enforce"] is False
    assert default_config.DEFAULT_CONFIG["data_vendors"]["news_data"] == "finmind,yfinance"
    second = cli_run._build_run_config(_SELECTIONS, checkpoint=None)
    assert second["data_coverage"]["enforce"] is False and second["data_vendors"]["news_data"] == "finmind,yfinance"


class _Client:
    def get_llm(self):
        return object()


@pytest.mark.unit
@pytest.mark.parametrize("passed", [None, "own"])
def test_a_graph_keeps_its_own_copy_of_the_settings(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda *a, **k: _Client())
    import copy
    config = None
    if passed:
        config = copy.deepcopy(default_config.DEFAULT_CONFIG)
        config.update(results_dir=str(tmp_path / "r"), data_cache_dir=str(tmp_path / "c"),
                      memory_log_path=str(tmp_path / "m.md"))
    else:
        monkeypatch.setitem(default_config.DEFAULT_CONFIG, "results_dir", str(tmp_path / "r"))
        monkeypatch.setitem(default_config.DEFAULT_CONFIG, "data_cache_dir", str(tmp_path / "c"))
        monkeypatch.setitem(default_config.DEFAULT_CONFIG, "memory_log_path", str(tmp_path / "m.md"))
    graph = trading_graph.TradingAgentsGraph(["market"], config=config)
    graph.config["data_coverage"]["enforce"] = True

    assert default_config.DEFAULT_CONFIG["data_coverage"]["enforce"] is False
    if config is not None:
        assert config["data_coverage"]["enforce"] is False


@pytest.mark.unit
def test_a_backtest_sweep_does_not_change_the_settings_it_was_given(tmp_path, monkeypatch):
    from tradingagents import backtest
    seen = {}

    class Graph:
        def __init__(self, analysts, config):
            config["data_coverage"]["enforce"] = True
            seen["config"] = config
            self.memory_log = type("Log", (), {"load_entries": lambda self: []})()

        def settle_pending(self, ticker):
            pass

    monkeypatch.setattr(backtest, "TradingAgentsGraph", Graph)
    monkeypatch.setitem(default_config.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    backtest.run_backtest(["2383.TW"], [], default_config.DEFAULT_CONFIG)
    assert seen["config"]["data_coverage"]["enforce"] is True
    assert default_config.DEFAULT_CONFIG["data_coverage"]["enforce"] is False
