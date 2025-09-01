import os
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
from agno.tools import Toolkit


def _load_pattern_module():
    """Dynamically load the pattern engine module without requiring package imports.

    This avoids modifying PYTHONPATH at app startup and keeps the original
    file structure intact.
    """
    import importlib.util

    current_dir = os.path.dirname(__file__)  # .../src/agente
    src_dir = os.path.dirname(current_dir)   # .../src
    module_path = os.path.join(src_dir, "patterns", "OCOs", "necklineconfirmada.py")

    spec = importlib.util.spec_from_file_location("pattern_engine", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pattern module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pattern_engine"] = module
    spec.loader.exec_module(module)
    return module


class PatternToolKit(Toolkit):
    """
    Chart Pattern Detection Toolkit (H&S, Double/Triple Top/Bottom)

    Wraps the logic from `src/patterns/OCOs/necklineconfirmada.py` and exposes a
    single-asset detection entrypoint, so only the coin selected in the
    Streamlit interface is processed (no full TICKERS iteration).
    """

    def __init__(self, default_vs_currency: Optional[str] = "usd"):
        super().__init__(name="pattern_tools")
        self.default_vs_currency = (default_vs_currency or "usd").lower()

        # Lazy-loaded module handle
        self._pattern_mod = None

        # Register public tools
        self.register(self.detect_patterns)

    # ----------------------- internal helpers -----------------------
    def _ensure_module(self):
        if self._pattern_mod is None:
            self._pattern_mod = _load_pattern_module()
        return self._pattern_mod

    def _term_defaults(self, term_type: str) -> Tuple[str, List[str], str]:
        """Map term_type to (strategy_name, intervals, period).

        - short: intraday momentum, 15m/1h, period ~90d
        - medium: swing medium, 4h/1d, period ~1y
        - long: position trend, 1d/1wk, period ~5y
        """
        t = (term_type or "short").lower()
        if t == "short":
            return "intraday_momentum", ["15m", "1h"], "90d"
        if t == "medium":
            return "swing_medium", ["4h", "1d"], "1y"
        return "position_trend", ["1d", "1wk"], "5y"

    def _build_ticker(self, coin_id: str, vs_currency: str) -> str:
        # The pattern engine accepts a ticker like "BTC-USD". It internally
        # maps to CoinGecko IDs; when not found, it falls back to the symbol
        # left of '-' lower-cased as coin_id, which works for e.g. "bitcoin-USD".
        return f"{(coin_id or '').upper()}-{(vs_currency or self.default_vs_currency).upper()}"

    # ------------------------- public tools -------------------------
    def detect_patterns(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        term_type: str = "short",
        patterns: str = "ALL",
        strategies: Optional[str] = None,
        intervals: Optional[str] = None,
        period: Optional[str] = None,
        save_csv: bool = False,
        output_csv: Optional[str] = None,
    ) -> str:
        """
        Detect chart patterns for a single asset using the pattern engine.

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum").
            vs_currency: Quote currency (e.g., "usd").
            term_type: "short" | "medium" | "long" (maps to strategy/interval defaults).
            patterns: Comma-separated types: HNS, DTB, TTB, or "ALL".
            strategies: Optional comma-separated strategy names to override defaults.
            intervals: Optional comma-separated intervals to override defaults.
            period: Optional period string (e.g., "90d", "2y"). If not provided, inferred from term_type.
            save_csv: If True, saves a CSV with detected patterns.
            output_csv: Optional custom path for the CSV when save_csv is True.

        Returns:
            JSON string containing a summary and (optionally) the saved CSV path.
        """
        print("🎯 [DEBUG] pattern_detect CHAMADA!")
        mod = self._ensure_module()

        # Resolve defaults based on term_type unless explicitly provided
        default_strategy, default_intervals, default_period = self._term_defaults(term_type)
        selected_strategies = [s.strip() for s in (strategies or default_strategy).split(",") if s.strip()]
        selected_intervals = [i.strip() for i in (intervals or ",".join(default_intervals)).split(",") if i.strip()]
        effective_period = (period or default_period)

        # Configure the engine period (avoids scanning huge ranges unintentionally)
        mod.Config.DATA_PERIOD = effective_period

        ticker = self._build_ticker(coin_id, vs_currency or self.default_vs_currency)

        all_found: List[Dict[str, Any]] = []
        errors: List[str] = []

        try:
            for strategy_name in selected_strategies:
                strategy_cfg = mod.Config.ZIGZAG_STRATEGIES.get(strategy_name)
                if not strategy_cfg:
                    errors.append(f"Unknown strategy: {strategy_name}")
                    continue

                for interval in selected_intervals:
                    if interval not in strategy_cfg:
                        errors.append(f"Interval '{interval}' not available for strategy '{strategy_name}'")
                        continue

                    params = strategy_cfg[interval]
                    try:
                        df = mod.buscar_dados(ticker, mod.Config.DATA_PERIOD, interval)
                        df = mod.calcular_indicadores(df)
                        pivots = mod.calcular_zigzag_oficial(df, params['depth'], params['deviation'])

                        found_now: List[Dict[str, Any]] = []

                        # Always run all pattern detectors
                        if len(pivots) >= 7:
                            found_now.extend(mod.identificar_padroes_hns(pivots, df))

                        found_now.extend(mod.identificar_padroes_double_top_bottom(pivots, df))

                        candidatos_ttb = mod.identificar_padroes_ttb(pivots)
                        for cand in candidatos_ttb:
                            scored = mod.validate_and_score_triple_pattern(cand, df)
                            if scored:
                                found_now.append(scored)

                        for p in found_now:
                            p['strategy'] = strategy_name
                            p['timeframe'] = interval
                            p['ticker'] = ticker

                        all_found.extend(found_now)
                    except Exception as e_inner:
                        errors.append(f"{ticker}/{interval}/{strategy_name}: {str(e_inner)}")
                        continue

            # CSV generation disabled: only list what would be written

            summary = {
                "ok": True,
                "coin_id": coin_id,
                "vs_currency": (vs_currency or self.default_vs_currency).lower(),
                "term_type": term_type,
                "period": effective_period,
                "strategies": selected_strategies,
                "intervals": selected_intervals,
                "patterns": ["HNS", "DTB", "TTB"],
                "found_count": len(all_found),
                "sample": all_found[:5],
                "records": all_found,
                "errors": errors,
            }
            return json.dumps(summary, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "ok": False,
                "error": str(e),
                "coin_id": coin_id,
            }, ensure_ascii=False)


