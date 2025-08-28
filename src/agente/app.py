from typing import Any, Dict, Optional
from dotenv import load_dotenv
import os
import uuid
import json
import argparse
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
from agno.tools.googlesearch import GoogleSearchTools
from coingeckoToolKit import CoinGeckoToolKit
from coindeskToolKit import CoinDeskToolKit

load_dotenv()

# Description and instructions aligned with the new purpose: JSON-only output (no server)
API_DESCRIPTION: str = (
    "Feature-based crypto analysis via an internal Agent, returning structured JSON. "
    "No HTTP server at this stage; invoke via CLI or programmatically."
)

API_INSTRUCTIONS = [
    "Call run_feature(feature, **params) or use the CLI to execute features",
    "Identify assets by CoinGecko coin_id (e.g., 'bitcoin', 'ethereum')",
    "Output is always JSON with the shape: { ok, feature, data, errors, meta }",
]


def build_response(
    *, ok: bool, feature: str, data: Dict[str, Any], errors: Optional[list] = None, meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Builds the standard JSON response envelope."""

    return {
        "ok": ok,
        "feature": feature,
        "data": data or {},
        "errors": errors or [],
        "meta": meta or {},
    }


def handle_analysis(
    *, coin_id: str, vs_currency: Optional[str] = None, timeframe: Optional[str] = None
) -> Dict[str, Any]:
    """Calls the internal Agent to produce an analysis and returns JSON-safe data."""

    vs = vs_currency or os.getenv("DEFAULT_VS_CURRENCY", "usd")

    analysis_agent = Agent(
        model=OpenRouter(id="openai/gpt-5-mini"),
        tools=[
            ReasoningTools(add_instructions=True),
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(),
            CoinGeckoToolKit(),
            CoinDeskToolKit(timeout=30),
        ],
        description=(
            "You are a backend analysis Agent. You DO NOT chat. You strictly return structured, machine-consumable JSON. "
            "Primary capabilities: market/technical/news analysis using toolkits; never include emojis or prose."
        ),
        instructions=[
            "Always return valid JSON objects only. No markdown, no text outside JSON.",
            "Keys should be snake_case.",
            "Identify assets by CoinGecko coin_id.",
            "If timeframe <= 90 use RSI availability; if > 90 omit RSI and emphasize MACD/SMA.",
            "Include fields: current, technical, supports_resistances, news_summary, meta.",
            "meta should include timeframe and vs_currency.",
        ],
        markdown=False,
    )

    prompt = (
        "Build a JSON object for an integrated crypto analysis with keys: "
        "current, technical, supports_resistances, news_summary, meta. "
        f"Use coin_id='{coin_id}', vs_currency='{vs}', timeframe='{timeframe or ''}'. "
        "Use CoinGeckoToolKit and CoinDeskToolKit. Do not include any text outside JSON."
    )

    # Execute and ensure the output is a dict
    result = analysis_agent.run(prompt)
    if isinstance(result, dict):
        return result
    # Attempt to coerce JSON-like strings
    try:
        import json as _json

        return _json.loads(str(result))
    except Exception:
        return {
            "current": {},
            "technical": {},
            "supports_resistances": {},
            "news_summary": {},
            "meta": {"coin_id": coin_id, "vs_currency": vs, "timeframe": timeframe},
        }


FEATURE_HANDLERS = {
    "analysis": handle_analysis,
}


def run_feature(feature: str, **params: Any) -> Dict[str, Any]:
    """Runs a feature handler and wraps the result in the standard envelope."""

    request_id: str = str(uuid.uuid4())
    if feature not in FEATURE_HANDLERS:
        return build_response(
            ok=False,
            feature=feature,
            data={},
            errors=[f"Unsupported feature '{feature}'"],
            meta={"request_id": request_id},
        )

    try:
        handler = FEATURE_HANDLERS[feature]

        if feature == "analysis":
            coin_id = params.get("coin_id")
            if not coin_id:
                return build_response(
                    ok=False,
                    feature=feature,
                    data={},
                    errors=["'coin_id' is required for feature 'analysis'"],
                    meta={"request_id": request_id},
                )
            data = handler(
                coin_id=coin_id,
                vs_currency=params.get("vs_currency"),
                timeframe=params.get("timeframe"),
            )
        else:
            data = handler()  # type: ignore[misc]

        return build_response(
            ok=True,
            feature=feature,
            data=data,
            errors=[],
            meta={"request_id": request_id},
        )
    except Exception as exc:  # noqa: BLE001
        return build_response(
            ok=False,
            feature=feature,
            data={},
            errors=[str(exc)],
            meta={"request_id": request_id},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=API_DESCRIPTION)
    parser.add_argument("feature", type=str,
                        help="Feature to execute (e.g., analysis)")
    parser.add_argument("--coin_id", type=str,
                        help="CoinGecko coin_id (required for analysis)")
    parser.add_argument("--vs_currency", type=str,
                        help="Fiat currency code (e.g., usd)")
    parser.add_argument("--timeframe", type=str,
                        help="Time horizon hint (e.g., 30, 365)")

    args = parser.parse_args()

    payload = run_feature(
        args.feature,
        coin_id=args.coin_id,
        vs_currency=args.vs_currency,
        timeframe=args.timeframe,
    )
    print(json.dumps(payload, ensure_ascii=False))
