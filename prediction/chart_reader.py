"""
prediction/chart_reader.py — Read a chart screenshot with Claude vision.

The screenshot is only an *input method*: the vision model extracts the ticker
and timeframe from the image, and the caller then pulls real OHLCV for that
symbol and runs the normal expected-move math. We deliberately do NOT ask the
model to read prices or predict direction off the pixels — that would be the
low-information, coin-flip version. Identify the chart; trust real data for the
numbers.

Requires ANTHROPIC_API_KEY in the environment (the SDK reads it automatically).
"""

from __future__ import annotations

import json

# valid interval codes the rest of the app understands
_VALID_INTERVALS = {"1m", "5m", "15m", "1h", "1d"}

_SYSTEM = (
    "You read financial chart screenshots (TradingView, brokerages, etc.). "
    "Identify the ticker symbol and the chart timeframe. Report only what is "
    "visibly labelled — never guess prices, and never opine on direction."
)

# structured-output schema: guarantees a parseable, fixed shape
_SCHEMA = {
    "type": "object",
    "properties": {
        "is_chart": {"type": "boolean"},
        "ticker": {"type": "string"},            # "" if not legible
        "timeframe": {
            "type": "string",
            "enum": ["1m", "5m", "15m", "1h", "1d", "unknown"],
        },
        "confidence": {"type": "number"},         # 0..1
    },
    "required": ["is_chart", "ticker", "timeframe", "confidence"],
    "additionalProperties": False,
}


def read_chart(image_b64: str, media_type: str = "image/png") -> dict:
    """
    Extract {ticker, interval, is_chart, confidence} from a base64 chart image.

    `interval` is normalised to one of the app's codes (defaults to "1d" when the
    timeframe is unknown). Raises if the SDK/key is unavailable so the caller can
    surface a clear error.
    """
    import anthropic  # imported lazily so the app boots without the dep present

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        thinking={"type": "disabled"},           # simple extraction — no thinking
        system=_SYSTEM,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "What ticker and timeframe does this chart show? "
                        "If it isn't a price chart, set is_chart to false."
                    ),
                },
            ],
        }],
    )

    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)

    tf = data.get("timeframe", "unknown")
    interval = tf if tf in _VALID_INTERVALS else "1d"

    return {
        "is_chart": bool(data.get("is_chart", False)),
        "ticker": (data.get("ticker") or "").strip().upper(),
        "interval": interval,
        "timeframe_detected": tf,
        "confidence": float(data.get("confidence", 0.0)),
    }
