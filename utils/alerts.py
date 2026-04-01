"""
utils/alerts.py — Push notification alerts via ntfy.sh (free, no account needed).

Setup:
  1. Install the ntfy app on your phone (iOS or Android)
  2. Subscribe to your topic (e.g. "stockvol-ayden")
  3. Set NTFY_TOPIC in your .env file

Alerts fire when:
  - Forecast volatility exceeds the HIGH threshold (spike alert)
  - Direction confidence exceeds CONF_THRESHOLD (strong signal alert)
"""

from __future__ import annotations

import os
from utils.logger import get_logger

log = get_logger(__name__)

CONF_THRESHOLD = 0.65   # minimum confidence to trigger a direction alert


def send_alert(
    topic: str,
    title: str,
    message: str,
    priority: str = "default",
    tags: list[str] | None = None,
) -> bool:
    """
    Send a push notification via ntfy.sh.

    Parameters
    ----------
    topic    : your ntfy topic (subscribe to this in the ntfy app)
    title    : notification title
    message  : notification body
    priority : "low", "default", "high", "urgent"
    tags     : emoji tags shown in notification (e.g. ["warning", "chart"])

    Returns True if sent successfully.
    """
    try:
        import urllib.request
        import urllib.error

        headers = {
            "Title":    title.encode("utf-8"),
            "Priority": priority.encode("utf-8"),
        }
        if tags:
            headers["Tags"] = ",".join(tags).encode("utf-8")

        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
            if ok:
                log.info("Alert sent → ntfy.sh/%s: %s", topic, title)
            return ok
    except Exception as exc:
        log.warning("Alert failed: %s", exc)
        return False


def check_and_alert(
    topic: str,
    ticker: str,
    future_values: list[float],
    high_thresh: float,
    direction_up: bool | None,
    direction_conf: float | None,
    seen_events: set,
) -> None:
    """
    Check forecast data and fire alerts if conditions are met.

    Parameters
    ----------
    seen_events : set of event keys already alerted (prevents spam)
    """
    # ── Volatility spike alert ────────────────────────────────────────────────
    max_forecast = max(future_values)
    spike_key    = f"{ticker}_spike_{round(max_forecast, 4)}"

    if max_forecast > high_thresh and spike_key not in seen_events:
        bar = future_values.index(max_forecast) + 1
        msg = (
            f"{ticker} volatility spike predicted in +{bar * 5} min\n"
            f"Forecast RV: {max_forecast:.5f}  (threshold: {high_thresh:.5f})"
        )
        sent = send_alert(
            topic=topic,
            title=f"⚠️ {ticker} Volatility Spike",
            message=msg,
            priority="high",
            tags=["warning", "chart_increasing"],
        )
        if sent:
            seen_events.add(spike_key)

    # ── Strong direction signal alert ─────────────────────────────────────────
    if direction_conf is not None and direction_conf >= CONF_THRESHOLD:
        arrow    = "▲ UP" if direction_up else "▼ DOWN"
        dir_key  = f"{ticker}_dir_{direction_up}_{round(direction_conf, 2)}"

        if dir_key not in seen_events:
            msg = (
                f"{ticker} strong direction signal: {arrow}\n"
                f"Confidence: {direction_conf:.0%} over next 30 min"
            )
            sent = send_alert(
                topic=topic,
                title=f"{'📈' if direction_up else '📉'} {ticker} {arrow} {direction_conf:.0%}",
                message=msg,
                priority="high",
                tags=["up" if direction_up else "down"],
            )
            if sent:
                seen_events.add(dir_key)
