# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo actually contains

Two distinct codebases live side by side:

1. **`dashboard.py`** — the production Flask web app behind https://chartedge.trade. ~13k lines, single file. This is where 95% of work happens.
2. **`main.py` + `training/` + `models/` + `prediction/`** — a menu-driven CLI that trains and runs the LightGBM-GRU hybrid volatility model from Liao, Chen & Cai (2024). This is the original research code described in `README.md`. The web app consumes the trained model artifacts in `saved_models/`.

The README in this repo describes only the ML code. It does *not* describe `dashboard.py` — which is the bigger, more important codebase.

## Running

| Goal | Command |
|---|---|
| Run the web app locally | `python3 dashboard.py` (binds `0.0.0.0:$PORT`, default 5000) |
| Run the ML training CLI | `python3 main.py` |
| Install deps | `pip install -r requirements.txt` (Apple Silicon needs `tensorflow-macos` + `tensorflow-metal` first) |

`Procfile`, `Dockerfile`, and `nixpacks.toml` all just run `python3 dashboard.py` — there is no gunicorn/uvicorn wrapper, no separate worker. Flask's built-in dev server serves production.

There are no tests, no lint config, and no build step. Validation is `python3 -c "import ast; ast.parse(open('dashboard.py').read())"` to catch syntax errors after large edits.

## dashboard.py architecture (read this before editing)

**All pages are Python string templates** rendered via `render_template_string`. There are ~90 `*_HTML = """..."""` constants in `dashboard.py`. There is no `templates/` directory, no static templates, no Jinja files on disk. CSS and JS are inlined inside each HTML string. Edits to a page mean editing a Python triple-quoted string.

**Shared chrome is interpolated**. The following constants are spliced into most page templates via `""" + _NAV_CSS + """`-style concatenation:

- `_META` — `<head>` meta tags, GA, theme bootstrap
- `_NAV_CSS` — fixed scroll-aware nav styling, pill container, tubelight dropdown effect
- `_NAV_LINKS` — the nav menu HTML (Tools/Crypto/Community/Pricing/Account dropdowns); uses `{% if nav_plan == 'free' %}` to gate items
- `_THEME_JS` — theme toggle, dropdown logic, mobile nav, scroll-aware nav handler

**`nav_plan` is auto-injected** by the `inject_nav_plan` context processor (line ~294). Every template gets it for free; don't pass it manually.

**Plan gating pattern**: routes use `_get_user_plan(session.get("user_id"))` returning `"free" | "basic" | "pro"` and redirect to `/pricing?upgrade=<feature>` when access is denied. The same check repeats inside API endpoints — never trust client-side gating.

**A few pages have self-contained nav CSS** (login/register/pricing legacy paths) that don't pull in `_NAV_CSS`. Updates to shared nav styling won't reach these — search for `nav { background` to find inline nav declarations.

## Pine Script expiry injection (security-critical)

`_inject_expiry()` (line ~593) weaves three obfuscated expiry checks into every generated Pine Script via `/api/pine`. Scripts break at midnight after generation, forcing users to come back to the site rather than re-share a one-time copy. The watermark at the top of every generated script (`// ChartEdge.trade · <user> · <date> · chartedge.trade`) is intentional — do not remove it, and do not refactor `_inject_expiry` casually.

## Data + state

- **Postgres only.** `app.db` exists as a leftover SQLite file but is unused. `DATABASE_URL` is required to boot. `init_db()` + `_migrate_pg()` run at import time.
- **DB helpers**: `_q` (rows), `_one` (one row), `_scalar` (one value), `_run` (write), `_tx()` (transaction context). All return `RealDictCursor` rows.
- **Sessions** are Flask's signed cookies (`SECRET_KEY` env). User id lives in `session["user_id"]`.
- **No ORM**, no migrations framework. Schema changes go into `_migrate_pg()`.

## Required environment variables

`DATABASE_URL`, `SECRET_KEY`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `RESEND_API_KEY`, `FINNHUB_API_KEY`, `HCAPTCHA_SECRET`, `ADMIN_PASSWORD`, `GA_ID`, `PORT`. None of these are checked at startup — missing keys fail at first use.

## Brand naming

The brand is **ChartEdge.trade** (camelcase, `.trade` suffix visible). The site logo is split into colored spans: `<span>Chart</span><span style="color:#58a6ff">Edge</span><span class="muted">.trade</span>`. The `.trade` suffix is part of the displayed brand — keep it in titles, footers, emails, OG meta, and the Studio header.

## ML side (`main.py`, `models/`, `prediction/`, `training/`)

`README.md` covers this fully — pipeline is yfinance → 40+ features → LightGBM → GRU corrector → final vol prediction. The Flask app reads the trained artifact via `prediction/forecast_exporter.py` to power `/api/forecast` and `/volforecast`. Only relevant when changing forecast output that the web app consumes. `LIMITATIONS.md` is a candid list of what the model can't do — keep it accurate if you ship changes that affect the forecast contract.

## Editing patterns to follow

- **Replacing large HTML blocks**: use `Edit` with the exact old `_HTML` string as `old_string`. The strings are big but Edit handles them fine; reading them in chunks first is usually unnecessary if you already know what to swap.
- **Validating after edits**: `python3 -c "import ast; ast.parse(open('dashboard.py').read())"` catches Python-level syntax errors (an unterminated triple-quoted string, a bad f-string, etc.). Browser-level rendering issues won't show up — the dev server has to be running for that.
- **Avoid renaming `nav-links` and `mobile-nav` IDs** — `_THEME_JS` and the per-page hamburger button reference them by exact name.
- **Don't add unwired social buttons** to auth pages. Only Google OAuth is implemented (`/login/google`). Microsoft/Apple/SSO buttons would lead nowhere.
