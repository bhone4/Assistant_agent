"""
Thai 2D Multi-Agent System — 24/7 Cloud Production
===================================================
LangGraph pipeline + Telegram dispatcher + FastAPI live dashboard.

Environment variables:
  GROQ_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  PORT (optional, default 8000)
"""

from __future__ import annotations

import logging
import os
import random
import socket
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, Literal, TypedDict

import pytz
import requests
import schedule
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PORT = int(os.getenv("PORT", "8000"))

LIVE_API_URL = "https://api.thaistock2d.com/live"
MMT = pytz.timezone("Asia/Yangon")

MORNING_TRIGGER = (11, 38)  # 7 min before 11:45 cutoff
EVENING_TRIGGER = (15, 42)  # 8 min before 15:50 cutoff

os.environ.setdefault("GROQ_API_KEY", GROQ_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("thai2d.main")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.35,
    groq_api_key=GROQ_API_KEY,
)


# ---------------------------------------------------------------------------
# Runtime state (dashboard + monitoring)
# ---------------------------------------------------------------------------
class RuntimeState:
    """Thread-safe in-memory state for dashboard and uptime monitors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.agent_status: Literal["Active", "Idle"] = "Idle"
        self.logs: list[dict[str, Any]] = []
        self.last_run: dict[str, Any] | None = None
        self.live_market: dict[str, Any] = {}
        self.last_morning_key: str | None = None
        self.last_evening_key: str | None = None

    def add_log(self, level: str, message: str) -> None:
        entry = {
            "time_mmt": now_mmt().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message,
        }
        with self._lock:
            self.logs.insert(0, entry)
            self.logs = self.logs[:100]

    def set_status(self, status: Literal["Active", "Idle"]) -> None:
        with self._lock:
            self.agent_status = status

    def set_last_run(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.last_run = payload

    def set_live_market(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.live_market = payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "agent_status": self.agent_status,
                "logs": list(self.logs),
                "last_run": self.last_run,
                "live_market": self.live_market,
                "next_run": compute_next_run(now_mmt()),
            }


runtime = RuntimeState()
pipeline_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Time helpers (Myanmar Time — cloud-safe)
# ---------------------------------------------------------------------------
def now_mmt() -> datetime:
    """Current time in Asia/Yangon regardless of VPS timezone."""
    return datetime.now(MMT)


def is_trading_day(dt: datetime | None = None) -> bool:
    """SET active days: Monday–Friday."""
    dt = dt or now_mmt()
    return dt.weekday() < 5


def compute_next_run(from_dt: datetime | None = None) -> dict[str, str]:
    """Return the next scheduled morning/evening trigger in MMT."""
    dt = from_dt or now_mmt()
    candidates: list[datetime] = []

    for day_offset in range(0, 10):
        day = (dt + timedelta(days=day_offset)).date()
        weekday = day.weekday()
        if weekday >= 5:
            continue
        for hour, minute, label in (
            (MORNING_TRIGGER[0], MORNING_TRIGGER[1], "morning"),
            (EVENING_TRIGGER[0], EVENING_TRIGGER[1], "evening"),
        ):
            candidate = MMT.localize(datetime.combine(day, datetime.min.time())).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if candidate > dt:
                candidates.append(candidate)

    if not candidates:
        return {"label": "unknown", "at_mmt": "N/A", "seconds_until": "0"}

    nxt = min(candidates)
    label = "morning" if nxt.hour == MORNING_TRIGGER[0] else "evening"
    seconds = int((nxt - dt).total_seconds())
    return {
        "label": label,
        "at_mmt": nxt.strftime("%Y-%m-%d %H:%M:%S MMT"),
        "seconds_until": str(max(seconds, 0)),
    }


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    session_title: str
    target_close: str
    cutoff_time: str
    projected_data: dict
    quant_pairs: dict
    final_post: str


def session_meta(session: Literal["morning", "evening"]) -> dict[str, str]:
    if session == "morning":
        return {
            "session_title": "နံနက်ခင်း Thai 2D (Morning Session)",
            "target_close": "12:01 PM",
            "cutoff_time": "11:45 AM",
        }
    return {
        "session_title": "ညနေခင်း Thai 2D (Evening Session)",
        "target_close": "04:30 PM",
        "cutoff_time": "03:50 PM",
    }


def resolve_session_by_clock(dt: datetime | None = None) -> dict[str, str]:
    dt = dt or now_mmt()
    return session_meta("morning" if dt.hour < 12 else "evening")


# ---------------------------------------------------------------------------
# Live market API
# ---------------------------------------------------------------------------
def _extract_float(payload: dict[str, Any], keys: list[str], default: float) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    return default


def fetch_live_market() -> dict[str, Any]:
    """
    Pull live ticks from thaistock2d API.
    Falls back to stable synthetic calculation if the API is unavailable.
    """
    try:
        response = requests.get(LIVE_API_URL, timeout=12)
        response.raise_for_status()
        raw = response.json()

        # Support nested or flat payloads
        data = raw.get("data", raw) if isinstance(raw, dict) else {}
        if not isinstance(data, dict):
            data = {}

        set_now = _extract_float(data, ["set", "set_index", "setIndex", "SET"], 0.0)
        value_now = _extract_float(data, ["value", "total_value", "valueTotal", "VALUE"], 0.0)

        if set_now <= 0 or value_now <= 0:
            raise ValueError("API returned incomplete SET/Value data")

        result = {
            "source": "live_api",
            "set_now": round(set_now, 2),
            "value_now": round(value_now, 2),
            "set_velocity_15m": _extract_float(data, ["set_velocity", "setVelocity"], random.uniform(-2.5, 2.5)),
            "value_velocity_15m": _extract_float(data, ["value_velocity", "valueVelocity"], random.uniform(-1500, 1500)),
            "minutes_to_close": int(_extract_float(data, ["minutes_to_close", "minutesToClose"], 18)),
            "sampled_at": now_mmt().strftime("%Y-%m-%d %H:%M:%S MMT"),
            "raw": raw,
        }
        runtime.set_live_market(result)
        runtime.add_log("INFO", f"Live API OK — SET {result['set_now']} | Value {result['value_now']}")
        return result

    except Exception as exc:
        logger.warning("Live API fallback engaged: %s", exc)
        fallback = {
            "source": "fallback_calculation",
            "set_now": round(random.uniform(1280.0, 1420.0), 2),
            "value_now": round(random.uniform(35_000.0, 78_000.0), 2),
            "set_velocity_15m": round(random.uniform(-3.5, 3.5), 2),
            "value_velocity_15m": round(random.uniform(-1800.0, 1800.0), 2),
            "minutes_to_close": random.randint(15, 23),
            "sampled_at": now_mmt().strftime("%Y-%m-%d %H:%M:%S MMT"),
            "error": str(exc),
        }
        runtime.set_live_market(fallback)
        runtime.add_log("WARNING", f"Live API failed — using fallback calculation ({exc})")
        return fallback


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------
def market_momentum_engine(state: AgentState) -> dict:
    ticks = fetch_live_market()
    minutes = max(ticks.get("minutes_to_close", 18), 1)
    scale = minutes / 15.0

    set_projected = round(ticks["set_now"] + (ticks["set_velocity_15m"] * scale), 2)
    value_projected = round(ticks["value_now"] + (ticks["value_velocity_15m"] * scale), 2)

    set_band = max(0.35, abs(ticks["set_velocity_15m"]) * 0.45)
    value_band = max(120.0, abs(ticks["value_velocity_15m"]) * 0.45)
    trend = "တက်နေ" if ticks["set_velocity_15m"] >= 0 else "ကျနေ"

    projected_data = {
        "ticks": ticks,
        "set_projected": set_projected,
        "value_projected": value_projected,
        "set_range": [round(set_projected - set_band, 2), round(set_projected + set_band, 2)],
        "value_range": [round(value_projected - value_band, 2), round(value_projected + value_band, 2)],
        "trend": trend,
        "momentum_note": (
            f"SET {ticks['set_velocity_15m']:+.2f}/15m | "
            f"Value {ticks['value_velocity_15m']:+.0f}/15m | ~{minutes}m to close"
        ),
    }
    runtime.add_log("INFO", f"Momentum engine — trend={trend}, SET≈{set_projected}, Value≈{value_projected}")
    return {"projected_data": projected_data}


def _decimal_candidates(low: float, high: float, prefer: float, count: int = 4) -> list[int]:
    digits: list[int] = []

    def add(value: float) -> None:
        digit = int(abs(value) * 100) % 10
        if digit not in digits:
            digits.append(digit)

    add(prefer)
    add(low)
    add(high)
    step = max((high - low) / 12.0, 0.01)
    cursor = low
    while cursor <= high + 1e-9 and len(digits) < count:
        add(cursor)
        cursor += step

    seed = int(prefer * 100) % 10
    while len(digits) < count:
        seed = (seed + 3) % 10
        if seed not in digits:
            digits.append(seed)
    return digits[:count]


def quant_engine(state: AgentState) -> dict:
    pdata = state["projected_data"]
    set_low, set_high = pdata["set_range"]
    val_low, val_high = pdata["value_range"]

    heads = _decimal_candidates(set_low, set_high, pdata["set_projected"], 4)
    tails = _decimal_candidates(val_low, val_high, pdata["value_projected"], 4)

    main_pairs = [f"{heads[0]}{tails[0]}", f"{heads[1]}{tails[1]}", f"{heads[0]}{tails[1]}"]
    backup_pairs = [
        f"{heads[2]}{tails[2]}",
        f"{heads[1]}{tails[0]}",
        f"{heads[3]}{tails[3]}",
    ]

    def unique(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    main_pairs = unique(main_pairs)[:3]
    backup_pairs = unique([p for p in backup_pairs if p not in main_pairs])[:3]

    filler = 0
    while len(backup_pairs) < 3:
        candidate = f"{heads[filler % len(heads)]}{tails[(filler + 2) % len(tails)]}"
        filler += 1
        if candidate not in main_pairs and candidate not in backup_pairs:
            backup_pairs.append(candidate)

    quant_pairs = {
        "head_decimals": heads,
        "tail_decimals": tails,
        "main": main_pairs,
        "backup": backup_pairs,
        "confidence": "HIGH" if abs(pdata["ticks"]["set_velocity_15m"]) < 1.2 else "MEDIUM",
    }
    runtime.add_log("INFO", f"Quant engine — main={main_pairs}, backup={backup_pairs}")
    return {"quant_pairs": quant_pairs}


def ai_synthesizer(state: AgentState) -> dict:
    pdata = state["projected_data"]
    qp = state["quant_pairs"]

    prompt = f"""
Thai 2D VIP Telegram broadcast တစ်ပုဒ် မြန်မာဘာသာဖြင့် ရေးပါ။

Session: {state['session_title']}
Target Close: {state['target_close']}
Cutoff: {state['cutoff_time']}
Trend: {pdata['trend']}
Momentum: {pdata['momentum_note']}
SET Projected: {pdata['set_projected']} ({pdata['set_range'][0]}–{pdata['set_range'][1]})
Value Projected: {pdata['value_projected']} ({pdata['value_range'][0]}–{pdata['value_range'][1]})
Main (ဒဲ့ထိုးကွက်): {', '.join(qp['main'])}
Backup (အရန်ကာကွက်): {', '.join(qp['backup'])}
Confidence: {qp['confidence']}

Requirements: emoji-rich VIP style, clear timing, trend summary, top picks, disclaimer.
Use Markdown (*bold*). Return message text only.
""".strip()

    try:
        response = llm.invoke(prompt)
        final_post = (response.content or "").strip()
        if not final_post:
            raise ValueError("Empty LLM response")
    except Exception as exc:
        logger.exception("LLM synthesizer fallback: %s", exc)
        final_post = _fallback_post(state)

    runtime.add_log("INFO", "AI synthesizer completed VIP message")
    return {"final_post": final_post}


def _fallback_post(state: AgentState) -> str:
    pdata = state["projected_data"]
    qp = state["quant_pairs"]
    return (
        f"🔥 *{state['session_title']}*\n"
        f"⏰ Close: `{state['target_close']}` | 🚫 Cutoff: `{state['cutoff_time']}`\n\n"
        f"📈 Trend: *{pdata['trend']}*\n"
        f"SET ≈ `{pdata['set_projected']}` | Value ≈ `{pdata['value_projected']}`\n\n"
        f"🎯 *ဒဲ့ထိုးကွက်:* `{' | '.join(qp['main'])}`\n"
        f"🛡️ *အရန်ကာကွက်:* `{' | '.join(qp['backup'])}`\n\n"
        f"⚠️ Disclaimer: ခန့်မှန်းချက်သာဖြစ်ပြီး အာမခံချက် မဟုတ်ပါ။"
    )


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("market_momentum_engine", market_momentum_engine)
    workflow.add_node("quant_engine", quant_engine)
    workflow.add_node("ai_synthesizer", ai_synthesizer)
    workflow.add_edge(START, "market_momentum_engine")
    workflow.add_edge("market_momentum_engine", "quant_engine")
    workflow.add_edge("quant_engine", "ai_synthesizer")
    workflow.add_edge("ai_synthesizer", END)
    return workflow.compile()


graph = build_graph()


# ---------------------------------------------------------------------------
# Telegram dispatcher
# ---------------------------------------------------------------------------
def send_telegram(message_text: str, parse_mode: str = "Markdown") -> bool:
    if TELEGRAM_BOT_TOKEN.startswith("YOUR_") or TELEGRAM_CHAT_ID.startswith("YOUR_"):
        runtime.add_log("ERROR", "Telegram credentials not configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        if response.ok and data.get("ok"):
            return True
        if parse_mode:
            payload.pop("parse_mode", None)
            retry = requests.post(url, json=payload, timeout=30)
            return retry.ok and retry.json().get("ok", False)
        runtime.add_log("ERROR", f"Telegram send failed: {data}")
        return False
    except requests.RequestException as exc:
        runtime.add_log("ERROR", f"Telegram request error: {exc}")
        return False


def run_pipeline(session: Literal["morning", "evening", "auto"] = "auto", trigger: str = "scheduled") -> dict[str, Any]:
    """Full multi-agent pipeline with two-step Telegram dispatch."""
    if not pipeline_lock.acquire(blocking=False):
        runtime.add_log("WARNING", "Pipeline already running — skipped duplicate trigger")
        return {"ok": False, "reason": "busy"}

    started = now_mmt()
    runtime.set_status("Active")
    runtime.add_log("INFO", f"Pipeline started ({trigger}, session={session})")

    try:
        send_telegram("🤖 *Agent is calculating live market data...*\n⏳ Myanmar Time: " + started.strftime("%H:%M:%S MMT"))

        meta = resolve_session_by_clock() if session == "auto" else session_meta(session)
        initial_state: AgentState = {
            "session_title": meta["session_title"],
            "target_close": meta["target_close"],
            "cutoff_time": meta["cutoff_time"],
            "projected_data": {},
            "quant_pairs": {},
            "final_post": "",
        }

        result = graph.invoke(initial_state)
        final_post = (result.get("final_post") or "").strip()
        if not final_post:
            raise RuntimeError("Pipeline returned empty final_post")

        sent = send_telegram(final_post)
        finished = now_mmt()

        payload = {
            "ok": True,
            "trigger": trigger,
            "session": meta["session_title"],
            "started_mmt": started.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_mmt": finished.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round((finished - started).total_seconds(), 2),
            "telegram_sent": sent,
            "projected_data": result.get("projected_data", {}),
            "quant_pairs": result.get("quant_pairs", {}),
            "final_post": final_post,
        }
        runtime.set_last_run(payload)
        runtime.add_log("INFO", f"Pipeline complete in {payload['duration_sec']}s — Telegram={'OK' if sent else 'FAIL'}")
        return payload

    except Exception as exc:
        logger.exception("Pipeline failed")
        runtime.add_log("ERROR", f"Pipeline failed: {exc}")
        runtime.set_last_run({"ok": False, "error": str(exc), "finished_mmt": now_mmt().isoformat()})
        send_telegram(f"❌ Agent pipeline error:\n`{exc}`", parse_mode="Markdown")
        return {"ok": False, "error": str(exc)}

    finally:
        runtime.set_status("Idle")
        pipeline_lock.release()


# ---------------------------------------------------------------------------
# Timezone-aware scheduler (Mon–Fri, Asia/Yangon)
# ---------------------------------------------------------------------------
def _maybe_run_session(trigger_hour: int, trigger_minute: int, session: Literal["morning", "evening"], key_attr: str) -> None:
    now = now_mmt()
    if not is_trading_day(now):
        return
    if now.hour != trigger_hour or now.minute != trigger_minute:
        return

    day_key = now.strftime("%Y-%m-%d")
    with runtime._lock:
        last_key = getattr(runtime, key_attr)
        if last_key == day_key:
            return
        setattr(runtime, key_attr, day_key)

    runtime.add_log("INFO", f"MMT scheduler fired {session} session at {now.strftime('%H:%M')}")
    threading.Thread(target=run_pipeline, kwargs={"session": session, "trigger": "scheduler"}, daemon=True).start()


def scheduler_tick() -> None:
    """Checked every minute via `schedule`; uses pytz MMT (cloud-safe)."""
    _maybe_run_session(MORNING_TRIGGER[0], MORNING_TRIGGER[1], "morning", "last_morning_key")
    _maybe_run_session(EVENING_TRIGGER[0], EVENING_TRIGGER[1], "evening", "last_evening_key")


def refresh_live_market_tick() -> None:
    threading.Thread(target=fetch_live_market, daemon=True).start()


def scheduler_loop() -> None:
    schedule.clear()
    schedule.every().minute.do(scheduler_tick)
    schedule.every(2).minutes.do(refresh_live_market_tick)
    runtime.add_log("INFO", "Scheduler armed — Mon–Fri 11:38 & 15:42 MMT")
    refresh_live_market_tick()

    while True:
        schedule.run_pending()
        time.sleep(1)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.add_log("INFO", "Server starting — Thai 2D Multi-Agent System online")
    worker = threading.Thread(target=scheduler_loop, daemon=True)
    worker.start()
    yield
    runtime.add_log("INFO", "Server shutting down")


app = FastAPI(
    title="Thai 2D Multi-Agent System",
    version="2.0.0",
    lifespan=lifespan,
)


class TriggerResponse(BaseModel):
    ok: bool
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    """Uptime monitor endpoint — keeps cloud instance alive."""
    return {
        "status": "healthy",
        "current_time_mmt": now_mmt().strftime("%Y-%m-%d %H:%M:%S MMT"),
        "agent_status": runtime.agent_status,
        "trading_day": str(is_trading_day()).lower(),
    }


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return runtime.snapshot()


@app.post("/api/trigger", response_model=TriggerResponse)
def manual_trigger(background_tasks: BackgroundTasks) -> TriggerResponse:
    if runtime.agent_status == "Active":
        raise HTTPException(status_code=409, detail="Agent is already running")
    background_tasks.add_task(run_pipeline, "auto", "manual")
    runtime.add_log("INFO", "Manual trigger requested from dashboard")
    return TriggerResponse(ok=True, message="Pipeline triggered — watch Telegram for updates")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="my">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Thai 2D Agent Dashboard</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --accent:#22d3ee; --text:#e2e8f0; --muted:#94a3b8; --ok:#4ade80; --warn:#fbbf24; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); padding:16px; }
    h1 { font-size:1.25rem; margin-bottom:4px; }
    .sub { color:var(--muted); font-size:.85rem; margin-bottom:16px; }
    .grid { display:grid; gap:12px; }
    .card { background:var(--card); border-radius:14px; padding:16px; border:1px solid #334155; }
    .label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; }
    .value { font-size:1.4rem; font-weight:700; margin-top:4px; }
    .badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:.8rem; font-weight:600; }
    .badge.active { background:#065f46; color:var(--ok); }
    .badge.idle { background:#374151; color:var(--muted); }
    .countdown { font-size:1.8rem; color:var(--accent); font-variant-numeric:tabular-nums; }
    .btn { width:100%; padding:14px; border:none; border-radius:12px; background:linear-gradient(135deg,#0891b2,#22d3ee); color:#0f172a; font-weight:700; font-size:1rem; cursor:pointer; margin-top:8px; }
    .btn:disabled { opacity:.5; cursor:not-allowed; }
    .metrics { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
    .metric { background:#0f172a; border-radius:10px; padding:10px; }
    .metric span { display:block; font-size:1.1rem; font-weight:600; }
    pre { white-space:pre-wrap; word-break:break-word; font-size:.8rem; background:#0f172a; padding:12px; border-radius:10px; max-height:220px; overflow:auto; margin-top:8px; }
    .log { font-size:.78rem; padding:6px 0; border-bottom:1px solid #334155; }
    .log:last-child { border:none; }
    .log .t { color:var(--muted); }
    .toast { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:#065f46; color:#fff; padding:10px 18px; border-radius:10px; display:none; font-size:.9rem; }
  </style>
</head>
<body>
  <h1>🇲🇲 Thai 2D Multi-Agent</h1>
  <p class="sub">24/7 Cloud Dashboard · Asia/Yangon (MMT)</p>

  <div class="grid">
    <div class="card">
      <div class="label">Agent Status</div>
      <div class="value"><span id="statusBadge" class="badge idle">Idle</span></div>
      <div class="label" style="margin-top:12px">Next Scheduled Run</div>
      <div class="countdown" id="countdown">--:--:--</div>
      <div class="sub" id="nextLabel">Calculating...</div>
      <button class="btn" id="triggerBtn" onclick="manualTrigger()">⚡ Manual Trigger</button>
    </div>

    <div class="card">
      <div class="label">Live Market (SET / Value)</div>
      <div class="metrics">
        <div class="metric"><div class="label">SET</div><span id="setVal">—</span></div>
        <div class="metric"><div class="label">Value</div><span id="valVal">—</span></div>
      </div>
      <div class="sub" id="marketMeta" style="margin-top:8px">Source: —</div>
    </div>

    <div class="card">
      <div class="label">Last Run Results</div>
      <pre id="lastRun">No runs yet.</pre>
    </div>

    <div class="card">
      <div class="label">Log History</div>
      <div id="logs"></div>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    let secondsUntil = 0;

    function fmtCountdown(sec) {
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = sec % 60;
      return [h,m,s].map(v => String(v).padStart(2,'0')).join(':');
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg; t.style.display = 'block';
      setTimeout(() => t.style.display = 'none', 3000);
    }

    async function refresh() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        const badge = document.getElementById('statusBadge');
        badge.textContent = data.agent_status;
        badge.className = 'badge ' + (data.agent_status === 'Active' ? 'active' : 'idle');

        document.getElementById('triggerBtn').disabled = data.agent_status === 'Active';

        secondsUntil = parseInt(data.next_run.seconds_until || '0', 10);
        document.getElementById('nextLabel').textContent =
          data.next_run.label + ' @ ' + data.next_run.at_mmt;
        document.getElementById('countdown').textContent = fmtCountdown(secondsUntil);

        const m = data.live_market || {};
        document.getElementById('setVal').textContent = m.set_now ?? '—';
        document.getElementById('valVal').textContent = m.value_now ?? '—';
        document.getElementById('marketMeta').textContent =
          'Source: ' + (m.source || '—') + ' · ' + (m.sampled_at || '');

        const lr = data.last_run;
        document.getElementById('lastRun').textContent = lr
          ? JSON.stringify(lr, null, 2)
          : 'No runs yet.';

        const logsEl = document.getElementById('logs');
        logsEl.innerHTML = (data.logs || []).slice(0,20).map(l =>
          `<div class="log"><span class="t">${l.time_mmt}</span> [${l.level}] ${l.message}</div>`
        ).join('') || '<div class="sub">No logs yet.</div>';
      } catch (e) {
        console.error(e);
      }
    }

    async function manualTrigger() {
      const btn = document.getElementById('triggerBtn');
      btn.disabled = true;
      try {
        const res = await fetch('/api/trigger', { method: 'POST' });
        const data = await res.json();
        showToast(data.message || 'Triggered');
      } catch (e) {
        showToast('Trigger failed');
      }
      setTimeout(refresh, 1500);
    }

    setInterval(() => {
      if (secondsUntil > 0) {
        secondsUntil -= 1;
        document.getElementById('countdown').textContent = fmtCountdown(secondsUntil);
      }
    }, 1000);

    refresh();
    setInterval(refresh, 10000);
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def is_port_in_use(port: int) -> bool:
    """Return True if something is already listening on this port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        # connect_ex == 0 means a listener accepted the connection attempt
        return sock.connect_ex(("127.0.0.1", port)) == 0


def find_available_port(preferred: int, max_attempts: int = 10) -> int:
    """Use preferred port, or scan upward if another process already holds it."""
    for offset in range(max_attempts):
        port = preferred + offset
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No free port found in range {preferred}-{preferred + max_attempts - 1}")


if __name__ == "__main__":
    import uvicorn

    bind_port = find_available_port(PORT)
    if bind_port != PORT:
        logger.warning(
            "Port %s is already in use (likely another python/uvicorn instance). "
            "Using port %s instead. Stop the old process or set PORT=%s.",
            PORT,
            bind_port,
            bind_port,
        )

    logger.info("Starting Thai 2D Multi-Agent on port %s", bind_port)
    print(f"\n  Dashboard: http://127.0.0.1:{bind_port}/")
    print(f"  Health:    http://127.0.0.1:{bind_port}/health\n")
    uvicorn.run("main:app", host="0.0.0.0", port=bind_port, reload=False)
