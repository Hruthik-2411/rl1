"""Tiny shop API used as the system under test.

It talks to Postgres and Redis, writes one JSON log line per event, and
has two admin endpoints that let the fault injector simulate a memory
leak. Nothing clever on purpose: it only needs to behave like a real
service when things go wrong.
"""
import json
import os
import threading
import time

import psycopg2
import redis
from flask import Flask, jsonify, request

DB_HOST = os.environ.get("DB_HOST", "db")
REDIS_HOST = os.environ.get("REDIS_HOST", "cache")
SLOW_MS = 500
MEM_WARN_MB = 300

app = Flask(__name__)
cache = redis.Redis(host=REDIS_HOST, port=6379, socket_timeout=1, socket_connect_timeout=1)

_leak = []
_leak_lock = threading.Lock()
_leak_rate_mb = 0


def log(level, event, **kw):
    print(json.dumps({"ts": time.time(), "level": level, "event": event, **kw}), flush=True)


def db_conn():
    return psycopg2.connect(host=DB_HOST, dbname="shop", user="postgres",
                            password="fd", connect_timeout=2)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def init_db():
    for attempt in range(30):
        try:
            with db_conn() as c, c.cursor() as cur:
                cur.execute("CREATE TABLE IF NOT EXISTS items (id serial PRIMARY KEY, name text, price int)")
                cur.execute("SELECT count(*) FROM items")
                if cur.fetchone()[0] == 0:
                    for i in range(50):
                        cur.execute("INSERT INTO items (name, price) VALUES (%s, %s)", (f"item-{i}", 10 + i))
            log("info", "db_ready")
            return
        except Exception as e:  # db still starting
            log("warn", "db_wait", attempt=attempt, error=str(e)[:80])
            time.sleep(2)


@app.before_request
def _start_timer():
    request._t0 = time.time()


@app.after_request
def _log_request(resp):
    ms = (time.time() - request._t0) * 1000
    if ms > SLOW_MS:
        log("warn", "slow_request", path=request.path, ms=round(ms))
    return resp


@app.get("/items")
def items():
    item_id = int(request.args.get("id", 1 + int(time.time() * 1000) % 50))
    key = f"item:{item_id}"
    try:
        hit = cache.get(key)
        if hit:
            return jsonify(json.loads(hit))
    except redis.RedisError as e:
        log("error", "cache_error", error=str(e)[:80])

    try:
        with db_conn() as c, c.cursor() as cur:
            cur.execute("SELECT id, name, price FROM items WHERE id = %s", (item_id,))
            row = cur.fetchone()
    except Exception as e:
        log("error", "db_error", error=str(e)[:120])
        return jsonify(error="database unavailable"), 503

    data = {"id": row[0], "name": row[1], "price": row[2]} if row else {}
    try:
        cache.setex(key, 5, json.dumps(data))
    except redis.RedisError:
        pass
    return jsonify(data)


@app.get("/health")
def health():
    status = {"db": False, "cache": False, "rss_mb": round(rss_mb())}
    try:
        with db_conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        status["db"] = True
    except Exception:
        pass
    try:
        status["cache"] = bool(cache.ping())
    except redis.RedisError:
        pass
    ok = status["db"] and status["cache"]
    status["status"] = "ok" if ok else "degraded"
    return jsonify(status), 200 if ok else 503


# --- admin endpoints used by the fault injector --------------------------
def _leaker():
    while True:
        rate = _leak_rate_mb
        if rate:
            with _leak_lock:
                _leak.append(bytearray(rate * 1024 * 1024))
        time.sleep(1)


def _mem_watch():
    while True:
        mb = rss_mb()
        if mb > MEM_WARN_MB:
            log("warn", "memory_high", rss_mb=round(mb))
        time.sleep(5)


@app.post("/admin/leak")
def start_leak():
    global _leak_rate_mb
    _leak_rate_mb = int(request.args.get("mb", 8))
    return jsonify(leaking_mb_per_s=_leak_rate_mb)


@app.post("/admin/free")
def free_leak():
    global _leak_rate_mb
    _leak_rate_mb = 0
    with _leak_lock:
        _leak.clear()
    return jsonify(freed=True)


if __name__ == "__main__":
    init_db()
    threading.Thread(target=_leaker, daemon=True).start()
    threading.Thread(target=_mem_watch, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
