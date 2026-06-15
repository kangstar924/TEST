"""app.py — Flask + SocketIO 서버.

프론트엔드(index.html)와 수어 인식 파이프라인을 연결하는 웹 서버.

SocketIO 이벤트 요약
--------------------
클라이언트 → 서버:
    push_gloss      {"gloss": "수원"}          인식기가 글로스 한 개를 전송
    confirmation    {"confirmed": true/false}   '맞아요' / '다시' 응답
    start_pipeline  {}                          파이프라인 전체 시작

서버 → 클라이언트:
    connected       {"status": "connected"}     연결 확인
    pipeline_done   {required_keys dict}        파이프라인 완료 결과
    sign_result     {mode, glosses, display, parsed}  각 인식 단계 결과
"""
from __future__ import annotations

import threading

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

from pipeline import SignPipeline

app = Flask(__name__, static_folder=".", static_url_path="")
app.config["SECRET_KEY"] = "sign-kiosk-secret-key"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

pipeline = SignPipeline(socketio=socketio)
_pipeline_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """현재 파이프라인 상태와 수집된 required_keys를 반환."""
    return jsonify({"status": "ok", "required_keys": pipeline.required_keys})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """파이프라인 상태를 초기화."""
    pipeline.reset()
    return jsonify({"status": "reset"})


# ---------------------------------------------------------------------------
# SocketIO event handlers
# ---------------------------------------------------------------------------


def _run_pipeline() -> None:
    """백그라운드 스레드에서 전체 파이프라인을 실행하고 완료 결과를 emit."""
    result = pipeline.run(socketio=socketio)
    socketio.emit("pipeline_done", result)


@socketio.on("connect")
def on_connect():
    emit("connected", {"status": "connected"})


@socketio.on("start_pipeline")
def on_start_pipeline(_data=None):
    """파이프라인 전체 흐름을 백그라운드 스레드로 시작."""
    global _pipeline_thread
    if _pipeline_thread and _pipeline_thread.is_alive():
        return  # 이미 실행 중
    pipeline.reset()
    _pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True)
    _pipeline_thread.start()
    emit("pipeline_started", {"status": "started"})


@socketio.on("push_gloss")
def on_push_gloss(data: dict):
    """수어 인식기(또는 테스트 클라이언트)로부터 글로스를 수신."""
    gloss = data.get("gloss", "")
    if gloss:
        pipeline.push_gloss(str(gloss))


@socketio.on("confirmation")
def on_confirmation(data: dict):
    """사용자의 '맞아요'(confirmed=true) / '다시'(confirmed=false) 응답 처리."""
    confirmed = bool(data.get("confirmed", False))
    pipeline.handle_confirmation(confirmed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
