"""pipeline.py — 수어 인식 파이프라인 (SignPipeline).

외부 수어 인식기(MediaPipe + 분류 모델 등)가 push_gloss()로 글로스를 공급하면,
각 recognize_* 단계가 수집→파싱→emit→사용자 확인 사이클을 반복해
required_keys 딕셔너리를 채워 나간다.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from core import (
    NUM_DIGITS,
    VALID_TERMINALS,
    glosses_to_display,
    parse_date_glosses,
    parse_num_glosses,
    parse_seat_glosses,
    parse_terminal_glosses,
    parse_time_glosses,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

COLLECT_DURATION: float = 3.0   # 한 번의 인식 수집에 허용하는 시간(초)
CONFIRM_TIMEOUT: float = 15.0   # 사용자 확인 응답 대기 최대 시간(초)

#: 파이프라인 완료 결과의 기본 구조
REQUIRED_KEYS_DEFAULTS: Dict[str, Any] = {
    "terminal": None,
    "date": None,
    "time": None,
    "adult": 0,
    "child": 0,
    "seat": None,
    "total_price": 0,
}


# ---------------------------------------------------------------------------
# SignPipeline
# ---------------------------------------------------------------------------


class SignPipeline:
    """수어 인식 파이프라인.

    인식 흐름(목적지 → 날짜 → 시간대 → 좌석 → 인원)을 단계별로 진행하며,
    각 단계의 결과를 socketio 이벤트로 프론트엔드에 전달하고
    사용자의 '맞아요' / '다시' 응답을 기다려 확정한다.

    사용 예::

        pipeline = SignPipeline(socketio=sio)
        result = pipeline.run()
        # result == {"terminal": "수원", "date": "2026-06-20", ...}
    """

    def __init__(self, socketio=None) -> None:
        self.socketio = socketio
        self.required_keys: Dict[str, Any] = dict(REQUIRED_KEYS_DEFAULTS)
        self._gloss_buffer: deque[str] = deque(maxlen=100)
        self._confirm_event = threading.Event()
        self._confirm_result: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public interface for the external sign recognizer
    # ------------------------------------------------------------------

    def push_gloss(self, gloss: str) -> None:
        """외부 인식기에서 한 프레임의 글로스를 버퍼에 추가."""
        self._gloss_buffer.append(gloss)

    def handle_confirmation(self, confirmed: bool) -> None:
        """프론트엔드에서 수신한 '맞아요'(True) / '다시'(False) 응답을 처리."""
        self._confirm_result = confirmed
        self._confirm_event.set()

    def reset(self) -> None:
        """파이프라인 상태(인식 결과·버퍼·확인 플래그)를 초기화."""
        self.required_keys = dict(REQUIRED_KEYS_DEFAULTS)
        self._gloss_buffer.clear()
        self._confirm_event.clear()
        self._confirm_result = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_glosses(self, duration: float = COLLECT_DURATION) -> List[str]:
        """버퍼를 초기화한 뒤 duration 초 동안 수집된 글로스 목록을 반환."""
        self._gloss_buffer.clear()
        time.sleep(duration)
        return list(self._gloss_buffer)

    def _wait_confirmation(self, timeout: float = CONFIRM_TIMEOUT) -> bool:
        """사용자 확인 이벤트를 최대 timeout 초 대기. 타임아웃 시 False 반환."""
        self._confirm_event.clear()
        self._confirm_result = None
        received = self._confirm_event.wait(timeout=timeout)
        return bool(received and self._confirm_result)

    def _collect_and_confirm(
        self,
        mode: str,
        parse_fn: Callable[[List[str]], Any],
        duration: float = COLLECT_DURATION,
    ) -> Any:
        """글로스 수집 → 파싱 → socketio emit → 사용자 확인의 사이클을 반복.

        사용자가 '맞아요'로 확인하면 parsed 결과를 반환하고,
        '다시' 또는 타임아웃이면 재수집을 시도한다.

        socketio 이벤트 형식::

            {
                "mode": "terminal" | "date" | "time" | "num" | "seat",
                "glosses": [...],          # 원시 글로스 배열
                "display": "어른 3 아이 2", # 화면 표시용 문자열
                "parsed": {...},            # 항상 dict (문자열은 {"terminal": ...}로 래핑)
            }
        """
        while True:
            glosses = self._collect_glosses(duration)
            parsed = parse_fn(glosses)
            display = glosses_to_display(glosses)

            if self.socketio:
                self.socketio.emit('sign_result', {
                    'mode': mode,
                    'glosses': glosses,
                    'display': display,
                    'parsed': parsed if isinstance(parsed, dict) else {"terminal": parsed},
                })

            if self._wait_confirmation():
                return parsed

    # ------------------------------------------------------------------
    # Recognition steps
    # ------------------------------------------------------------------

    def recognize_terminal(self, required_keys: dict, socketio=None) -> Optional[str]:
        """목적지 인식 단계.

        유효한 역/도시 이름 글로스를 찾아 required_keys["terminal"]에 저장한다.
        """
        if socketio:
            self.socketio = socketio

        terminal = self._collect_and_confirm(
            mode="terminal",
            parse_fn=lambda glosses: parse_terminal_glosses(glosses, VALID_TERMINALS),
        )
        required_keys["terminal"] = terminal
        return terminal

    def recognize_date(self, required_keys: dict, socketio=None) -> Optional[str]:
        """날짜 인식 단계.

        숫자 글로스를 조합해 'YYYY-MM-DD' 형식으로 required_keys["date"]에 저장한다.
        """
        if socketio:
            self.socketio = socketio

        date = self._collect_and_confirm(
            mode="date",
            parse_fn=parse_date_glosses,
        )
        required_keys["date"] = date
        return date

    def recognize_time(self, required_keys: dict, socketio=None) -> Optional[str]:
        """시간대 인식 단계.

        오전/오후 + 숫자 글로스를 조합해 'HH:MM' 형식으로 required_keys["time"]에 저장한다.
        """
        if socketio:
            self.socketio = socketio

        time_val = self._collect_and_confirm(
            mode="time",
            parse_fn=parse_time_glosses,
        )
        required_keys["time"] = time_val
        return time_val

    def recognize_num(self, required_keys: dict, socketio=None) -> Dict[str, int]:
        """인원 인식 단계.

        '어른/아이' 키워드와 숫자 글로스를 파싱해
        required_keys["adult"] 및 required_keys["child"]에 저장한다.
        반환값은 {"adult": n, "child": m} dict.
        """
        if socketio:
            self.socketio = socketio

        num_result = self._collect_and_confirm(
            mode="num",
            parse_fn=lambda glosses: parse_num_glosses(glosses, NUM_DIGITS),
        )
        if isinstance(num_result, dict):
            required_keys["adult"] = num_result.get("adult", 0)
            required_keys["child"] = num_result.get("child", 0)
        return num_result  # type: ignore[return-value]

    def recognize_seat(self, required_keys: dict, socketio=None) -> Optional[str]:
        """좌석 유형 인식 단계.

        좌석 키워드 글로스를 찾아 required_keys["seat"]에 저장한다.
        """
        if socketio:
            self.socketio = socketio

        seat = self._collect_and_confirm(
            mode="seat",
            parse_fn=parse_seat_glosses,
        )
        required_keys["seat"] = seat
        return seat

    # ------------------------------------------------------------------
    # Full pipeline runner
    # ------------------------------------------------------------------

    def run(self, socketio=None) -> Dict[str, Any]:
        """전체 인식 파이프라인을 순서대로 실행하고 최종 required_keys를 반환.

        순서: 목적지 → 날짜 → 시간대 → 좌석 → 인원

        Returns:
            terminal/date/time/adult/child/seat/total_price 키를 가진 dict.
        """
        if socketio:
            self.socketio = socketio
        self.reset()

        self.recognize_terminal(self.required_keys)
        self.recognize_date(self.required_keys)
        self.recognize_time(self.required_keys)
        self.recognize_seat(self.required_keys)
        self.recognize_num(self.required_keys)

        return dict(self.required_keys)
