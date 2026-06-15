"""UI 헬퍼: 한글 렌더링, 아바타 말풍선, 터치 버튼, TTS.

OpenCV는 한글을 직접 못 그리므로 PIL로 그려서 합성한다.
'터치 UI'는 데스크톱에서 마우스 클릭으로 동작한다(키오스크 터치 대체).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FONT_CANDIDATES

# ------------------------------------------
# 폰트 캐시 (크기별로 1회만 로드)
# ------------------------------------------
_FONT_CACHE: dict = {}
_FONT_PATH: Optional[str] = None


def _resolve_font_path() -> Optional[str]:
    global _FONT_PATH
    if _FONT_PATH is not None:
        return _FONT_PATH or None
    import os

    for cand in FONT_CANDIDATES:
        if cand and os.path.exists(cand):
            _FONT_PATH = cand
            return cand
    _FONT_PATH = ""  # 탐색 완료, 못 찾음
    return None


def _get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    path = _resolve_font_path()
    try:
        font = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def put_korean_text(img, text, pos, font_size=30, color=(0, 255, 0)):
    """color는 BGR(OpenCV 관례)로 받는다."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _get_font(font_size)
    color_rgb = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def text_size(text: str, font_size: int) -> Tuple[int, int]:
    font = _get_font(font_size)
    try:
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    except Exception:
        return len(text) * font_size // 2, font_size


def draw_korean_centered(img, text, center_x, y, font_size=30, color=(255, 255, 255)):
    tw, _ = text_size(text, font_size)
    return put_korean_text(img, text, (int(center_x - tw / 2), int(y)), font_size, color)


# ------------------------------------------
# 터치 버튼
# ------------------------------------------
@dataclass
class Button:
    label: str
    x: int
    y: int
    w: int
    h: int
    key: str                       # 논리적 식별자 (콜백 매칭용)
    color: Tuple[int, int, int] = (90, 90, 90)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    font_size: int = 28

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, frame):
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      self.color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      (255, 255, 255), 2)
        tw, th = text_size(self.label, self.font_size)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h - th) // 2 - 2
        return put_korean_text(frame, self.label, (tx, ty), self.font_size,
                               self.text_color)


class ButtonRow:
    """현재 프레임에 그릴 버튼 묶음 + 클릭 히트테스트."""

    def __init__(self):
        self.buttons: List[Button] = []

    def clear(self):
        self.buttons = []

    def add(self, button: Button):
        self.buttons.append(button)

    def draw(self, frame):
        for b in self.buttons:
            frame = b.draw(frame)
        return frame

    def hit(self, px: int, py: int) -> Optional[str]:
        for b in self.buttons:
            if b.contains(px, py):
                return b.key
        return None


class MouseState:
    """OpenCV 마우스 콜백으로부터 클릭 좌표를 한 번만 소비한다."""

    def __init__(self):
        self._click: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self._lock:
                self._click = (x, y)

    def consume(self) -> Optional[Tuple[int, int]]:
        with self._lock:
            c = self._click
            self._click = None
        return c


# ------------------------------------------
# 아바타 말풍선
# ------------------------------------------
def draw_avatar(frame, question: str, w: int):
    """상단 중앙에 아바타 얼굴 + 질문 말풍선을 그린다."""
    # 반투명 배경 띠
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (40, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # 아바타 얼굴 (간단한 원형 캐릭터)
    cx, cy, r = 75, 75, 45
    cv2.circle(frame, (cx, cy), r, (210, 190, 170), -1)
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
    cv2.circle(frame, (cx - 16, cy - 8), 6, (60, 50, 50), -1)   # 왼눈
    cv2.circle(frame, (cx + 16, cy - 8), 6, (60, 50, 50), -1)   # 오른눈
    cv2.ellipse(frame, (cx, cy + 12), (16, 9), 0, 0, 180, (60, 50, 50), 2)  # 입

    # 말풍선
    bx, by, bw, bh = 145, 30, w - 175, 90
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 252, 245), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
    # 꼬리
    pts = np.array([[bx, by + 40], [bx - 18, by + 55], [bx, by + 70]], np.int32)
    cv2.fillPoly(frame, [pts], (255, 252, 245))

    frame = put_korean_text(frame, question, (bx + 20, by + 26), 34, (60, 50, 50))
    return frame


# ------------------------------------------
# TTS (선택적, 실패해도 무시)
# ------------------------------------------
class Speaker:
    def __init__(self, enabled: bool = True):
        self.engine = None
        self._last = None
        if not enabled:
            return
        try:  # pragma: no cover - 환경 의존
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 170)
        except Exception:
            self.engine = None

    def say(self, text: str):
        """같은 문장 반복 방지 + 비차단 재생."""
        if self.engine is None or text == self._last:
            return
        self._last = text

        def _run():
            try:  # pragma: no cover
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

    def reset(self):
        self._last = None
