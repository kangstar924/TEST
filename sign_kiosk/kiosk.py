"""수어 인식 발권 키오스크 상태 머신.

다이어그램 플로우:
    버튼 누름
      → [아바타] "목적지가 어디세요?"  → [수어 인식] 목적지(region 모델)
      → [터치] 맞아요 / 다시
      → [아바타] "날짜를 선택하세요"    → [터치] 날짜
      → [아바타] "시간대를 선택하세요"  → [터치] 시간
      → [아바타] "어른/아이 몇 명인가요?" → [수어 숫자] 인원(num_age 모델)
      → [터치] 맞아요 / 다시
      → [아바타] "좌석을 선택하세요"     → [터치] 좌석
      → 발권 완료
"""

from __future__ import annotations

import collections
import time
from typing import List, Optional

import cv2
import numpy as np

from .config import CONFIDENCE_THRESHOLD, KioskConfig, SEQ_LEN
from .core import (
    SignClassifier,
    extract_keypoints,
    get_hand_center,
    sample_from_buffer,
)
from .ui import (
    Button,
    ButtonRow,
    MouseState,
    Speaker,
    draw_avatar,
    draw_korean_centered,
    put_korean_text,
)

try:  # pragma: no cover - 환경 의존
    import mediapipe as mp
    _MP_OK = True
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _MP_OK = False


# ==========================================
# 상태 정의
# ==========================================
class State:
    WELCOME = "welcome"
    RECOGNIZE_DEST = "recognize_dest"
    CONFIRM_DEST = "confirm_dest"
    SELECT_DATE = "select_date"
    SELECT_TIME = "select_time"
    RECOGNIZE_COUNT = "recognize_count"
    CONFIRM_COUNT = "confirm_count"
    SELECT_SEAT = "select_seat"
    COMPLETE = "complete"


WINDOW = "Sign Language Ticket Kiosk"

# 터치 UI 선택지
DATE_OPTIONS = ["오늘", "내일", "모레", "이번 주말"]
TIME_OPTIONS = ["오전", "오후", "저녁", "심야"]
SEAT_OPTIONS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# 수어 인식 파라미터
STABLE_THRESHOLD = 0.012
STABLE_REQUIRED = 12
RAW_BUFFER_SIZE = 80
CAPTURE_COOLDOWN = 1.2     # 연속 캡처 방지(초)
STATE_WARMUP = 0.8         # 상태 진입 후 캡처 시작까지 대기(초)


class KioskApp:
    def __init__(self, config: Optional[KioskConfig] = None):
        self.cfg = config or KioskConfig()

        if not _MP_OK:
            raise RuntimeError("mediapipe가 설치되어 있지 않습니다. requirements.txt를 설치하세요.")

        # --- 두 모델 로드 ---
        rc = self.cfg.region_model()
        nc = self.cfg.numage_model()
        print(f"[모델] 목적지 모델 로드: {rc.weights_path}")
        self.region_clf = SignClassifier(rc.weights_path, rc.label_map_path)
        print(f"[모델] 숫자 모델 로드:   {nc.weights_path}")
        self.num_clf = SignClassifier(nc.weights_path, nc.label_map_path)
        print(f"  - 목적지 클래스({self.region_clf.num_classes}): "
              f"{', '.join(self.region_clf.labels)}")
        print(f"  - 숫자 클래스({self.num_clf.num_classes}):   "
              f"{', '.join(self.num_clf.labels)}")

        # --- MediaPipe ---
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # --- 상태 ---
        self.state = State.WELCOME
        self.state_enter_time = time.time()
        self.reservation = {
            "destination": None, "date": None, "time": None,
            "adults": None, "children": None, "seat": None,
        }

        # --- 수어 버퍼/안정성 ---
        self.frame_buffer = collections.deque(maxlen=RAW_BUFFER_SIZE)
        self.hand_center_history = collections.deque(maxlen=10)
        self.hand_stable_frames = 0
        self.last_capture_time = 0.0

        # --- 인식 후보 ---
        self.candidate_label = ""
        self.candidate_conf = 0.0
        self.candidate_top3: List = []

        # --- 인원(어른/아이) 수집 ---
        self.count_phase = "adult"      # adult → child
        self.pending_adult: Optional[str] = None
        self.pending_child: Optional[str] = None

        # --- UI ---
        self.mouse = MouseState()
        self.buttons = ButtonRow()
        self.speaker = Speaker(enabled=self.cfg.enable_tts)
        self.running = True

    # ------------------------------------------
    # 상태 전환
    # ------------------------------------------
    def go(self, new_state: str, keep_candidate: bool = False):
        self.state = new_state
        self.state_enter_time = time.time()
        self.frame_buffer.clear()
        self.hand_center_history.clear()
        self.hand_stable_frames = 0
        if not keep_candidate:
            self.candidate_label = ""
            self.candidate_conf = 0.0
            self.candidate_top3 = []
        self.speaker.reset()

    # ------------------------------------------
    # 프레임 처리 (랜드마크 + 손 안정성)
    # ------------------------------------------
    def _process(self, frame):
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1))
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

        self.frame_buffer.append(extract_keypoints(results))

        hand_detected = (results.left_hand_landmarks is not None
                         or results.right_hand_landmarks is not None)

        hc = get_hand_center(results)
        is_stable = False
        if hc is not None:
            self.hand_center_history.append(hc)
            if len(self.hand_center_history) >= 5:
                recent = np.array(list(self.hand_center_history)[-5:])
                is_stable = np.std(recent, axis=0).mean() < STABLE_THRESHOLD
                self.hand_stable_frames = self.hand_stable_frames + 1 if is_stable else 0
            else:
                self.hand_stable_frames = 0
        else:
            self.hand_stable_frames = 0
            self.hand_center_history.clear()

        return results, hand_detected, is_stable

    def _try_capture(self, clf: SignClassifier, force: bool = False
                     ) -> Optional[tuple]:
        """조건 충족 시(또는 force) 예측을 수행하고 결과 반환."""
        now = time.time()
        if not force:
            if now - self.state_enter_time < STATE_WARMUP:
                return None
            if now - self.last_capture_time < CAPTURE_COOLDOWN:
                return None
            if len(self.frame_buffer) < SEQ_LEN:
                return None
            if self.hand_stable_frames < STABLE_REQUIRED:
                return None
        if len(self.frame_buffer) < SEQ_LEN:
            return None

        seq = sample_from_buffer(self.frame_buffer, SEQ_LEN)
        label, conf, top3 = clf.predict(seq)
        self.last_capture_time = now
        return label, conf, top3

    # ------------------------------------------
    # 공통 UI 요소
    # ------------------------------------------
    def _draw_sign_hud(self, frame, hand_detected, is_stable, w):
        if hand_detected:
            sc = (0, 255, 0) if is_stable else (0, 200, 255)
            st = f"손 안정 ({self.hand_stable_frames}f)" if is_stable else "손 움직임 감지"
        else:
            sc = (0, 0, 255)
            st = "손이 보이지 않음"
        cv2.circle(frame, (20, 175), 8, sc, -1)
        frame = put_korean_text(frame, st, (36, 162), 22, sc)

        # 버퍼 바
        bx = w - 175
        br = len(self.frame_buffer) / RAW_BUFFER_SIZE
        cv2.rectangle(frame, (bx, 165), (bx + 150, 182), (80, 80, 80), -1)
        bc = (0, 255, 0) if len(self.frame_buffer) >= SEQ_LEN else (0, 200, 200)
        cv2.rectangle(frame, (bx, 165), (bx + int(150 * br), 182), bc, -1)
        cv2.putText(frame, f"Buffer {len(self.frame_buffer)}/{RAW_BUFFER_SIZE}",
                    (bx, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        frame = put_korean_text(frame, "손을 멈추면 자동 인식 · SPACE 수동 인식",
                                (bx - 120, 205), 18, (200, 200, 200))
        return frame

    def _draw_top3(self, frame, top3, w):
        if not top3:
            return frame
        tx = w - 300
        ty = 240
        frame = put_korean_text(frame, "Top-3", (tx, ty - 28), 22, (220, 220, 220))
        for i, (lbl, prob) in enumerate(top3):
            y = ty + 38 * i
            bl = int(prob * 160)
            clr = (0, 200, 0) if i == 0 else (90, 90, 90)
            cv2.rectangle(frame, (tx, y), (tx + bl, y + 28), clr, -1)
            frame = put_korean_text(frame, f"{lbl} {prob:.0%}", (tx + 6, y + 2),
                                    22, (255, 255, 255))
        return frame

    def _draw_progress(self, frame, w, h):
        steps = ["목적지", "날짜", "시간", "인원", "좌석"]
        done = {
            "목적지": self.reservation["destination"],
            "날짜": self.reservation["date"],
            "시간": self.reservation["time"],
            "인원": self.reservation["adults"],
            "좌석": self.reservation["seat"],
        }
        x = 20
        y = h - 36
        for s in steps:
            color = (0, 220, 0) if done[s] else (120, 120, 120)
            cv2.circle(frame, (x, y), 7, color, -1)
            frame = put_korean_text(frame, s, (x + 14, y - 14), 20, color)
            x += 110
        return frame

    def _make_option_buttons(self, options, w, h, cols=4):
        self.buttons.clear()
        bw, bh, gap = 220, 90, 24
        total_w = cols * bw + (cols - 1) * gap
        start_x = (w - total_w) // 2
        start_y = 230
        for i, opt in enumerate(options):
            row, col = divmod(i, cols)
            x = start_x + col * (bw + gap)
            y = start_y + row * (bh + gap)
            self.buttons.add(Button(opt, x, y, bw, bh, key=opt,
                                    color=(150, 110, 60), font_size=34))

    def _make_confirm_buttons(self, w, h):
        self.buttons.clear()
        bw, bh, gap = 240, 100, 60
        total_w = 2 * bw + gap
        start_x = (w - total_w) // 2
        y = h - 180
        self.buttons.add(Button("맞아요", start_x, y, bw, bh, key="yes",
                                color=(40, 160, 40), font_size=40))
        self.buttons.add(Button("다시", start_x + bw + gap, y, bw, bh, key="no",
                                color=(40, 40, 180), font_size=40))

    # ------------------------------------------
    # 메인 루프
    # ------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.mouse.on_mouse)

        print("\n=== 수어 인식 발권 키오스크 ===")
        print("  마우스 클릭 : 터치 버튼")
        print("  SPACE       : 수어 수동 인식")
        print("  R           : 처음으로")
        print("  Q / ESC     : 종료\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                results, hand_detected, is_stable = self._process(frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    self._reset_all()
                space = key == ord(' ')
                click = self.mouse.consume()

                frame = self._dispatch(frame, w, h, hand_detected, is_stable,
                                       space, click)

                cv2.imshow(WINDOW, frame)
        finally:
            cap.release()
            self.holistic.close()
            cv2.destroyAllWindows()
            print("종료.")

    def _reset_all(self):
        self.reservation = {k: None for k in self.reservation}
        self.count_phase = "adult"
        self.pending_adult = None
        self.pending_child = None
        self.go(State.WELCOME)

    # ------------------------------------------
    # 상태 디스패치
    # ------------------------------------------
    def _dispatch(self, frame, w, h, hand_detected, is_stable, space, click):
        s = self.state
        if s == State.WELCOME:
            return self._st_welcome(frame, w, h, click)
        if s == State.RECOGNIZE_DEST:
            return self._st_recognize(frame, w, h, hand_detected, is_stable, space,
                                      self.region_clf, "목적지가 어디세요?",
                                      State.CONFIRM_DEST)
        if s == State.CONFIRM_DEST:
            return self._st_confirm(frame, w, h, click, "목적지", self.candidate_label,
                                    on_yes=self._dest_confirmed,
                                    on_no=lambda: self.go(State.RECOGNIZE_DEST))
        if s == State.SELECT_DATE:
            return self._st_select(frame, w, h, click, "날짜를 선택하세요",
                                   DATE_OPTIONS, "date", State.SELECT_TIME)
        if s == State.SELECT_TIME:
            return self._st_select(frame, w, h, click, "시간대를 선택하세요",
                                   TIME_OPTIONS, "time", State.RECOGNIZE_COUNT)
        if s == State.RECOGNIZE_COUNT:
            return self._st_recognize_count(frame, w, h, hand_detected, is_stable, space)
        if s == State.CONFIRM_COUNT:
            val = f"어른 {self.pending_adult} · 아이 {self.pending_child}"
            return self._st_confirm(frame, w, h, click, "인원", val,
                                    on_yes=self._count_confirmed,
                                    on_no=self._count_retry)
        if s == State.SELECT_SEAT:
            return self._st_select(frame, w, h, click, "좌석을 선택하세요",
                                   SEAT_OPTIONS, "seat", State.COMPLETE, cols=3)
        if s == State.COMPLETE:
            return self._st_complete(frame, w, h, click)
        return frame

    # ------------------------------------------
    # 각 상태 핸들러
    # ------------------------------------------
    def _st_welcome(self, frame, w, h, click):
        frame = draw_avatar(frame, "수어로 표를 예매해요. 시작할까요?", w)
        self.buttons.clear()
        bw, bh = 360, 120
        bx, by = (w - bw) // 2, h // 2
        self.buttons.add(Button("시작하기", bx, by, bw, bh, key="start",
                                color=(40, 160, 120), font_size=46))
        frame = self.buttons.draw(frame)
        self.speaker.say("수어로 표를 예매해요. 시작하려면 버튼을 눌러주세요.")
        if click and self.buttons.hit(*click) == "start":
            self.go(State.RECOGNIZE_DEST)
        return frame

    def _st_recognize(self, frame, w, h, hand_detected, is_stable, space,
                      clf, question, next_state):
        frame = draw_avatar(frame, question, w)
        frame = self._draw_sign_hud(frame, hand_detected, is_stable, w)

        cap = self._try_capture(clf, force=space and hand_detected)
        if cap is not None:
            label, conf, top3 = cap
            self.candidate_top3 = top3
            print(f"[인식] {question} → {label} ({conf:.0%}) {top3}")
            if conf >= CONFIDENCE_THRESHOLD:
                self.candidate_label = label
                self.candidate_conf = conf
                self.candidate_top3 = top3
                self.go(next_state, keep_candidate=True)
                return frame

        frame = self._draw_top3(frame, self.candidate_top3, w)
        frame = self._draw_progress(frame, w, h)
        return frame

    def _st_recognize_count(self, frame, w, h, hand_detected, is_stable, space):
        phase_txt = "어른 인원" if self.count_phase == "adult" else "아이 인원"
        q = f"{phase_txt}을(를) 수어 숫자로 보여주세요"
        frame = draw_avatar(frame, q, w)
        frame = self._draw_sign_hud(frame, hand_detected, is_stable, w)

        if self.pending_adult is not None:
            frame = put_korean_text(frame, f"어른: {self.pending_adult}",
                                    (40, 240), 30, (0, 220, 0))

        cap = self._try_capture(self.num_clf, force=space and hand_detected)
        if cap is not None:
            label, conf, top3 = cap
            self.candidate_top3 = top3
            print(f"[인식] {phase_txt} → {label} ({conf:.0%}) {top3}")
            if conf >= CONFIDENCE_THRESHOLD:
                if self.count_phase == "adult":
                    self.pending_adult = label
                    self.count_phase = "child"
                    self.state_enter_time = time.time()
                    self.last_capture_time = time.time()
                    self.frame_buffer.clear()
                    self.hand_stable_frames = 0
                    self.candidate_top3 = []
                else:
                    self.pending_child = label
                    self.go(State.CONFIRM_COUNT)
                return frame

        frame = self._draw_top3(frame, self.candidate_top3, w)
        frame = self._draw_progress(frame, w, h)
        return frame

    def _st_confirm(self, frame, w, h, click, title, value, on_yes, on_no):
        frame = draw_avatar(frame, f"{value}, 맞나요?", w)
        frame = draw_korean_centered(frame, title, w / 2, 220, 30, (200, 200, 200))
        frame = draw_korean_centered(frame, value, w / 2, 280, 64, (0, 255, 0))

        self._make_confirm_buttons(w, h)
        frame = self.buttons.draw(frame)
        frame = self._draw_progress(frame, w, h)

        if click:
            hit = self.buttons.hit(*click)
            if hit == "yes":
                on_yes()
            elif hit == "no":
                on_no()
        return frame

    def _st_select(self, frame, w, h, click, question, options, field, next_state,
                   cols=4):
        frame = draw_avatar(frame, question, w)
        self._make_option_buttons(options, w, h, cols=cols)
        frame = self.buttons.draw(frame)
        frame = self._draw_progress(frame, w, h)
        self.speaker.say(question)
        if click:
            hit = self.buttons.hit(*click)
            if hit is not None:
                self.reservation[field] = hit
                print(f"[선택] {field} = {hit}")
                self.go(next_state)
        return frame

    def _st_complete(self, frame, w, h, click):
        frame = draw_avatar(frame, "예매가 완료되었어요! 감사합니다.", w)
        r = self.reservation
        lines = [
            f"목적지 : {r['destination']}",
            f"날짜    : {r['date']}",
            f"시간    : {r['time']}",
            f"인원    : 어른 {r['adults']} · 아이 {r['children']}",
            f"좌석    : {r['seat']}",
        ]
        y = 230
        for ln in lines:
            frame = put_korean_text(frame, ln, (w // 2 - 240, y), 36, (255, 255, 255))
            y += 56

        self.buttons.clear()
        bw, bh = 320, 90
        self.buttons.add(Button("처음으로", (w - bw) // 2, y + 20, bw, bh,
                                key="restart", color=(40, 120, 160), font_size=38))
        frame = self.buttons.draw(frame)
        self.speaker.say("예매가 완료되었습니다. 감사합니다.")
        if click and self.buttons.hit(*click) == "restart":
            self._reset_all()
        return frame

    # ------------------------------------------
    # 확정 콜백
    # ------------------------------------------
    def _dest_confirmed(self):
        self.reservation["destination"] = self.candidate_label
        self.go(State.SELECT_DATE)

    def _count_confirmed(self):
        self.reservation["adults"] = self.pending_adult
        self.reservation["children"] = self.pending_child
        self.go(State.SELECT_SEAT)

    def _count_retry(self):
        self.count_phase = "adult"
        self.pending_adult = None
        self.pending_child = None
        self.go(State.RECOGNIZE_COUNT)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="수어 인식 발권 키오스크")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--region-dir", default=None, help="목적지 모델 폴더")
    parser.add_argument("--numage-dir", default=None, help="숫자/나이 모델 폴더")
    parser.add_argument("--no-tts", action="store_true", help="음성 안내 끄기")
    args = parser.parse_args()

    cfg = KioskConfig()
    cfg.camera_index = args.camera
    if args.region_dir:
        cfg.region_dir = args.region_dir
    if args.numage_dir:
        cfg.numage_dir = args.numage_dir
    if args.no_tts:
        cfg.enable_tts = False

    app = KioskApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
