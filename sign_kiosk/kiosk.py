"""수어 인식 발권 키오스크 (단일 윈도우 데모).

다이어그램 플로우를 하나의 OpenCV 윈도우에서 진행한다.
수어 인식 단계는 pipeline 과 동일하게 core.GlossCollector(글로스 수집)를 쓴다.

    버튼 누름
      → [아바타] "목적지?"      → [수어] 목적지 글로스 수집(region) → 파싱
      → [터치] 맞아요 / 다시
      → [아바타] "날짜?"        → [터치] 날짜
      → [아바타] "시간대?"      → [터치] 시간
      → [아바타] "어른/아이 몇명?" → [수어 숫자] 인원 글로스 수집(num_age) → 파싱
      → [터치] 맞아요 / 다시
      → [아바타] "좌석?"        → [터치] 좌석
      → 발권 완료

웹 UI(socketio) 기반 흐름이 필요하면 pipeline.SignPipeline 을 사용한다.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from .config import (
    KioskConfig,
    LOW_CONF_MAX,
    RAW_BUFFER_SIZE,
    SEQ_LEN,
    STABLE_THRESHOLD_NUM,
    STABLE_THRESHOLD_TERMINAL,
    TERMINAL_NOISE,
    TERMINAL_VALID,
)
from .core import (
    GlossCollector,
    SignClassifier,
    parse_num_glosses,
    parse_terminal_glosses,
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

DATE_OPTIONS = ["오늘", "내일", "모레", "이번 주말"]
TIME_OPTIONS = ["오전", "오후", "저녁", "심야"]
SEAT_OPTIONS = ["A1", "A2", "B1", "B2", "C1", "C2"]

RECOG_MSG_SEC = 2.5      # 인식 실패 안내문 표시 시간


class KioskApp:
    def __init__(self, config: Optional[KioskConfig] = None):
        self.cfg = config or KioskConfig()
        if not _MP_OK:
            raise RuntimeError("mediapipe가 설치되어 있지 않습니다. requirements.txt를 설치하세요.")

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

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.state = State.WELCOME
        self.reservation = {
            "destination": None, "date": None, "time": None,
            "adults": None, "children": None, "seat": None,
        }

        # 현재 진행 중인 글로스 수집기
        self.collector: Optional[GlossCollector] = None
        self.recog_msg = ""
        self.recog_msg_time = 0.0

        # 확인 단계 표시용
        self.candidate_label = ""
        self.pending_adult: Optional[int] = None
        self.pending_child: Optional[int] = None

        self.mouse = MouseState()
        self.buttons = ButtonRow()
        self.speaker = Speaker(enabled=self.cfg.enable_tts)
        self.running = True

    # ------------------------------------------
    # 상태 전환
    # ------------------------------------------
    def go(self, new_state: str):
        self.state = new_state
        self.collector = None
        self.speaker.reset()

    def _set_recog_msg(self, msg: str):
        self.recog_msg = msg
        self.recog_msg_time = time.time()

    def _active_recog_msg(self) -> Optional[str]:
        if self.recog_msg and time.time() - self.recog_msg_time < RECOG_MSG_SEC:
            return self.recog_msg
        return None

    # ------------------------------------------
    # 프레임 처리 (랜드마크 그리기 + results 반환)
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
        return results

    # ------------------------------------------
    # 공통 UI
    # ------------------------------------------
    def _draw_sign_hud(self, frame, col: GlossCollector, w):
        if col.hand_detected:
            sc = (0, 255, 0) if col.is_stable else (0, 200, 255)
            st = (f"손 안정 ({col.hand_stable_frames}f)"
                  if col.is_stable else "손 움직임 감지")
        else:
            sc = (0, 0, 255)
            st = "손이 보이지 않음"
        cv2.circle(frame, (20, 175), 8, sc, -1)
        frame = put_korean_text(frame, st, (36, 162), 22, sc)

        bx = w - 175
        n = len(col.frame_buffer)
        cv2.rectangle(frame, (bx, 165), (bx + 150, 182), (80, 80, 80), -1)
        bc = (0, 255, 0) if n >= SEQ_LEN else (0, 200, 200)
        cv2.rectangle(frame, (bx, 165), (bx + int(150 * n / RAW_BUFFER_SIZE), 182), bc, -1)
        cv2.putText(frame, f"Buffer {n}/{RAW_BUFFER_SIZE}", (bx, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        gap = col.gap_remaining()
        if gap is not None:
            gc = (0, 220, 0) if gap > 1.0 else (0, 100, 255)
            cv2.putText(frame, f"다음 입력까지 {gap:.1f}s", (bx - 60, 222),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, gc, 1)
        if col.low_conf_count > 0:
            cv2.putText(frame, f"Low conf {col.low_conf_count}/{LOW_CONF_MAX}",
                        (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        return frame

    def _draw_gloss(self, frame, col: GlossCollector, w, h):
        gloss_str = " → ".join(col.glosses) if col.glosses else "수어를 보여주세요..."
        frame = put_korean_text(frame, gloss_str, (20, h - 90), 34, (0, 255, 200))
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
        x, y = 20, h - 36
        for s in steps:
            color = (0, 220, 0) if done[s] not in (None,) else (120, 120, 120)
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
        print("  R           : 처음으로 / Q·ESC : 종료")
        print("  수어 인식: 손을 멈추면 글로스가 하나씩 쌓이고, "
              f"공백이 생기면 자동 종료됩니다.\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                results = self._process(frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    self._reset_all()
                click = self.mouse.consume()

                frame = self._dispatch(frame, w, h, results, click)
                cv2.imshow(WINDOW, frame)
        finally:
            cap.release()
            self.holistic.close()
            cv2.destroyAllWindows()
            print("종료.")

    def _reset_all(self):
        self.reservation = {k: None for k in self.reservation}
        self.pending_adult = None
        self.pending_child = None
        self.candidate_label = ""
        self.recog_msg = ""
        self.go(State.WELCOME)

    # ------------------------------------------
    # 상태 디스패치
    # ------------------------------------------
    def _dispatch(self, frame, w, h, results, click):
        s = self.state
        if s == State.WELCOME:
            return self._st_welcome(frame, w, h, click)
        if s == State.RECOGNIZE_DEST:
            return self._st_recognize_dest(frame, w, h, results)
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
            return self._st_recognize_count(frame, w, h, results)
        if s == State.CONFIRM_COUNT:
            val = f"어른 {self.pending_adult}명 · 아이 {self.pending_child}명"
            return self._st_confirm(frame, w, h, click, "인원", val,
                                    on_yes=self._count_confirmed,
                                    on_no=lambda: self.go(State.RECOGNIZE_COUNT))
        if s == State.SELECT_SEAT:
            return self._st_select(frame, w, h, click, "좌석을 선택하세요",
                                   SEAT_OPTIONS, "seat", State.COMPLETE, cols=3)
        if s == State.COMPLETE:
            return self._st_complete(frame, w, h, click)
        return frame

    # ------------------------------------------
    # 상태 핸들러
    # ------------------------------------------
    def _st_welcome(self, frame, w, h, click):
        frame = draw_avatar(frame, "수어로 표를 예매해요. 시작할까요?", w)
        self.buttons.clear()
        bw, bh = 360, 120
        self.buttons.add(Button("시작하기", (w - bw) // 2, h // 2, bw, bh,
                                key="start", color=(40, 160, 120), font_size=46))
        frame = self.buttons.draw(frame)
        self.speaker.say("수어로 표를 예매해요. 시작하려면 버튼을 눌러주세요.")
        if click and self.buttons.hit(*click) == "start":
            self.go(State.RECOGNIZE_DEST)
        return frame

    def _st_recognize_dest(self, frame, w, h, results):
        if self.collector is None:
            self.collector = GlossCollector(
                self.region_clf, "terminal", STABLE_THRESHOLD_TERMINAL,
                valid_set=TERMINAL_VALID, noise_set=TERMINAL_NOISE)

        col = self.collector
        added = col.update(results)
        if added:
            print(f"  [글로스] {col.glosses}")

        q = self._active_recog_msg() or "목적지가 어디세요? 수어로 보여주세요"
        frame = draw_avatar(frame, q, w)
        frame = self._draw_sign_hud(frame, col, w)
        frame = self._draw_gloss(frame, col, w, h)
        frame = self._draw_progress(frame, w, h)

        if col.finished:
            if col.need_reinput or not col.glosses:
                self._set_recog_msg("인식이 잘 안돼요. 다시 보여주세요.")
                self.collector = None
            else:
                parsed = parse_terminal_glosses(col.glosses, TERMINAL_VALID)
                if parsed is None:
                    self._set_recog_msg("목적지를 인식하지 못했어요. 다시요.")
                    self.collector = None
                else:
                    self.candidate_label = parsed
                    self.go(State.CONFIRM_DEST)
        return frame

    def _st_recognize_count(self, frame, w, h, results):
        if self.collector is None:
            self.collector = GlossCollector(
                self.num_clf, "num", STABLE_THRESHOLD_NUM)

        col = self.collector
        added = col.update(results)
        if added:
            print(f"  [글로스] {col.glosses}")

        q = self._active_recog_msg() or "어른/아이 몇 명? (어른→숫자→아이→숫자)"
        frame = draw_avatar(frame, q, w)
        frame = self._draw_sign_hud(frame, col, w)
        frame = self._draw_gloss(frame, col, w, h)
        frame = self._draw_progress(frame, w, h)

        if col.finished:
            if col.need_reinput or not col.glosses:
                self._set_recog_msg("인식이 잘 안돼요. 다시 보여주세요.")
                self.collector = None
            else:
                parsed = parse_num_glosses(col.glosses)
                self.pending_adult = parsed["adult"]
                self.pending_child = parsed["child"]
                self.go(State.CONFIRM_COUNT)
        return frame

    def _st_confirm(self, frame, w, h, click, title, value, on_yes, on_no):
        frame = draw_avatar(frame, f"{value}, 맞나요?", w)
        frame = draw_korean_centered(frame, title, w / 2, 220, 30, (200, 200, 200))
        frame = draw_korean_centered(frame, str(value), w / 2, 280, 60, (0, 255, 0))
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
            f"인원    : 어른 {r['adults']}명 · 아이 {r['children']}명",
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
