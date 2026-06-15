"""수어 인식 파이프라인 — 글로스 수집 방식.

목적지(terminal) 모델 + 인원(adult/child) 모델을 순서대로 돌려서
글로스 배열을 수집하고 규칙으로 파싱한다.

- 콘솔 단독 실행: 결과를 보여주고 Enter(맞아요)/r(다시)로 확인
- 웹 연동: socketio 객체를 넘기면 avatar_message / sign_result 이벤트를 emit 하고
  on_touch_confirm() / on_touch_retry() 로 확인을 받는다.

핵심 수집 로직은 core.GlossCollector 가 담당한다(kiosk와 공용).
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

import cv2

from .config import (
    KioskConfig,
    NUM_DIGITS,
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
from .ui import put_korean_text

try:  # pragma: no cover - 환경 의존
    import mediapipe as mp
    _MP_OK = True
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _MP_OK = False

WINDOW = "Sign Recognition"


# ==========================================
# 카메라 루프 — 글로스 배열 수집
# ==========================================
def collect_glosses(classifier: SignClassifier, mode: str, stable_threshold: float,
                    on_glosses: Callable[[Optional[List[str]]], None],
                    stop_event: threading.Event,
                    valid_set=None, noise_set=None,
                    camera_index: int = 0):
    """한 번의 수집 세션. 종료 시 on_glosses(glosses) 호출.

    - need_reinput(낮은 conf 초과) 인 경우 on_glosses(None) 으로 재입력 요청.
    - 사용자가 q 로 중단하면 stop_event 를 세팅하고 콜백을 호출하지 않는다.
    """
    if not _MP_OK:
        raise RuntimeError("mediapipe가 설치되어 있지 않습니다.")

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    collector = GlossCollector(
        classifier, mode, stable_threshold,
        valid_set=valid_set, noise_set=noise_set,
    )

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            added = collector.update(results)
            if added:
                print(f"  [글로스 추가] {collector.glosses}")
            if collector.finished:
                if collector.need_reinput:
                    print("  [재입력 요청] confidence가 계속 낮습니다.")
                elif collector.terminal_confirmed:
                    print(f"  [터미널 확정] {collector.glosses[-1]} → 즉시 종료")
                else:
                    print(f"  [gap 종료] 수집 종료: {collector.glosses}")

            frame = _draw_ui(frame, results, collector, mode,
                             mp_holistic, mp_drawing)
            cv2.imshow(WINDOW, frame)

            if collector.finished:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()

    if not stop_event.is_set():
        on_glosses(None if collector.need_reinput else collector.glosses)


def _draw_ui(frame, results, collector: GlossCollector, mode,
             mp_holistic, mp_drawing):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    mc = (0, 255, 255) if mode == "terminal" else (255, 200, 0)
    cv2.putText(frame, f"Mode: {mode.upper()}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc, 2)

    if collector.hand_detected:
        sc = (0, 255, 0) if collector.is_stable else (0, 200, 255)
        st = (f"Hand STABLE ({collector.hand_stable_frames}f)"
              if collector.is_stable else "Hand MOVING")
    else:
        sc, st = (0, 0, 255), "No Hand"
    cv2.circle(frame, (15, 55), 8, sc, -1)
    cv2.putText(frame, st, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1)

    bx = w - 165
    n = len(collector.frame_buffer)
    cv2.rectangle(frame, (bx, 10), (bx + 150, 25), (80, 80, 80), -1)
    bc = (0, 255, 0) if n >= SEQ_LEN else (0, 200, 200)
    cv2.rectangle(frame, (bx, 10), (bx + int(150 * n / RAW_BUFFER_SIZE), 25), bc, -1)
    cv2.putText(frame, f"Buffer {n}/{RAW_BUFFER_SIZE}", (bx, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    gap_remain = collector.gap_remaining()
    if gap_remain is not None:
        gap_color = (0, 255, 0) if gap_remain > 1.0 else (0, 100, 255)
        cv2.putText(frame, f"next: {gap_remain:.1f}s", (bx, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, gap_color, 1)

    gloss_str = " → ".join(collector.glosses) if collector.glosses else "수어를 보여주세요..."
    frame = put_korean_text(frame, gloss_str, (15, h - 80), 32, (0, 255, 200))

    if collector.low_conf_count > 0:
        from .config import LOW_CONF_MAX
        cv2.putText(frame, f"Low conf {collector.low_conf_count}/{LOW_CONF_MAX}",
                    (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
    return frame


# ==========================================
# SignPipeline
# ==========================================
class SignPipeline:
    def __init__(self, config: Optional[KioskConfig] = None):
        self.cfg = config or KioskConfig()
        print("수어 모델 로드 중...")
        rc = self.cfg.region_model()
        nc = self.cfg.numage_model()
        self.terminal_clf = SignClassifier(rc.weights_path, rc.label_map_path)
        self.num_clf = SignClassifier(nc.weights_path, nc.label_map_path)
        print(f"  목적지 모델: {self.terminal_clf.num_classes}개 클래스")
        print(f"  숫자 모델:   {self.num_clf.num_classes}개 클래스")

        self._confirm_event = threading.Event()
        self._confirm_result: Optional[str] = None
        self._stop_event = threading.Event()
        self._glosses: Optional[List[str]] = None
        self._glosses_event = threading.Event()
        print("수어 파이프라인 준비 완료.")

    # --- 웹(socketio) 터치 콜백 ---
    def on_touch_confirm(self):
        self._confirm_result = "confirm"
        self._confirm_event.set()

    def on_touch_retry(self):
        self._confirm_result = "retry"
        self._confirm_event.set()

    def _avatar(self, msg: str, socketio):
        if socketio:
            socketio.emit('avatar_message', {'message': msg})
        else:
            print(f"\n[아바타] {msg}")

    def _collect_and_confirm(self, classifier, stable_threshold, mode,
                             valid_set=None, noise_set=None, socketio=None):
        while True:
            self._glosses = None
            self._glosses_event.clear()
            self._stop_event.clear()

            def on_glosses(glosses):
                self._glosses = glosses
                self._glosses_event.set()

            t = threading.Thread(
                target=collect_glosses,
                args=(classifier, mode, stable_threshold, on_glosses,
                      self._stop_event),
                kwargs=dict(valid_set=valid_set, noise_set=noise_set,
                            camera_index=self.cfg.camera_index),
                daemon=True,
            )
            t.start()
            self._glosses_event.wait()
            t.join(timeout=1.0)

            glosses = self._glosses

            # 낮은 confidence → 재입력
            if glosses is None:
                self._avatar("인식이 잘 되지 않습니다. 다시 시도해주세요.", socketio)
                continue

            if not glosses:
                self._avatar("인식된 내용이 없습니다. 다시 시도해주세요.", socketio)
                continue

            if mode == "terminal":
                parsed = parse_terminal_glosses(glosses, TERMINAL_VALID)
                if parsed is None:
                    self._avatar("목적지를 인식하지 못했습니다. 다시 보여주세요.", socketio)
                    continue
                display = parsed
            else:
                parsed = parse_num_glosses(glosses, NUM_DIGITS)
                display = f"어른 {parsed['adult']}명, 아이 {parsed['child']}명"

            # 확인
            if socketio is None:
                print(f"\n[인식 결과] {display}")
                print("  Enter = 맞아요 / r = 다시")
                ans = input(">> ").strip().lower()
                if ans == "r":
                    continue
                return parsed
            else:
                self._confirm_event.clear()
                self._confirm_result = None
                socketio.emit('sign_result', {
                    'mode': mode,
                    'glosses': glosses,
                    'display': display,
                    'parsed': parsed if isinstance(parsed, dict) else {"terminal": parsed},
                })
                self._confirm_event.wait()
                if self._confirm_result == "confirm":
                    return parsed

    def recognize_terminal(self, required_keys: dict, socketio=None):
        if required_keys.get("terminal") is not None:
            return
        self._avatar("목적지가 어디신가요? 수어로 보여주세요.", socketio)
        terminal = self._collect_and_confirm(
            self.terminal_clf, STABLE_THRESHOLD_TERMINAL, "terminal",
            valid_set=TERMINAL_VALID, noise_set=TERMINAL_NOISE, socketio=socketio,
        )
        required_keys["terminal"] = terminal
        print(f"[저장] terminal = {terminal}")

    def recognize_num(self, required_keys: dict, socketio=None):
        self._avatar(
            "어른과 아이가 몇 명인가요? 수어로 보여주세요.\n예) 어른 → 숫자 → 아이 → 숫자",
            socketio,
        )
        result = self._collect_and_confirm(
            self.num_clf, STABLE_THRESHOLD_NUM, "num", socketio=socketio,
        )
        if required_keys.get("adult") is None:
            required_keys["adult"] = result["adult"]
            print(f"[저장] adult = {result['adult']}")
        if required_keys.get("child") is None:
            required_keys["child"] = result["child"]
            print(f"[저장] child = {result['child']}")

        # total_people = 어른 + 아이 (스키마에 있으면 자동 계산)
        if "total_people" in required_keys:
            adult = required_keys.get("adult") or 0
            child = required_keys.get("child") or 0
            required_keys["total_people"] = adult + child
            print(f"[저장] total_people = {required_keys['total_people']}")

    def run_sign_booking(self, required_keys: dict, socketio=None):
        self.recognize_terminal(required_keys, socketio)
        self.recognize_num(required_keys, socketio)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="수어 인식 파이프라인 (글로스 수집)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--region-dir", default=None)
    parser.add_argument("--numage-dir", default=None)
    args = parser.parse_args()

    cfg = KioskConfig()
    cfg.camera_index = args.camera
    if args.region_dir:
        cfg.region_dir = args.region_dir
    if args.numage_dir:
        cfg.numage_dir = args.numage_dir

    print("=" * 50)
    print("수어 파이프라인 테스트")
    print("=" * 50)

    required_keys = {
        "terminal": None,
        "date": None,
        "date_code": None,
        "time": None,
        "total_people": None,
        "seat": None,
        "adult": None,
        "child": None,
        "adult seat": None,
        "child seat": None,
        "total_price": None,
    }

    pipeline = SignPipeline(cfg)
    pipeline.run_sign_booking(required_keys)

    print("\n" + "=" * 50)
    print("[최종 required_keys]")
    for k, v in required_keys.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
