"""수어 인식 핵심 로직.

학습 코드(SignGRU, Dataset.__getitem__의 정규화)와 동일한 전처리/모델을 사용한다.
하나의 클래스(SignClassifier)로 region/num_age 두 모델을 똑같이 다룬다.
"""

from __future__ import annotations

import collections
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    CONFIDENCE_THRESHOLD,
    FEATURE_DIM,
    GLOSS_GAP_SEC,
    LH_END,
    LH_START,
    LOW_CONF_MAX,
    NUM_DIGITS,
    POSE_END,
    RAW_BUFFER_SIZE,
    RH_END,
    RH_START,
    SEQ_LEN,
    TIMEOUT_SEC,
)

# torch는 무겁고 환경에 따라 없을 수 있으므로 사용 시점에 import 한다.
try:  # pragma: no cover - 환경 의존
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    _TORCH_OK = False


# ==========================================
# 모델 (학습 코드의 SignGRU와 동일 구조)
# ==========================================
if _TORCH_OK:

    class SignGRU(nn.Module):  # type: ignore[misc]
        def __init__(self, input_size=FEATURE_DIM, hidden_size=128,
                     num_layers=2, num_classes=11):
            super().__init__()
            self.gru = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.3,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            # 0이 아닌 마지막 유효 프레임의 hidden state 사용
            nz = (x.abs().sum(dim=2) > 0)
            lengths = nz.long().sum(dim=1).clamp(min=1)
            last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
            last_hidden = out.gather(1, last_idx).squeeze(1)
            return self.classifier(last_hidden)


# ==========================================
# 키포인트 추출 / 전처리 (학습 코드와 동일)
# ==========================================
def extract_keypoints(results) -> np.ndarray:
    pose = (
        np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])


def normalize_sequence(sequence: Sequence[np.ndarray]) -> np.ndarray:
    """학습 코드 Dataset.__getitem__과 동일한 상대좌표화 + max 정규화."""
    data = np.array(sequence, dtype=np.float64).copy()
    for t in range(data.shape[0]):
        fr = data[t]

        lw = fr[LH_START:LH_START + 3].copy()
        if np.any(lw != 0):
            c = fr[LH_START:LH_END].reshape(21, 3)
            c -= lw
            fr[LH_START:LH_END] = c.flatten()

        rw = fr[RH_START:RH_START + 3].copy()
        if np.any(rw != 0):
            c = fr[RH_START:RH_END].reshape(21, 3)
            c -= rw
            fr[RH_START:RH_END] = c.flatten()

        ns = fr[0:3].copy()
        c = fr[0:POSE_END].reshape(33, 3)
        c -= ns
        fr[0:POSE_END] = c.flatten()

        data[t] = fr

    data /= (np.max(np.abs(data)) + 1e-6)
    return data


def sample_from_buffer(buffer, target_len: int = SEQ_LEN) -> List[np.ndarray]:
    """가변 길이 버퍼를 항상 target_len 프레임으로 맞춘다(보간/다운샘플)."""
    buf_list = list(buffer)
    n = len(buf_list)
    if n == 0:
        return [np.zeros(FEATURE_DIM) for _ in range(target_len)]
    if n <= target_len:
        indices = np.linspace(0, n - 1, target_len)
        result = []
        for idx_f in indices:
            lo = int(np.floor(idx_f))
            hi = min(lo + 1, n - 1)
            f = idx_f - lo
            result.append(np.array(buf_list[lo]) * (1 - f) + np.array(buf_list[hi]) * f)
        return result
    indices = np.linspace(0, n - 1, target_len, dtype=int)
    return [buf_list[i] for i in indices]


def get_hand_center(results):
    centers = []
    if results.right_hand_landmarks:
        coords = np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark])
        centers.append(coords.mean(axis=0))
    if results.left_hand_landmarks:
        coords = np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark])
        centers.append(coords.mean(axis=0))
    return np.mean(centers, axis=0) if centers else None


# ==========================================
# 라벨맵 로딩 (학습 코드가 저장한 형식 지원)
# ==========================================
def load_label_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lm_data = json.load(f)
    raw_labels = lm_data["labels"]
    if isinstance(raw_labels, dict):
        return {int(k): v for k, v in raw_labels.items()}
    if isinstance(raw_labels, list):
        return {i: name for i, name in enumerate(raw_labels)}
    raise ValueError(f"label_map.json 형식 오류: {path}")


# ==========================================
# 분류기 래퍼 (region / num_age 공용)
# ==========================================
class SignClassifier:
    def __init__(self, weights_path: str, label_map_path: str, device=None):
        if not _TORCH_OK:
            raise RuntimeError("torch가 설치되어 있지 않습니다. requirements.txt를 설치하세요.")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"라벨맵 없음: {label_map_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"모델 가중치 없음: {weights_path}")

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.idx_to_label = load_label_map(label_map_path)
        self.num_classes = len(self.idx_to_label)

        self.model = SignGRU(num_classes=self.num_classes).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @property
    def labels(self) -> List[str]:
        return [self.idx_to_label[i] for i in sorted(self.idx_to_label.keys())]

    def predict(self, sequence: Sequence[np.ndarray]
                ) -> Tuple[str, float, List[Tuple[str, float]]]:
        normalized = normalize_sequence(sequence)
        inp = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(inp), dim=1).cpu().numpy()[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(self.idx_to_label[int(i)], float(probs[i])) for i in top3_idx]
        best = int(top3_idx[0])
        return self.idx_to_label[best], float(probs[best]), top3


# ==========================================
# 글로스 배열 파싱
# ==========================================
def parse_terminal_glosses(glosses: List[str], valid_set) -> Optional[str]:
    """글로스 배열에서 첫 번째 유효 목적지명을 반환."""
    for g in glosses:
        if g in valid_set:
            return g
    return None


def parse_num_glosses(glosses: List[str], digits=NUM_DIGITS) -> Dict[str, int]:
    """'어른/아이' 키워드로 카운트 대상을 바꾸며 숫자를 채운다.

    예) [어른, 3, 아이, 2] -> {"adult": 3, "child": 2}
    """
    result = {"adult": 0, "child": 0}
    current_key = "adult"
    for g in glosses:
        if g == "어른":
            current_key = "adult"
        elif g == "아이":
            current_key = "child"
        elif g in digits:
            result[current_key] = int(g)
    return result


# ==========================================
# 글로스 수집기 (프레임 구동)
#   매 프레임 update(results)를 호출하면 내부적으로 버퍼/안정성/예측/
#   다수결/중복·노이즈 제거/gap 종료를 처리한다.
#   카메라 루프(pipeline)와 단일 윈도우 루프(kiosk) 양쪽에서 공용으로 쓴다.
# ==========================================
class GlossCollector:
    def __init__(self, classifier: "SignClassifier", mode: str,
                 stable_threshold: float, valid_set=None, noise_set=None,
                 on_message: Optional[Callable[[str], None]] = None,
                 cooldown: float = 1.2, appear_min: float = 1.0):
        self.clf = classifier
        self.mode = mode                       # "terminal" | "num"
        self.stable_threshold = stable_threshold
        self.valid_set = valid_set or set()
        self.noise_set = noise_set or set()
        self.on_message = on_message
        self.cooldown = cooldown
        self.appear_min = appear_min
        self.reset()

    def reset(self):
        self.frame_buffer = collections.deque(maxlen=RAW_BUFFER_SIZE)
        self.hand_center_history = collections.deque(maxlen=10)
        self.prediction_history = collections.deque(maxlen=7)
        self.hand_stable_frames = 0
        self.hand_appear_time: Optional[float] = None
        self.is_collecting = False
        self.hand_lost_time: Optional[float] = None
        self.result_time = 0.0
        self.glosses: List[str] = []
        self.last_gloss_time: Optional[float] = None
        self.low_conf_count = 0
        self.finished = False
        self.need_reinput = False            # 낮은 confidence 초과
        self.terminal_confirmed = False
        self.is_stable = False
        self.hand_detected = False

    # --- gap 잔여시간(초): UI 카운트다운용 ---
    def gap_remaining(self) -> Optional[float]:
        if self.glosses and self.last_gloss_time:
            return max(0.0, GLOSS_GAP_SEC - (time.time() - self.last_gloss_time))
        return None

    def _emit(self, msg: str):
        if self.on_message:
            self.on_message(msg)

    def _try_add_gloss(self, label: str, conf: float) -> bool:
        if conf < CONFIDENCE_THRESHOLD:
            self.low_conf_count += 1
            if self.low_conf_count >= LOW_CONF_MAX:
                self.need_reinput = True
                self.finished = True
            return False
        self.low_conf_count = 0

        if self.mode == "terminal" and label in self.noise_set:
            return False

        if self.glosses and self.glosses[-1] == label:
            return False

        self.glosses.append(label)
        self.last_gloss_time = time.time()
        return True

    def update(self, results) -> Optional[str]:
        """한 프레임 처리. 새로 추가된 글로스를 반환(없으면 None)."""
        if self.finished:
            return None

        self.frame_buffer.append(extract_keypoints(results))
        self.hand_detected = (results.left_hand_landmarks is not None
                              or results.right_hand_landmarks is not None)

        hc = get_hand_center(results)
        self.is_stable = False
        if hc is not None:
            self.hand_center_history.append(hc)
            if len(self.hand_center_history) >= 5:
                recent = np.array(list(self.hand_center_history)[-5:])
                self.is_stable = np.std(recent, axis=0).mean() < self.stable_threshold
                self.hand_stable_frames = (
                    self.hand_stable_frames + 1 if self.is_stable else 0
                )
            else:
                self.hand_stable_frames = 0
        else:
            self.hand_stable_frames = 0
            self.hand_center_history.clear()

        now = time.time()
        cooldown_over = (now - self.result_time >= self.cooldown)

        # gap 종료
        if self.glosses and self.last_gloss_time:
            if now - self.last_gloss_time >= GLOSS_GAP_SEC:
                self.finished = True
                return None
            if now - self.last_gloss_time >= TIMEOUT_SEC:
                self.finished = True
                return None

        added_label = None
        if self.hand_detected:
            self.hand_lost_time = None
            if self.hand_appear_time is None:
                self.hand_appear_time = now

            if not self.is_stable:
                self.is_collecting = True
            elif self.is_stable and self.is_collecting and cooldown_over:
                if (len(self.frame_buffer) >= SEQ_LEN
                        and now - self.hand_appear_time >= self.appear_min):
                    seq = sample_from_buffer(self.frame_buffer, SEQ_LEN)
                    label, conf, _ = self.clf.predict(seq)

                    self.prediction_history.append(label)
                    if len(self.prediction_history) >= 3:
                        counts: Dict[str, int] = {}
                        for p in self.prediction_history:
                            counts[p] = counts.get(p, 0) + 1
                        majority = max(counts, key=counts.get)
                        if counts[majority] / len(self.prediction_history) >= 0.5:
                            self.result_time = now
                            self.prediction_history.clear()
                            self.frame_buffer.clear()
                            self.hand_appear_time = None
                            self.is_collecting = False

                            if self._try_add_gloss(majority, conf):
                                added_label = majority
                                if (self.mode == "terminal"
                                        and majority in self.valid_set):
                                    self.terminal_confirmed = True
                                    self.finished = True
        else:
            if self.is_collecting:
                if self.hand_lost_time is None:
                    self.hand_lost_time = now
                elif now - self.hand_lost_time >= 0.5:
                    self.hand_appear_time = None
                    self.is_collecting = False
                    self.prediction_history.clear()
            else:
                self.hand_appear_time = None
                self.prediction_history.clear()

        return added_label
