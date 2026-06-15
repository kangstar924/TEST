"""수어 인식 핵심 로직.

학습 코드(SignGRU, Dataset.__getitem__의 정규화)와 동일한 전처리/모델을 사용한다.
하나의 클래스(SignClassifier)로 region/num_age 두 모델을 똑같이 다룬다.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np

from .config import (
    FEATURE_DIM,
    LH_END,
    LH_START,
    POSE_END,
    RH_END,
    RH_START,
    SEQ_LEN,
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
