"""설정값 모음.

- 모델/라벨맵 경로는 환경변수로 덮어쓸 수 있고, 없으면 Windows 기본 경로를 사용한다.
- 좌표 구조, 시퀀스 길이 등 전처리 상수는 학습 코드와 반드시 동일해야 한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ==========================================
# 좌표 구조 (학습 코드와 동일해야 함)
#   pose(33*3=99) + left_hand(21*3=63) + right_hand(21*3=63) = 225
# ==========================================
POSE_END = 33 * 3            # 99
LH_START = POSE_END          # 99
LH_END = LH_START + 21 * 3   # 162
RH_START = LH_END            # 162
RH_END = RH_START + 21 * 3   # 225
FEATURE_DIM = RH_END         # 225

SEQ_LEN = 30
CONFIDENCE_THRESHOLD = 0.5

# ==========================================
# 글로스(gloss) 수집 / 파싱 규칙
# ==========================================
RAW_BUFFER_SIZE = 80

# 모드별 손 안정성 임계값 (작을수록 더 멈춰 있어야 인식)
STABLE_THRESHOLD_TERMINAL = 0.008
STABLE_THRESHOLD_NUM = 0.02

GLOSS_GAP_SEC = 2.0    # 마지막 글로스 후 이 시간 이상 공백이면 수집 종료
TIMEOUT_SEC = 8.0      # 첫 글로스 이후 전체 타임아웃
LOW_CONF_MAX = 3       # 낮은 confidence 연속 허용 횟수(초과 시 재입력 요청)

# 목적지(터미널) 라벨: 유효 역명 / 노이즈
TERMINAL_VALID = {
    "강릉", "고양", "광명", "대구", "수원",
    "영등포", "용인", "의정부", "인천", "잠실",
    "전주", "청량리", "평택",
}
TERMINAL_NOISE = {"가다", "어떻게"}

# 숫자/인원 라벨: 키워드 / 숫자
NUM_KEYWORD = {"어른", "아이"}
NUM_DIGITS = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}

# ==========================================
# 한글 폰트 후보 (OS 무관 동작을 위해 여러 경로 시도)
# ==========================================
FONT_CANDIDATES = (
    os.environ.get("SIGN_FONT_PATH", ""),
    "C:/Windows/Fonts/malgun.ttf",          # Windows 맑은 고딕
    "C:/Windows/Fonts/malgunbd.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "/Library/Fonts/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _default_dir(env_key: str, fallback: str) -> str:
    return os.environ.get(env_key, fallback)


@dataclass
class ModelConfig:
    """단일 수어 모델 묶음(가중치 + 라벨맵)."""

    name: str
    weights_path: str
    label_map_path: str


@dataclass
class KioskConfig:
    """키오스크 전체 설정."""

    region_dir: str = field(
        default_factory=lambda: _default_dir(
            "SIGN_REGION_DIR", r"C:\Users\Admin\Desktop\MORE_NEW"
        )
    )
    numage_dir: str = field(
        default_factory=lambda: _default_dir(
            "SIGN_NUMAGE_DIR", r"C:\Users\Admin\Desktop\NUM_AGE"
        )
    )
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    enable_tts: bool = True

    def region_model(self) -> ModelConfig:
        return ModelConfig(
            name="목적지",
            weights_path=os.path.join(self.region_dir, "best_model_region.pth"),
            label_map_path=os.path.join(self.region_dir, "label_map.json"),
        )

    def numage_model(self) -> ModelConfig:
        return ModelConfig(
            name="숫자",
            weights_path=os.path.join(self.numage_dir, "best_model_num_age.pth"),
            label_map_path=os.path.join(self.numage_dir, "label_map.json"),
        )
