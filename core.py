"""core.py — 수어 인식 파이프라인의 글로스 파싱 유틸리티.

각 인식 단계(목적지·날짜·시간·인원·좌석)에 대한 파싱 함수와
공통 상수를 제공한다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Digit constants
# ---------------------------------------------------------------------------

#: 숫자로 인식되는 글로스 집합 (아라비아 숫자 문자열 + 한국어 수어 수사)
NUM_DIGITS: Set[str] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구",
    "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열",
}

_DIGIT_VALUE: Dict[str, int] = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "영": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
    "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9,
    "하나": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

_ADULT_KEYWORDS: frozenset[str] = frozenset({"어른", "성인"})
_CHILD_KEYWORDS: frozenset[str] = frozenset({"아이", "어린이", "소아"})

# ---------------------------------------------------------------------------
# Terminal (destination) constants
# ---------------------------------------------------------------------------

#: KTX/SRT 주요 정차역 목록 (수어 인식 결과와 대조)
VALID_TERMINALS: Set[str] = {
    "서울", "용산", "수원", "천안아산", "오송", "대전", "김천구미",
    "동대구", "경주", "포항", "울산", "부산", "마산",
    "광주송정", "목포", "여수엑스포", "순천",
    "전주", "익산", "정읍", "광주", "나주",
    "인천", "청주공항", "춘천", "강릉", "원주",
    "제주", "진주", "창원", "평택",
    # 단축형
    "천안", "대구", "광주", "청주",
}

# ---------------------------------------------------------------------------
# Seat type constants
# ---------------------------------------------------------------------------

_SEAT_MAP: Dict[str, str] = {
    "창가": "window",
    "복도": "aisle",
    "일반실": "general",
    "특실": "special",
    "우등": "premium",
    "KTX": "ktx",
    "무궁화": "mugunghwa",
    "새마을": "saemaeul",
    "ITX": "itx",
}

# ---------------------------------------------------------------------------
# Time keyword constants
# ---------------------------------------------------------------------------

_PM_KEYWORDS: frozenset[str] = frozenset({"오후", "PM", "pm"})
_AM_KEYWORDS: frozenset[str] = frozenset({"오전", "AM", "am"})

# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def parse_terminal_glosses(glosses: List[str], valid_set: Set[str] = VALID_TERMINALS) -> Optional[str]:
    """글로스 배열에서 첫 번째 유효 목적지명을 반환.

    Args:
        glosses: 수어 인식기가 출력한 글로스 배열.
        valid_set: 허용되는 목적지 이름 집합.

    Returns:
        유효한 목적지 문자열, 없으면 None.

    Examples:
        >>> parse_terminal_glosses(["안녕", "수원", "어디"])
        '수원'
        >>> parse_terminal_glosses(["어디"]) is None
        True
    """
    for g in glosses:
        if g in valid_set:
            return g
    return None


def parse_num_glosses(glosses: List[str], digits: Set[str] = NUM_DIGITS) -> Dict[str, int]:
    """'어른/아이' 키워드로 카운트 대상을 바꾸며 숫자를 채운다.

    연속된 글로스를 순회하면서 '어른'/'성인' 키워드를 만나면 이후 숫자를
    adult 카운트에, '아이'/'어린이'/'소아' 키워드를 만나면 child 카운트에
    할당한다. 숫자 글로스가 연속되면 마지막 값이 해당 카운트를 덮어쓴다.

    Args:
        glosses: 수어 인식기가 출력한 글로스 배열.
        digits: 숫자로 인식할 글로스 집합.

    Returns:
        {"adult": n, "child": m} 형태의 dict.

    Examples:
        >>> parse_num_glosses(["어른", "3", "아이", "2"])
        {'adult': 3, 'child': 2}
        >>> parse_num_glosses(["성인", "일", "어린이", "둘"])
        {'adult': 1, 'child': 2}
    """
    result: Dict[str, int] = {"adult": 0, "child": 0}
    current_key = "adult"
    for g in glosses:
        if g in _ADULT_KEYWORDS:
            current_key = "adult"
        elif g in _CHILD_KEYWORDS:
            current_key = "child"
        elif g in digits:
            if g in _DIGIT_VALUE:
                val: Optional[int] = _DIGIT_VALUE[g]
            elif g.isdigit():
                val = int(g)
            else:
                val = None
            if val is not None:
                result[current_key] = val
    return result


def parse_date_glosses(glosses: List[str]) -> Optional[str]:
    """글로스 배열에서 날짜를 추출하여 'YYYY-MM-DD' 형식으로 반환.

    Args:
        glosses: 수어 인식기가 출력한 글로스 배열.

    Returns:
        날짜 문자열(YYYY-MM-DD), 숫자가 부족하면 None.

    Examples:
        >>> parse_date_glosses(["2026", "6", "15"])
        '2026-06-15'
        >>> parse_date_glosses(["6", "20"])  # 연도 생략 시 현재 연도 사용
        '2026-06-20'
    """
    numbers: List[int] = []
    for g in glosses:
        if g.isdigit():
            numbers.append(int(g))
        elif g in _DIGIT_VALUE:
            numbers.append(_DIGIT_VALUE[g])

    if len(numbers) >= 3:
        year, month, day = numbers[0], numbers[1], numbers[2]
        if year < 100:
            year += 2000
    elif len(numbers) == 2:
        year = datetime.now().year
        month, day = numbers[0], numbers[1]
    else:
        return None

    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_time_glosses(glosses: List[str]) -> Optional[str]:
    """글로스 배열에서 시각을 추출하여 'HH:MM' 형식으로 반환.

    '오전'/'오후' 키워드로 AM/PM을 판별한다.

    Args:
        glosses: 수어 인식기가 출력한 글로스 배열.

    Returns:
        시각 문자열(HH:MM), 숫자가 없으면 None.

    Examples:
        >>> parse_time_glosses(["오전", "10", "30"])
        '10:30'
        >>> parse_time_glosses(["오후", "2"])
        '14:00'
    """
    is_pm = False
    numbers: List[int] = []

    for g in glosses:
        if g in _PM_KEYWORDS:
            is_pm = True
        elif g in _AM_KEYWORDS:
            is_pm = False
        elif g.isdigit():
            numbers.append(int(g))
        elif g in _DIGIT_VALUE:
            numbers.append(_DIGIT_VALUE[g])

    if not numbers:
        return None

    hour = numbers[0]
    minute = numbers[1] if len(numbers) > 1 else 0

    if is_pm and hour < 12:
        hour += 12
    elif not is_pm and hour == 12:
        hour = 0

    return f"{hour:02d}:{minute:02d}"


def parse_seat_glosses(
    glosses: List[str],
    valid_seats: Optional[Set[str]] = None,
) -> Optional[str]:
    """글로스 배열에서 좌석 유형을 추출한다.

    Args:
        glosses: 수어 인식기가 출력한 글로스 배열.
        valid_seats: 허용할 좌석 키워드 집합(None이면 _SEAT_MAP 전체 사용).

    Returns:
        정규화된 좌석 유형 문자열, 없으면 None.

    Examples:
        >>> parse_seat_glosses(["창가"])
        'window'
        >>> parse_seat_glosses(["KTX"])
        'ktx'
    """
    if valid_seats is None:
        valid_seats = set(_SEAT_MAP.keys())
    for g in glosses:
        if g in valid_seats:
            return _SEAT_MAP.get(g, g)
    return None


def glosses_to_display(glosses: List[str]) -> str:
    """글로스 배열을 화면 표시용 단일 문자열로 변환."""
    return " ".join(str(g) for g in glosses)
