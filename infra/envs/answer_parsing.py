"""Numeric answer extraction for \\boxed{} math tasks. Ported from CS285 HW4
(hw4/utils/answer_parsing.py), XML paths dropped."""

from __future__ import annotations

import re

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?")
BOXED_START_RE = re.compile(r"\\boxed\s*\{")
LATEX_FRAC_RE = re.compile(r"\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}")
LATEX_SIGNED_FRAC_RE = re.compile(r"([+-]?)\s*\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}")
LATEX_MIXED_FRAC_RE = re.compile(r"([-+]?\d+)\s*\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}")
SIMPLE_MIXED_FRAC_RE = re.compile(r"([-+]?\d+)\s+([-+]?\d+)\s*/\s*([-+]?\d+)")
TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm)\s*\{(.*)\}", re.DOTALL)
PLAIN_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def strip_think_blocks(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).replace("</think>", "").strip()


def parse_number(text: str) -> float | None:
    t = text.strip()
    if not t:
        return None
    t = t.replace("\\$", "").replace("$", "").replace(",", "")
    t = t.replace("\\left", "").replace("\\right", "").strip()
    if t.startswith("{") and t.endswith("}") and len(t) >= 2:
        inner = t[1:-1].strip()
        if inner:
            t = inner
    text_wrap = TEXT_WRAPPER_RE.fullmatch(t)
    if text_wrap:
        return parse_number(text_wrap.group(1))

    def _frac(num: float, den: float) -> float | None:
        return None if abs(den) < 1e-12 else num / den

    m = LATEX_SIGNED_FRAC_RE.fullmatch(t)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        v = _frac(float(m.group(2)), float(m.group(3)))
        return None if v is None else sign * v
    m = LATEX_MIXED_FRAC_RE.fullmatch(t)
    if m:
        whole = float(m.group(1))
        v = _frac(float(m.group(2)), float(m.group(3)))
        if v is None:
            return None
        return whole - abs(v) if whole < 0 else whole + abs(v)
    m = LATEX_FRAC_RE.fullmatch(t)
    if m:
        return _frac(float(m.group(1)), float(m.group(2)))
    m = SIMPLE_MIXED_FRAC_RE.fullmatch(t)
    if m:
        whole = float(m.group(1))
        v = _frac(float(m.group(2)), float(m.group(3)))
        if v is None:
            return None
        return whole - abs(v) if whole < 0 else whole + abs(v)
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", t):
        num_s, den_s = (x.strip() for x in t.split("/", 1))
        try:
            return _frac(float(num_s), float(den_s))
        except ValueError:
            return None
    if not PLAIN_NUMBER_RE.fullmatch(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def extract_last_number(text: str) -> float | None:
    nums = NUMBER_RE.findall(strip_think_blocks(text))
    return parse_number(nums[-1]) if nums else None


def _find_matching_closing_brace(text: str, opening_brace_idx: int) -> int | None:
    depth = 0
    for i in range(opening_brace_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_last_boxed_content(text: str) -> str | None:
    cleaned = strip_think_blocks(text)
    for m in reversed(list(BOXED_START_RE.finditer(cleaned))):
        open_idx = cleaned.find("{", m.start())
        if open_idx < 0:
            continue
        close_idx = _find_matching_closing_brace(cleaned, open_idx)
        if close_idx is None:
            continue
        return cleaned[open_idx + 1 : close_idx].strip()
    return None


def extract_number_from_boxed_answer(text: str) -> float | None:
    content = extract_last_boxed_content(text)
    if content is None:
        return None
    # Intentionally strict: avoid mapping symbolic answers to a scalar.
    return parse_number(content)
