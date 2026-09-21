"""Estimate how long a planned song needs, from its ABC score or its lyrics.

The webUI applies the duration as a token cap on the semantic stage, and the
planner writes a score that is often longer than a hand-typed cap. Sizing the
cap from the plan removes that truncation without asking for a number.
"""
from __future__ import annotations

import re

FRAMES_PER_SECOND = 25  # YuE2 codec frame rate: 1 semantic token = 0.04 s

_TOKEN = re.compile(r"[A-Ga-gzZxX](?:\d+|/\d+)?|[-<>^_=.]")


def _header(abc: str, pattern: str):
    return re.search(pattern, abc, re.M)


def _voice_units(body: str, units_per_bar: float) -> float:
    """Duration of one ABC voice, in units of the L: value (Z counts whole bars)."""
    body = re.sub(r'"[^"]*"', "", body)             # chord symbols
    body = re.sub(r"\[[^\]]*\]", "", body)          # inline fields
    body = re.sub(r"^%.*$", "", body, flags=re.M)   # comments
    total = 0.0
    for measure in body.split("|"):
        units, last = 0.0, 0.0
        for token in _TOKEN.findall(measure):
            if token == "-":                        # tie extends the previous note
                units += last
                continue
            if token in "<>^_=.":                   # articulations carry no duration
                continue
            letter, number = token[0], token[1:]
            if number.startswith("/"):
                value = 1 / int(number[1:]) if number[1:] else 0.5
            elif number:
                value = int(number)
            else:
                value = 1
            if letter == "Z":                       # multi-measure rest
                value *= units_per_bar
            last = value
            units += value
        total += units
    return total


def duration_seconds(abc: str | None) -> float:
    """Length implied by an ABC score, or 0.0 when there is nothing to measure."""
    if not abc or not abc.strip():
        return 0.0
    tempo_match = _header(abc, r"^Q:\S*?=\s*(\d+)")
    tempo = int(tempo_match.group(1)) if tempo_match else 120
    meter = _header(abc, r"^M:\s*(\d+)/(\d+)")
    meter_num, meter_den = (int(meter.group(1)), int(meter.group(2))) if meter else (4, 4)
    length = _header(abc, r"^L:\s*1/(\d+)")
    length_den = int(length.group(1)) if length else 16
    units_per_bar = (meter_num / meter_den) * length_den
    seconds_per_unit = 240 / (length_den * tempo)

    voices: dict[str, str] = {}
    current = None
    for line in abc.splitlines():
        match = re.match(r"^V:\s*(\S+)(.*)$", line)
        if match:
            current = match.group(1)
            voices.setdefault(current, "")
            continue
        if current and not re.match(r"^[XTMLKQPR]:", line):
            voices[current] += line
    longest = max((_voice_units(body, units_per_bar) for body in voices.values()), default=0.0)
    return longest * seconds_per_unit


def estimate_seconds_from_lyrics(lyrics: str | None) -> float:
    """Rough sung length of a lyric sheet: words dominate, section breaks add air."""
    text = lyrics or ""
    if not text.strip():
        return 0.0
    body = [line.strip() for line in text.splitlines()]
    words = len(re.findall(r"\S+", re.sub(r"\[[^\]]*\]", " ", text)))
    lines = len([line for line in body if line and not line.startswith("[")])
    markers = len([line for line in body if line.startswith("[")])
    return max(0.0, words / 2.0 + lines * 1.5 + markers * 6.0 + 20.0)


def auto_seconds(abc: str | None, lyrics: str | None, *, margin: float = 1.15) -> float:
    """Cap for the semantic stage: whatever the plan or the lyrics actually need."""
    planned = duration_seconds(abc)
    sung = estimate_seconds_from_lyrics(lyrics)
    return max(60.0, planned, sung) * margin
