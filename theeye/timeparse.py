"""Turn the time words in a search ("today", "this morning", "yesterday after 5pm", "past 2 hours") into a range.

Done in plain code rather than by the LLM: small local models get relative dates wrong surprisingly often,
and search must work with the LLM turned off. Matched words are removed, so the rest of the question can be
used for "looks like" matching without "today" in it.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Optional

PARTS = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (20, 6)}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "couple of": 2, "couple": 2, "few": 3}
UNITS = {"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 7 * 86400}
NUM = r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple of|couple|few)"
CLOCK = r"((\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?|noon|midnight)"   # groups: whole, h, m, am/pm


@dataclass
class TimeRange:
    start: Optional[dt.datetime] = None
    end: Optional[dt.datetime] = None
    hours: list[int] = field(default_factory=list)     # recurring time of day, e.g. "at night" with no date
    label: str = ""                                     # how it was understood, for the results line
    rest: str = ""                                      # the question with the time words removed

    @property
    def found(self) -> bool:
        return bool(self.start or self.end or self.hours)


def _num(s: str) -> int:
    return int(s) if s.isdigit() else NUMS.get(s, 1)


def _clock(m: re.Match, pos: int, meridiem_hint: Optional[str] = None) -> Optional[tuple[int, int, Optional[str]]]:
    """Read a CLOCK group starting at group index pos. Returns (hour 0-23, minute, meridiem)."""
    whole = m.group(pos)
    if whole is None:
        return None
    w = whole.strip().lower()
    if w == "noon":
        return 12, 0, "pm"
    if w == "midnight":
        return 0, 0, "am"
    h, mi, ap = int(m.group(pos + 1)), int(m.group(pos + 2) or 0), (m.group(pos + 3) or "").replace(".", "")
    ap = ap or meridiem_hint or ""
    if h > 23 or mi > 59:
        return None
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    return h, mi, ap or None


def _day_label(d: dt.date, today: dt.date) -> str:
    if d == today:
        return "Today"
    if d == today - dt.timedelta(days=1):
        return "Yesterday"
    return d.strftime("%a %b %-d")


def parse(q: str, now: Optional[dt.datetime] = None) -> TimeRange:
    now = now or dt.datetime.now()
    today = now.date()
    text = " " + q.lower() + " "
    day: Optional[dt.date] = None
    part: Optional[str] = None
    after = before = None          # (hour, minute)
    around = None
    out = TimeRange()

    def take(pattern: str):
        nonlocal text
        m = re.search(pattern, text)
        if m:
            text = text[:m.start()] + " " + text[m.end():]
        return m

    def midnight(d: dt.date) -> dt.datetime:
        return dt.datetime.combine(d, dt.time())

    # Windows that end now: "past 2 hours", "last 30 minutes", "in the last day", "recently"
    m = take(rf"\b(?:in\s+the\s+|within\s+the\s+|over\s+the\s+)?(?:past|last)\s+(?:{NUM}\s+)?(minute|min|hour|hr|day|week)s?\b")
    if m:
        secs = _num(m.group(1) or "1") * UNITS[m.group(2)]
        out.start, out.end = now - dt.timedelta(seconds=secs), now
        n = _num(m.group(1) or "1")
        out.label = f"Past {n} {m.group(2).replace('min', 'minute').replace('hr', 'hour')}{'s' if n != 1 else ''}"
    elif take(r"\brecent(?:ly)?\b|\blately\b"):
        out.start, out.end, out.label = now - dt.timedelta(hours=24), now, "Past 24 hours"
    elif take(r"\blast\s+night\b"):
        y = today - dt.timedelta(days=1)
        out.start, out.end, out.label = midnight(y).replace(hour=20), midnight(today).replace(hour=6), "Last night"
    elif take(r"\btonight\b"):
        out.start, out.end, out.label = midnight(today).replace(hour=17), now, "Tonight"
    elif take(r"\bthis\s+week\b"):
        out.start, out.end, out.label = midnight(today - dt.timedelta(days=today.weekday())), now, "This week"
    elif take(r"\blast\s+week\b"):
        mon = today - dt.timedelta(days=today.weekday())
        out.start, out.end, out.label = midnight(mon - dt.timedelta(days=7)), midnight(mon), "Last week"
    if out.start:
        out.rest = _tidy(text)
        return out

    # A day, a part of the day, and/or clock times, combined
    m = take(r"\bthis\s+(morning|afternoon|evening)\b")
    if m:
        day, part = today, m.group(1)
    if take(r"\b(?:the\s+)?day\s+before\s+yesterday\b"):
        day = today - dt.timedelta(days=2)
    elif take(r"\byesterday\b"):
        day = today - dt.timedelta(days=1)
    elif take(r"\btoday\b|\bso\s+far\b"):
        day = today
    m = take(rf"\b{NUM}\s+days?\s+ago\b")
    if m:
        day = today - dt.timedelta(days=_num(m.group(1)))
    m = take(r"\b(?:on\s+|last\s+|this\s+(?:past\s+)?)?(" + "|".join(WEEKDAYS) + r")\b")
    if m:
        back = (today.weekday() - WEEKDAYS.index(m.group(1))) % 7
        if back == 0 and m.group(0).strip().startswith("last"):
            back = 7
        day = today - dt.timedelta(days=back)
    m = take(r"\b(?:in\s+the\s+|during\s+the\s+|at\s+)?(morning|afternoon|evening|night)\b")
    if m and not part:
        part = m.group(1)

    m = take(rf"\bbetween\s+{CLOCK}\s+(?:and|-|to)\s+{CLOCK}\b")
    if m:
        b = _clock(m, 5)
        a = _clock(m, 1, b[2] if b else None)
        if a and b:
            after, before = a[:2], b[:2]
    m = take(rf"\b(?:after|since|from)\s+{CLOCK}\b")
    if m and (c := _clock(m, 1)) and (m.group(4) or ":" in m.group(0) or m.group(1) in ("noon", "midnight")):
        after = c[:2]
    m = take(rf"\b(?:before|until|till|til|by)\s+{CLOCK}\b")
    if m and (c := _clock(m, 1)) and (m.group(4) or ":" in m.group(0) or m.group(1) in ("noon", "midnight")):
        before = c[:2]
    m = take(rf"\b(?:at|around|about|near)\s+{CLOCK}\b")
    if m and (c := _clock(m, 1)) and (m.group(4) or ":" in m.group(0) or m.group(1) in ("noon", "midnight")):
        around = c[:2]

    out.rest = _tidy(text)
    clock = after or before or around
    if not (day or part or clock):
        return out
    if part and not day and not clock:            # "at night", "in the morning": every day at that time
        a, b = PARTS[part]
        out.hours = list(range(a, b)) if a < b else list(range(a, 24)) + list(range(0, b))
        out.label = f"Any day, {part} ({a:02d}:00–{b:02d}:00)"
        return out

    if day is None:                                # times without a day: the most recent such time
        day = today
        first = after or around or before
        if first and midnight(today).replace(hour=first[0], minute=first[1]) > now:
            day = today - dt.timedelta(days=1)
    start, end = midnight(day), midnight(day + dt.timedelta(days=1))
    what = _day_label(day, today)
    if part:
        what = f"This {part}" if day == today and part != "night" else f"{what} {part}"
        a, b = PARTS[part]
        start = midnight(day).replace(hour=a)
        end = midnight(day + dt.timedelta(days=1 if b < a else 0)).replace(hour=b)
    if around:
        t = midnight(day).replace(hour=around[0], minute=around[1])
        start, end = t - dt.timedelta(minutes=30), t + dt.timedelta(minutes=30)
    if after:
        start = midnight(day).replace(hour=after[0], minute=after[1])
    if before:
        end = midnight(day).replace(hour=before[0], minute=before[1])
        if end <= start:
            end += dt.timedelta(days=1)
    end = min(end, max(now, start + dt.timedelta(minutes=1)))
    out.start, out.end = start, end
    span = f"{start:%H:%M}–{'now' if end == now else f'{end:%H:%M}'}"
    out.label = f"{what}, {span}" if (part or clock) else (f"{what} (so far)" if end == now else what)
    return out


FILLER = {"in", "the", "on", "at", "from", "during", "of", "since", "around", "and", "for", "any", "between",
          "to", "by", "until", "after", "before", "this", "that", "was", "were", "who", "show", "me", "find",
          "all", "there", "anything", "any"}


def _tidy(text: str) -> str:
    words = re.sub(r"[?!.,]+", " ", text).split()
    while words and words[-1] in FILLER:
        words.pop()
    while words and words[0] in FILLER:
        words.pop(0)
    return " ".join(words)
