from __future__ import annotations

import re
from collections import Counter


WORD_RE = re.compile(r"[a-zA-Z0-9']+")
SUSPECT_FRAGMENTS = (
    "ang",
    "arer",
    "cher",
    "eler",
    "erer",
    "lere",
    "thel",
    "wing",
)


def repeated_character_run(text: str) -> int:
    longest = 0
    current = 0
    previous = ""
    for char in text:
        if char == previous:
            current += 1
        else:
            previous = char
            current = 1
        longest = max(longest, current)
    return longest


def word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def repeated_word_fraction(text: str) -> float:
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(tokens)


def max_repeated_ngram_count(text: str, n: int = 3) -> int:
    tokens = word_tokens(text)
    if len(tokens) < n:
        return 0
    ngrams = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return max(ngrams.values(), default=0)


def repeated_fragment_count(text: str) -> int:
    lowered = text.casefold()
    return sum(lowered.count(fragment) for fragment in SUSPECT_FRAGMENTS)


def unique_word_fraction(text: str) -> float:
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def quality_failures(text: str) -> list[str]:
    failures: list[str] = []
    stripped = text.strip()
    tokens = word_tokens(stripped)
    if not stripped:
        failures.append("empty response")
        return failures
    if repeated_character_run(stripped) >= 5:
        failures.append("repeated character run")
    if repeated_fragment_count(stripped) >= 6:
        failures.append("repeated generated fragments")
    if len(tokens) >= 8 and unique_word_fraction(stripped) < 0.45:
        failures.append("low unique word fraction")
    if len(tokens) >= 12 and repeated_word_fraction(stripped) > 0.45:
        failures.append("excessive repeated words")
    if max_repeated_ngram_count(stripped, n=3) >= 3:
        failures.append("repeated word sequence")
    return failures
