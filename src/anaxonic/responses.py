from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from fractions import Fraction
from operator import add, mul, sub, truediv
from pathlib import Path
from typing import Any

from anaxonic.quality import quality_failures


DEFAULT_RESPONSE_DATA = "data/frontier_chat.jsonl"
LOW_CONFIDENCE_RESPONSE = (
    "I do not know that yet. Ask a clearer question or add a training example for it."
)

TOKEN_REPLACEMENTS = {
    "ya": "you",
    "u": "you",
    "r": "are",
    "ur": "your",
    "dis": "this",
    "doin": "doing",
    "gonna": "going to",
    "wanna": "want to",
    "wassup": "what is up",
    "sup": "what is up",
    "whts": "what is",
    "wtf": "what is happening",
}

FILLER_TOKENS = {
    "aight",
    "boi",
    "boy",
    "bro",
    "bruh",
    "dude",
    "g",
    "lol",
    "ma",
    "man",
    "okay",
    "umm",
    "um",
    "yo",
}


def normalize_prompt(prompt: str) -> str:
    text = prompt.casefold()
    text = text.replace("'", "")
    text = text.replace("`", "")
    text = text.replace("’", "")
    text = re.sub(r"\bwhat\s*s\b|\bwhats\b", "what is", text)
    text = re.sub(r"\bwho\s*s\b|\bwhos\b", "who is", text)
    text = re.sub(r"\byou\s*re\b|\byoure\b", "you are", text)
    text = re.sub(r"\bhow\s*s\b|\bhows\b", "how is", text)
    text = re.sub(r"\bcan\s*t\b|\bcant\b", "can not", text)
    text = re.sub(r"\bdon\s*t\b|\bdont\b", "do not", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for token in text.split():
        replacement = TOKEN_REPLACEMENTS.get(token, token)
        tokens.extend(replacement.split())
    return " ".join(tokens)


def content_tokens(prompt: str) -> set[str]:
    return {token for token in normalize_prompt(prompt).split() if token not in FILLER_TOKENS}


@dataclass(frozen=True)
class ResponseMatch:
    prompt: str
    response: str
    score: float


class ResponseMemory:
    def __init__(self, examples: list[tuple[str, str]]):
        self.examples = [(prompt, response, normalize_prompt(prompt)) for prompt, response in examples]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ResponseMemory":
        examples: list[tuple[str, str]] = []
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.extend(_examples_from_record(payload, line_number, path))
        return cls(examples)

    @classmethod
    def from_jsonl_files(cls, paths: list[str | Path]) -> "ResponseMemory":
        examples: list[tuple[str, str]] = []
        for path in paths:
            path = Path(path)
            if path.exists():
                examples.extend(cls.from_jsonl(path).raw_examples())
        return cls(examples)

    def raw_examples(self) -> list[tuple[str, str]]:
        return [(prompt, response) for prompt, response, _ in self.examples]

    def add(self, prompt: str, response: str) -> None:
        self.examples.append((prompt, response, normalize_prompt(prompt)))

    def lookup(self, prompt: str, min_score: float = 0.78) -> ResponseMatch | None:
        normalized = normalize_prompt(prompt)
        if not normalized:
            return None

        best: ResponseMatch | None = None
        for source_prompt, response, source_normalized in self.examples:
            if normalized == source_normalized:
                return ResponseMatch(source_prompt, response, 1.0)
            score = _match_score(normalized, source_normalized)
            if best is None or score > best.score:
                best = ResponseMatch(source_prompt, response, score)

        if best is None or best.score < min_score:
            return None
        return best


def built_in_response(prompt: str) -> str | None:
    normalized = normalize_prompt(prompt)
    if not normalized:
        return None

    arithmetic = _answer_arithmetic(prompt)
    if arithmetic is not None:
        return arithmetic

    tokens = content_tokens(prompt)
    if tokens <= {"hi", "hello", "hey"} or tokens <= {"what", "is", "up"}:
        return "Hey. How can I help?"
    if tokens <= {"alright", "allright", "ok"}:
        return "Alright. What would you like to work on?"
    if tokens <= {"good", "morning"}:
        return "Good morning. How can I help?"
    if {"how", "are", "you"} <= tokens or {"you", "good"} <= tokens:
        return "I am doing well and ready to help."
    if {"what", "do", "you", "know"} <= tokens:
        return "I know the local Anaxonic project basics and can answer simple questions from my training data."
    if {"what", "can", "you", "do"} <= tokens:
        return "I can answer simple questions, explain project concepts, and keep replies concise."
    if {"need", "data"} <= tokens:
        return "Tell me what topic or format you need, and I will help shape the data."
    if {"what", "is", "happening"} <= tokens or {"what", "is", "going", "on"} <= tokens:
        return "I am running normally and ready to help."
    if {"damn", "fuck", "shit"} & tokens or {"holy", "cow"} <= tokens:
        return "That response missed. Ask again and I will try a clearer reply."
    if "age" in tokens:
        return "I do not have an age. I am a local model in the Anaxonic project."
    if {"why", "important"} <= tokens:
        return "I matter when I give useful answers. Right now I am a small model being improved."
    if {"why", "like", "this"} <= tokens:
        return "Because this is still an early local model. More data, better retrieval, and more training will improve me."
    if {"what", "wrong"} <= tokens or {"whats", "wrong"} <= tokens:
        return "I am running normally, but my answers can still be weak while the model is early."
    if {"validation", "loss"} <= tokens:
        return "Validation loss measures how well the model predicts held-out examples it did not train on."
    if {"training", "loss"} <= tokens or {"train", "loss"} <= tokens:
        return "Training loss measures how well the model predicts examples it is learning from."
    if {"response", "only", "loss"} <= tokens:
        return "Response-only loss trains on assistant tokens and ignores system and user prompt tokens."
    if {"how", "many", "tokens"} <= tokens or {"token", "count"} <= tokens:
        return "I cannot see the live token count from this chat prompt."
    if {"who", "created", "you"} <= tokens or {"who", "made", "you"} <= tokens:
        return "I was built in the local Anaxonic model project."
    if {"your", "name"} <= tokens or {"what", "is", "name"} <= tokens:
        return "My name is Anaxonic."
    if ({"motor", "speed"} <= tokens) or ({"motors", "speed"} <= tokens):
        return "I need the motor specs or sensor data before I can calculate its speed."
    if {"keep", "it", "well"} <= tokens:
        return "I will keep replies clear and useful."
    if "unsure" in tokens or {"do", "not", "know"} <= tokens:
        return "I should say I am unsure instead of guessing, then ask for the missing detail."
    if set(normalized.split()) <= {"bruh", "bruhh", "bruhhh", "bro"}:
        return "Yeah, that was not a great answer. Ask me a real question and I will try to answer directly."
    if {"gimme", "give", "now"} & tokens and len(tokens) <= 4:
        return "Tell me what you want me to give you, and I will try."
    return None


def _match_score(prompt: str, candidate: str) -> float:
    prompt_tokens = {token for token in prompt.split() if token not in FILLER_TOKENS}
    candidate_tokens = {token for token in candidate.split() if token not in FILLER_TOKENS}
    if not prompt_tokens or not candidate_tokens:
        return 0.0

    overlap = len(prompt_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0
    coverage = overlap / len(prompt_tokens)
    precision = overlap / len(candidate_tokens)
    sequence = SequenceMatcher(None, prompt, candidate).ratio()

    if len(prompt_tokens) <= 2 and coverage < 1.0:
        return 0.0
    if candidate_tokens <= prompt_tokens and len(candidate_tokens) >= 2:
        coverage = 1.0
    identity_tokens = {"age", "name", "creator", "creators"}
    if (identity_tokens & prompt_tokens) != (identity_tokens & candidate_tokens):
        sequence *= 0.65
    return (coverage * 0.5) + (precision * 0.25) + (sequence * 0.25)


def _answer_arithmetic(normalized: str) -> str | None:
    normalized = normalize_prompt(
        normalized.replace("+", " plus ")
        .replace("-", " minus ")
        .replace("*", " times ")
        .replace("/", " divided by ")
    )
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for word, digit in number_words.items():
        normalized = re.sub(rf"\b{word}\b", digit, normalized)
    match = re.fullmatch(
        r"(?:what is |calculate |solve )?(-?\d+)\s*(plus|minus|times|multiplied by|x|divided by)\s*(-?\d+)",
        normalized,
    )
    if match is None:
        return None

    left = int(match.group(1))
    operator = match.group(2)
    right = int(match.group(3))
    operations = {
        "plus": add,
        "minus": sub,
        "times": mul,
        "multiplied by": mul,
        "x": mul,
        "divided by": truediv,
    }
    if operator == "divided by" and right == 0:
        return "Division by zero is undefined."

    result = operations[operator](left, right)
    if isinstance(result, float):
        result = Fraction(result).limit_denominator()
        if result.denominator == 1:
            result_text = str(result.numerator)
        else:
            result_text = f"{result.numerator}/{result.denominator}"
    else:
        result_text = str(result)
    return f"The answer is {result_text}."


def _examples_from_record(
    payload: dict[str, Any], line_number: int, path: str | Path
) -> list[tuple[str, str]]:
    if "prompt" in payload and "response" in payload:
        return [(str(payload["prompt"]), str(payload["response"]))]
    if "messages" not in payload:
        raise ValueError(
            f"{path}:{line_number} must contain either messages or prompt/response fields"
        )

    messages = payload["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"{path}:{line_number} messages must be a list")

    examples = []
    last_user: str | None = None
    for message in messages:
        role = str(message["role"]).strip().lower()
        content = str(message["content"])
        if role == "user":
            last_user = content
        elif role == "assistant" and last_user is not None:
            examples.append((last_user, content))
            last_user = None
    return examples


def response_is_low_quality(response: str) -> bool:
    cleaned = response.strip()
    if quality_failures(cleaned):
        return True
    if not cleaned:
        return True
    if "�" in cleaned:
        return True

    words = re.findall(r"[A-Za-z]+", cleaned)
    if len(cleaned) < 12 and len(words) <= 2:
        return True
    if len(words) >= 2 and len(set(word.casefold() for word in words)) == 1:
        return True
    if len(words) >= 6:
        folded_words = [word.casefold() for word in words]
        unique_word_ratio = len(set(folded_words)) / len(folded_words)
        average_word_length = sum(len(word) for word in folded_words) / len(folded_words)
        max_word_frequency = max(folded_words.count(word) for word in set(folded_words))
        if average_word_length < 4.5 and unique_word_ratio < 0.8 and max_word_frequency >= 2:
            return True
    if len(words) >= 12:
        folded_words = [word.casefold() for word in words]
        short_word_ratio = sum(len(word) <= 4 for word in folded_words) / len(folded_words)
        unique_word_ratio = len(set(folded_words)) / len(folded_words)
        if short_word_ratio > 0.75 and unique_word_ratio < 0.65:
            return True
    if re.fullmatch(r"(ye|yes?)\s+(an|ans|d|s|t)[\w\s.]*", cleaned.casefold()):
        return True
    normalized_cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned.casefold())
    generic_non_answers = {
        "i will use a short and simple answer",
        "i will use a short and simple answer.",
        "i will provide a concise answer",
        "i will provide a concise answer.",
        "sure the was a can compare dere",
        "sure! the was a can compare dere.",
    }
    if cleaned.casefold() in generic_non_answers or normalized_cleaned in generic_non_answers:
        return True
    if len(words) >= 6:
        folded = [word.casefold() for word in words]
        malformed = sum(
            word.endswith(("ints", "dere", "prosses", "bentes")) or word in {"sentime"}
            for word in folded
        )
        if malformed >= 1 and unique_word_ratio < 0.95:
            return True
    return False
