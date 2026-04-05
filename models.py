from uagents import Model
from typing import Optional


class LyricLine(Model):
    line_id: int
    original: str
    translated: str
    target_lang: str
    target_syllables: int


class CritiqueResult(Model):
    line_id: int
    critic: str       # "syllable" | "stress"
    passed: bool
    syllable_delta: int
    feedback: str


class RevisionRequest(Model):
    line_id: int
    original: str
    current_translation: str
    target_lang: str
    target_syllables: int
    revision_prompt: str
    attempt_number: int


class AcceptedLine(Model):
    line_id: int
    final_translation: str
    attempts: int