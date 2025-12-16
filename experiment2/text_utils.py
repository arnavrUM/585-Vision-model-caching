import re
from typing import Sequence


def normalize_text(text: str | None) -> str:
    """Lowercase and strip punctuation for loose string comparisons."""
    if not text:
        return ""
    lowered = text.lower()
    return " ".join(re.findall(r"[a-z0-9]+", lowered))


def answer_matches(reference: str | Sequence[str | None] | None, generated: str) -> bool | None:
    """Return True if any reference text appears to match the generated answer."""
    if isinstance(reference, str) or reference is None:
        references: Sequence[str | None] = [reference]
    else:
        references = reference

    normalized_refs: list[str] = []
    for ref in references:
        normalized = normalize_text(ref)
        if normalized:
            normalized_refs.append(normalized)
    if not normalized_refs:
        return None

    normalized_generated = normalize_text(generated)
    if not normalized_generated:
        return False

    for normalized_ref in normalized_refs:
        if normalized_ref in normalized_generated or normalized_generated in normalized_ref:
            return True
    return False
