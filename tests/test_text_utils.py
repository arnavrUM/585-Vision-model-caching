from experiment2.text_utils import answer_matches, normalize_text


def test_answer_matches_prefers_short_reference() -> None:
    references = ["laptop", "The laptop is on the table."]
    assert answer_matches(references, "A laptop is on the table.") is True


def test_answer_matches_handles_short_generation() -> None:
    references = ["No, there are no calculators or pictures."]
    assert answer_matches(references, "no") is True


def test_answer_matches_returns_none_without_reference() -> None:
    assert answer_matches([], "any response") is None


def test_normalize_text_drops_punctuation() -> None:
    assert normalize_text("A Loud Clock!!!") == "a loud clock"
