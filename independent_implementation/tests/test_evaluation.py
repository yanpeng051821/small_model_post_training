import pytest

from post_training_core.evaluation import classify_generation


@pytest.mark.parametrize(
    ("parsed", "verified", "eos", "finish", "error", "expected"),
    [
        ("42", True, True, "stop", None, ("correct",)),
        ("41", False, True, "stop", None, ("wrong_answer",)),
        (None, None, True, "stop", None, ("unparseable",)),
        (
            None,
            None,
            False,
            "length",
            None,
            ("unparseable", "no_eos", "length_truncated"),
        ),
        (
            None,
            None,
            False,
            "error",
            "CUDA OOM",
            ("runtime_error", "no_eos"),
        ),
    ],
)
def test_classifies_generation_health(parsed, verified, eos, finish, error, expected):
    outcome = classify_generation(
        parsed_answer=parsed,
        verifier_passed=verified,
        generated_eos=eos,
        finish_reason=finish,
        runtime_error=error,
    )

    assert outcome.labels == expected
