"""Tests for bottom_k_guardrails module."""

import json
import subprocess
import sys

import pytest
from bottom_k_guardrails import (
    BottomKGuardrail,
    BottomKGuardrailChain,
    TokenPrediction,
    bottom_k,
)


def _make_predictions(n: int = 10) -> list[TokenPrediction]:
    """Create n predictions with probabilities 0.01, 0.02, ..., n*0.01."""
    return [
        TokenPrediction(token_id=i, token=f"tok_{i}", probability=(i + 1) * 0.01)
        for i in range(n)
    ]


class TestBottomKGuardrail:
    def test_returns_k_least_probable(self):
        preds = _make_predictions(10)
        guardrail = BottomKGuardrail(k=3)
        result = guardrail.apply(preds)
        assert len(result) == 3
        assert [p.token_id for p in result] == [0, 1, 2]

    def test_deterministic_across_calls(self):
        preds = _make_predictions(10)
        guardrail = BottomKGuardrail(k=5)
        assert guardrail.apply(preds) == guardrail.apply(preds)

    def test_ties_broken_by_token_id(self):
        preds = [
            TokenPrediction(token_id=99, token="z", probability=0.01),
            TokenPrediction(token_id=2, token="a", probability=0.01),
            TokenPrediction(token_id=50, token="m", probability=0.01),
            TokenPrediction(token_id=1, token="b", probability=0.05),
        ]
        guardrail = BottomKGuardrail(k=2)
        result = guardrail.apply(preds)
        assert [p.token_id for p in result] == [2, 50]

    def test_k_larger_than_input_returns_all(self):
        preds = _make_predictions(3)
        guardrail = BottomKGuardrail(k=100)
        result = guardrail.apply(preds)
        assert len(result) == 3

    def test_excludes_top_k_tokens(self):
        preds = _make_predictions(10)
        guardrail = BottomKGuardrail(k=3)
        result = guardrail.apply(preds)
        result_ids = {p.token_id for p in result}
        top_ids = {7, 8, 9}  # highest probability tokens
        assert result_ids.isdisjoint(top_ids)

    def test_min_tokens_passthrough(self):
        preds = _make_predictions(2)
        guardrail = BottomKGuardrail(k=1, min_tokens=5)
        result = guardrail.apply(preds)
        assert len(result) == 2  # below min_tokens, returns all

    def test_output_sorted_ascending_by_probability(self):
        preds = _make_predictions(10)
        guardrail = BottomKGuardrail(k=5)
        result = guardrail.apply(preds)
        probs = [p.probability for p in result]
        assert probs == sorted(probs)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            BottomKGuardrail(k=0)

    def test_invalid_min_tokens_raises(self):
        with pytest.raises(ValueError):
            BottomKGuardrail(k=5, min_tokens=0)

    def test_single_prediction(self):
        preds = [TokenPrediction(token_id=0, token="only", probability=1.0)]
        guardrail = BottomKGuardrail(k=1)
        result = guardrail.apply(preds)
        assert len(result) == 1
        assert result[0].token == "only"

    def test_empty_input(self):
        guardrail = BottomKGuardrail(k=5)
        assert guardrail.apply([]) == []


class TestBottomKGuardrailChain:
    def test_progressive_narrowing(self):
        preds = _make_predictions(20)
        chain = BottomKGuardrailChain(stages=[
            BottomKGuardrail(k=10),
            BottomKGuardrail(k=3),
        ])
        result = chain.apply(preds)
        assert len(result) == 3
        assert [p.token_id for p in result] == [0, 1, 2]

    def test_single_stage_same_as_direct(self):
        preds = _make_predictions(10)
        direct = BottomKGuardrail(k=4).apply(preds)
        chain = BottomKGuardrailChain(stages=[BottomKGuardrail(k=4)])
        assert chain.apply(preds) == direct

    def test_empty_chain_returns_input(self):
        preds = _make_predictions(5)
        chain = BottomKGuardrailChain()
        result = chain.apply(preds)
        assert len(result) == 5


class TestBottomKConvenience:
    def test_basic_usage(self):
        preds = [(i, f"t{i}", (i + 1) * 0.1) for i in range(5)]
        result = bottom_k(preds, k=2)
        assert len(result) == 2
        assert result[0] == (0, "t0", 0.1)
        assert result[1] == (1, "t1", 0.2)

    def test_returns_tuples(self):
        preds = [(0, "a", 0.5), (1, "b", 0.3)]
        result = bottom_k(preds, k=1)
        assert isinstance(result[0], tuple)
        assert result[0] == (1, "b", 0.3)


class TestCLI:
    """Tests that the module only runs guardrails when invoked via CLI."""

    SCRIPT = "bottom_k_guardrails.py"

    @staticmethod
    def _sample_json() -> str:
        return json.dumps([
            {"token_id": 0, "token": "the", "probability": 0.9},
            {"token_id": 1, "token": "a", "probability": 0.05},
            {"token_id": 2, "token": "xylophone", "probability": 0.001},
            {"token_id": 3, "token": "an", "probability": 0.04},
            {"token_id": 4, "token": "quarks", "probability": 0.009},
        ])

    def test_cli_returns_bottom_k(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--k", "2"],
            input=self._sample_json(),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output) == 2
        tokens = {entry["token"] for entry in output}
        assert tokens == {"xylophone", "quarks"}

    def test_cli_excludes_top_tokens(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--k", "3"],
            input=self._sample_json(),
            capture_output=True,
            text=True,
        )
        output = json.loads(result.stdout)
        tokens = {entry["token"] for entry in output}
        assert "the" not in tokens  # highest probability excluded

    def test_cli_from_file(self, tmp_path):
        pred_file = tmp_path / "preds.json"
        pred_file.write_text(self._sample_json())
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--k", "1", "--file", str(pred_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert len(output) == 1
        assert output[0]["token"] == "xylophone"

    def test_cli_requires_k(self):
        result = subprocess.run(
            [sys.executable, self.SCRIPT],
            input=self._sample_json(),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_import_does_not_run_main(self):
        """Importing the module should not trigger CLI behavior."""
        result = subprocess.run(
            [sys.executable, "-c", "import bottom_k_guardrails"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout == ""
