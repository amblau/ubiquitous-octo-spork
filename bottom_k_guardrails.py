"""Deterministic bottom-k token prediction guardrails.

Provides guardrails that consistently filter token predictions to return
only the bottom-k (least probable) tokens, removing all top-k candidates.
All operations are deterministic — no randomness is involved in selection.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class TokenPrediction:
    """A single token with its associated probability."""
    token_id: int
    token: str
    probability: float


@dataclass
class BottomKGuardrail:
    """Deterministic guardrail that returns only bottom-k token predictions.

    Given a full set of token predictions sorted by probability, this guardrail
    strips away the top-ranked tokens and returns exclusively the k least
    probable ones.  Ties at the boundary are broken by token_id (lower id is
    kept) to guarantee deterministic output.

    Attributes:
        k: Number of bottom (least probable) tokens to retain.
        min_tokens: Minimum tokens required in input for the guardrail to
            activate.  If fewer tokens are supplied, all are returned as-is.
    """
    k: int
    min_tokens: int = 1

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")

    def apply(
        self, predictions: List[TokenPrediction]
    ) -> List[TokenPrediction]:
        """Filter predictions to only the bottom-k least probable tokens.

        Returns a list sorted ascending by probability (least probable first).
        Deterministic: ties in probability are broken by token_id ascending.
        """
        if len(predictions) < self.min_tokens:
            return sorted(predictions, key=lambda p: (p.probability, p.token_id))

        ranked = sorted(predictions, key=lambda p: (p.probability, p.token_id))
        return ranked[: self.k]


@dataclass
class BottomKGuardrailChain:
    """Chain multiple guardrails with successive bottom-k narrowing.

    Each stage takes the output of the previous stage and applies a further
    bottom-k filter.  This lets you progressively isolate the very least
    probable tokens in a deterministic pipeline.

    Example:
        chain = BottomKGuardrailChain(stages=[
            BottomKGuardrail(k=50),
            BottomKGuardrail(k=10),
        ])
        result = chain.apply(predictions)  # bottom-10 of the bottom-50
    """
    stages: List[BottomKGuardrail] = field(default_factory=list)

    def apply(
        self, predictions: List[TokenPrediction]
    ) -> List[TokenPrediction]:
        result = list(predictions)
        for stage in self.stages:
            result = stage.apply(result)
        return result


def bottom_k(
    predictions: List[Tuple[int, str, float]], k: int
) -> List[Tuple[int, str, float]]:
    """Convenience function: return the k least probable token predictions.

    Args:
        predictions: List of (token_id, token_str, probability) tuples.
        k: Number of bottom tokens to keep.

    Returns:
        The k least probable predictions sorted ascending by probability,
        with ties broken deterministically by token_id.
    """
    guardrail = BottomKGuardrail(k=k)
    typed = [TokenPrediction(tid, tok, prob) for tid, tok, prob in predictions]
    result = guardrail.apply(typed)
    return [(p.token_id, p.token, p.probability) for p in result]
