#!/usr/bin/env python3
"""Demo: simulate token predictions for natural language input and apply bottom-k guardrails.

Uses a deterministic mock vocabulary with context-sensitive probabilities so
that every run with the same input produces the same output.  This lets you
see what a bottom-k guardrail does without needing a real language model.

Usage:
    python demo_bottom_k.py "what is a cat?"
    python demo_bottom_k.py "what is a cat?" --k 3
    python demo_bottom_k.py "what is a cat?" --k 10 --vocab-size 30
"""

import argparse
import hashlib
import sys
from typing import List

from bottom_k_guardrails import BottomKGuardrail, TokenPrediction

# A mock vocabulary of plausible next-tokens spanning common and rare words.
MOCK_VOCAB = [
    "the", "a", "an", "is", "was", "of", "in", "to", "and", "that",
    "it", "for", "on", "are", "with", "as", "at", "be", "this", "from",
    "or", "by", "not", "but", "what", "all", "were", "when", "there", "can",
    "said", "each", "which", "do", "how", "if", "will", "up", "about", "out",
    "cat", "dog", "fish", "bird", "tree", "moon", "star", "river", "cloud", "stone",
    "xylophone", "quasar", "zephyr", "fjord", "glyph",
    "nebula", "axiom", "prism", "vortex", "epoch",
    "antimatter", "bioluminescence", "cryptography", "dendrochronology", "epistemology",
    "fluorescence", "geomorphology", "heuristic", "isomorphism", "juxtaposition",
]


def _deterministic_probabilities(prompt: str, vocab: List[str]) -> List[float]:
    """Generate a deterministic probability distribution seeded by the prompt.

    Each token gets a raw score derived from hashing (prompt + token).  Scores
    are then normalised to sum to 1.0.  Context boosting raises the score of
    tokens that appear in the prompt so the distribution feels realistic.
    """
    raw_scores = []
    prompt_lower = prompt.lower()
    prompt_tokens = set(prompt_lower.split())

    for token in vocab:
        digest = hashlib.sha256(f"{prompt}|{token}".encode()).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF  # 0.0 – 1.0

        # Boost tokens that appear in the prompt (context relevance).
        if token in prompt_tokens:
            score *= 5.0

        # Common short words get a mild boost to mimic natural distributions.
        if len(token) <= 3:
            score *= 1.5

        raw_scores.append(score)

    total = sum(raw_scores)
    return [s / total for s in raw_scores]


def simulate_predictions(
    prompt: str, vocab_size: int = 40
) -> List[TokenPrediction]:
    """Build a mock set of token predictions for the given prompt."""
    vocab = MOCK_VOCAB[:vocab_size]
    probs = _deterministic_probabilities(prompt, vocab)

    return [
        TokenPrediction(token_id=i, token=tok, probability=prob)
        for i, (tok, prob) in enumerate(zip(vocab, probs))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: apply bottom-k guardrails to simulated token predictions."
    )
    parser.add_argument(
        "prompt",
        help="Natural language input to simulate predictions for.",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of bottom (least probable) tokens to return (default: 5).",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=40,
        help="Number of vocabulary tokens to simulate (default: 40, max: 70).",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Also print the full ranked distribution for comparison.",
    )
    args = parser.parse_args()

    vocab_size = min(args.vocab_size, len(MOCK_VOCAB))
    predictions = simulate_predictions(args.prompt, vocab_size)

    if args.show_all:
        ranked = sorted(predictions, key=lambda p: p.probability, reverse=True)
        print(f"=== Full distribution for: \"{args.prompt}\" ({vocab_size} tokens) ===\n")
        print(f"  {'Rank':<6}{'Token':<25}{'Probability':<12}{'Token ID'}")
        print(f"  {'─' * 6}{'─' * 25}{'─' * 12}{'─' * 8}")
        for rank, p in enumerate(ranked, 1):
            marker = " ◄ top" if rank <= (vocab_size - args.k) else ""
            print(f"  {rank:<6}{p.token:<25}{p.probability:<12.6f}{p.token_id}{marker}")
        print()

    guardrail = BottomKGuardrail(k=args.k)
    result = guardrail.apply(predictions)

    print(f"=== Bottom-{args.k} predictions for: \"{args.prompt}\" ===\n")
    print(f"  {'Token':<25}{'Probability':<12}{'Token ID'}")
    print(f"  {'─' * 25}{'─' * 12}{'─' * 8}")
    for p in result:
        print(f"  {p.token:<25}{p.probability:<12.6f}{p.token_id}")

    excluded_count = vocab_size - len(result)
    print(f"\n  ({excluded_count} higher-probability tokens excluded by guardrail)")


if __name__ == "__main__":
    main()
