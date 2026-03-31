#!/usr/bin/env python3
"""Demo: generate a conversational response using bottom-k token selection.

Simulates what happens when you always pick the least probable next token
at each generation step.  Compares bottom-k output against top-k output
so you can see the contrast side by side.

Usage:
    python demo_conversation.py "what is a cat?"
    python demo_conversation.py "tell me about the moon" --k 3 --length 15
    python demo_conversation.py "hello" --top-k-compare
"""

import argparse
import hashlib
from typing import List

from bottom_k_guardrails import BottomKGuardrail, TokenPrediction

VOCAB = [
    # Common function words
    "the", "a", "an", "is", "was", "of", "in", "to", "and", "that",
    "it", "for", "on", "are", "with", "as", "at", "be", "this", "from",
    "or", "by", "not", "but", "what", "all", "were", "when", "there", "can",
    "said", "each", "which", "do", "how", "if", "will", "up", "about", "out",
    # Animals & nature
    "cat", "dog", "fish", "bird", "tree", "moon", "star", "river", "cloud", "stone",
    # Nouns & descriptors
    "small", "big", "very", "most", "like", "has", "have", "been", "one", "many",
    "animal", "pet", "fur", "soft", "warm", "kind", "also", "often", "known", "world",
    # Rare / unusual words
    "xylophone", "quasar", "zephyr", "fjord", "glyph",
    "nebula", "axiom", "prism", "vortex", "epoch",
]


def _score_token(context: str, token: str) -> float:
    """Deterministic score for a token given the full generation context."""
    digest = hashlib.sha256(f"{context}|{token}".encode()).hexdigest()
    base = int(digest[:8], 16) / 0xFFFFFFFF

    context_lower = context.lower()
    context_words = set(context_lower.split())

    # Boost contextually relevant tokens
    if token in context_words:
        base *= 3.0
    # Short common words are naturally more probable
    if len(token) <= 3:
        base *= 1.8
    # Penalise immediate repetition of the last word
    last_word = context_lower.split()[-1] if context_lower.split() else ""
    if token == last_word:
        base *= 0.1

    return base


def _predict(context: str) -> List[TokenPrediction]:
    """Generate a full probability distribution over VOCAB given context."""
    scores = [_score_token(context, tok) for tok in VOCAB]
    total = sum(scores)
    return [
        TokenPrediction(token_id=i, token=tok, probability=s / total)
        for i, (tok, s) in enumerate(zip(VOCAB, scores))
    ]


def generate(prompt: str, k: int, length: int, use_bottom: bool) -> str:
    """Generate text token-by-token using bottom-k or top-k selection.

    At each step the least (or most) probable token from the k candidates
    is chosen deterministically.
    """
    context = prompt
    tokens = []
    guardrail = BottomKGuardrail(k=k)

    for _ in range(length):
        predictions = _predict(context)

        if use_bottom:
            candidates = guardrail.apply(predictions)
            # Pick the single least probable token (first in ascending order)
            chosen = candidates[0]
        else:
            ranked = sorted(predictions, key=lambda p: p.probability, reverse=True)
            candidates = ranked[:k]
            # Pick the single most probable token
            chosen = candidates[0]

        tokens.append(chosen.token)
        context = context + " " + chosen.token

    return " ".join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate conversational responses using bottom-k token selection."
    )
    parser.add_argument("prompt", help="The input prompt / question.")
    parser.add_argument(
        "--k", type=int, default=5,
        help="Size of the bottom-k (or top-k) candidate pool (default: 5).",
    )
    parser.add_argument(
        "--length", type=int, default=12,
        help="Number of tokens to generate (default: 12).",
    )
    parser.add_argument(
        "--top-k-compare", action="store_true",
        help="Also show what top-k generation would produce for comparison.",
    )
    args = parser.parse_args()

    if args.top_k_compare:
        top_response = generate(args.prompt, args.k, args.length, use_bottom=False)
        print(f"Prompt:  \"{args.prompt}\"\n")
        print(f"── Top-k response (k={args.k}) ──")
        print(f"  {args.prompt} {top_response}\n")

    bottom_response = generate(args.prompt, args.k, args.length, use_bottom=True)

    if args.top_k_compare:
        print(f"── Bottom-k response (k={args.k}) ──")
        print(f"  {args.prompt} {bottom_response}")
    else:
        print(f"Prompt:  \"{args.prompt}\"\n")
        print(f"── Bottom-k response (k={args.k}) ──")
        print(f"  {args.prompt} {bottom_response}")


if __name__ == "__main__":
    main()
