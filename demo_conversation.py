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


def _score_token(prompt: str, response_so_far: str, token: str) -> float:
    """Deterministic score for a token given the prompt and response context.

    The prompt seeds the distribution but is treated as fixed input — only the
    response-side context evolves during generation.
    """
    # Hash includes both prompt (for conditioning) and response (for autoregression)
    seed = f"prompt={prompt}|response={response_so_far}|token={token}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    base = int(digest[:8], 16) / 0xFFFFFFFF

    prompt_lower = prompt.lower()
    prompt_words = set(prompt_lower.split())
    response_words = set(response_so_far.lower().split()) if response_so_far else set()

    # Tokens related to the prompt are more probable (the model "understands" the question)
    if token in prompt_words:
        base *= 4.0

    # Tokens already in the response get a mild coherence boost
    if token in response_words:
        base *= 1.5

    # Short common words are naturally more probable
    if len(token) <= 3:
        base *= 1.8

    # Penalise immediate repetition of the last generated word
    last_word = response_so_far.split()[-1].lower() if response_so_far else ""
    if token == last_word:
        base *= 0.1

    return base


def _predict(prompt: str, response_so_far: str) -> List[TokenPrediction]:
    """Generate a probability distribution conditioned on prompt + response so far."""
    scores = [_score_token(prompt, response_so_far, tok) for tok in VOCAB]
    total = sum(scores)
    return [
        TokenPrediction(token_id=i, token=tok, probability=s / total)
        for i, (tok, s) in enumerate(zip(VOCAB, scores))
    ]


def generate(prompt: str, k: int, length: int, use_bottom: bool) -> str:
    """Generate a response to the prompt using bottom-k or top-k selection.

    The prompt is treated as fixed input.  Tokens are generated autoregressively
    for the response only, conditioned on the prompt at each step.
    """
    response_tokens = []
    guardrail = BottomKGuardrail(k=k)

    for _ in range(length):
        response_so_far = " ".join(response_tokens)
        predictions = _predict(prompt, response_so_far)

        if use_bottom:
            candidates = guardrail.apply(predictions)
            chosen = candidates[0]
        else:
            ranked = sorted(predictions, key=lambda p: p.probability, reverse=True)
            chosen = ranked[0]

        response_tokens.append(chosen.token)

    return " ".join(response_tokens)


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

    print(f"User:  {args.prompt}\n")

    if args.top_k_compare:
        top_response = generate(args.prompt, args.k, args.length, use_bottom=False)
        print(f"── Top-k response (k={args.k}) ──")
        print(f"  Assistant: {top_response}\n")

    bottom_response = generate(args.prompt, args.k, args.length, use_bottom=True)
    print(f"── Bottom-k response (k={args.k}) ──")
    print(f"  Assistant: {bottom_response}")


if __name__ == "__main__":
    main()
