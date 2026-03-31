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

# ── Simple grammar model ────────────────────────────────────────────────────
# Word categories used for positional / grammatical heuristics.

DETERMINERS = {"the", "a", "an", "this", "each", "one", "many", "all"}
NOUNS = {
    "cat", "dog", "fish", "bird", "tree", "moon", "star", "river", "cloud",
    "stone", "animal", "pet", "fur", "world", "xylophone", "quasar", "zephyr",
    "fjord", "glyph", "nebula", "axiom", "prism", "vortex", "epoch",
}
ADJECTIVES = {"small", "big", "soft", "warm", "kind", "known", "many"}
VERBS = {
    "is", "was", "are", "were", "has", "have", "been", "can", "will", "do",
    "said", "like", "be",
}
ADVERBS = {"very", "most", "not", "also", "often", "about", "up", "out"}
PREPOSITIONS = {"of", "in", "to", "for", "on", "with", "as", "at", "from", "by"}
CONJUNCTIONS = {"and", "or", "but", "that", "when", "if", "which", "how", "what"}

# Bigram-style transition weights: given the POS of the previous token,
# which POS categories are likely to follow?  Higher = more probable.
_TRANSITIONS = {
    "START":       {"DETERMINER": 3.0, "NOUN": 2.0, "ADJECTIVE": 1.5, "PRONOUN": 2.5, "ADVERB": 1.0},
    "DETERMINER":  {"NOUN": 5.0, "ADJECTIVE": 3.0},
    "ADJECTIVE":   {"NOUN": 5.0, "ADJECTIVE": 1.5, "CONJUNCTION": 0.5},
    "NOUN":        {"VERB": 4.0, "PREPOSITION": 3.0, "CONJUNCTION": 2.0, "ADVERB": 1.0},
    "VERB":        {"DETERMINER": 3.0, "NOUN": 2.5, "ADJECTIVE": 2.0, "ADVERB": 2.0, "PREPOSITION": 2.0, "PRONOUN": 2.0},
    "PREPOSITION": {"DETERMINER": 3.5, "NOUN": 3.0, "ADJECTIVE": 1.5},
    "ADVERB":      {"VERB": 3.0, "ADJECTIVE": 2.5, "ADVERB": 1.0},
    "CONJUNCTION":  {"DETERMINER": 3.0, "NOUN": 2.0, "PRONOUN": 2.0, "ADJECTIVE": 1.5, "VERB": 1.5},
    "PRONOUN":     {"VERB": 4.5, "ADVERB": 1.5},
}

PRONOUNS = {"it", "there", "this", "what"}


def _pos(token: str) -> str:
    if token in DETERMINERS:
        return "DETERMINER"
    if token in PRONOUNS:
        return "PRONOUN"
    if token in NOUNS:
        return "NOUN"
    if token in ADJECTIVES:
        return "ADJECTIVE"
    if token in VERBS:
        return "VERB"
    if token in ADVERBS:
        return "ADVERB"
    if token in PREPOSITIONS:
        return "PREPOSITION"
    if token in CONJUNCTIONS:
        return "CONJUNCTION"
    return "NOUN"  # default


def _transition_weight(prev_pos: str, token: str) -> float:
    """Return how grammatically likely `token` is to follow a word of `prev_pos`."""
    tok_pos = _pos(token)
    weights = _TRANSITIONS.get(prev_pos, {})
    return weights.get(tok_pos, 0.15)  # small fallback for unlikely transitions


# ── Bigram affinity ─────────────────────────────────────────────────────────
# Hand-picked bigrams that are natural in English to give the mock model
# some semblance of fluency.

_BIGRAM_BOOST = {
    ("a", "cat"): 6.0, ("a", "dog"): 5.0, ("a", "small"): 4.0, ("a", "big"): 4.0,
    ("a", "pet"): 5.0, ("a", "bird"): 4.5, ("a", "fish"): 4.5, ("a", "kind"): 3.0,
    ("a", "warm"): 3.0, ("a", "soft"): 3.0, ("an", "animal"): 6.0, ("an", "epoch"): 4.0,
    ("the", "cat"): 5.0, ("the", "dog"): 5.0, ("the", "moon"): 6.0, ("the", "star"): 5.0,
    ("the", "world"): 5.5, ("the", "river"): 5.0, ("the", "cloud"): 4.5,
    ("the", "tree"): 4.5, ("the", "stone"): 4.0, ("the", "bird"): 4.5,
    ("is", "a"): 5.0, ("is", "an"): 4.0, ("is", "the"): 3.0, ("is", "not"): 4.5,
    ("is", "very"): 4.0, ("is", "often"): 4.0, ("is", "also"): 3.5,
    ("is", "known"): 4.5, ("is", "small"): 3.5, ("is", "warm"): 3.5,
    ("was", "a"): 4.5, ("was", "the"): 3.5, ("was", "not"): 4.0,
    ("are", "small"): 3.0, ("are", "known"): 4.0, ("are", "often"): 3.5,
    ("has", "been"): 6.0, ("has", "a"): 3.5, ("have", "been"): 5.5,
    ("have", "soft"): 3.0, ("have", "warm"): 3.0,
    ("it", "is"): 6.0, ("it", "has"): 5.0, ("it", "was"): 5.0, ("it", "can"): 4.0,
    ("can", "be"): 5.5, ("will", "be"): 4.5, ("will", "not"): 4.0,
    ("not", "be"): 3.5, ("not", "a"): 3.0,
    ("cat", "is"): 5.0, ("cat", "has"): 4.5, ("cat", "was"): 3.5,
    ("dog", "is"): 5.0, ("dog", "has"): 4.5,
    ("small", "animal"): 5.0, ("small", "cat"): 4.5, ("small", "pet"): 4.5,
    ("soft", "fur"): 6.0, ("warm", "fur"): 5.0,
    ("pet", "that"): 4.0, ("pet", "with"): 3.5,
    ("animal", "that"): 4.0, ("animal", "with"): 3.5,
    ("known", "for"): 5.5, ("known", "as"): 5.0,
    ("often", "known"): 4.0, ("often", "said"): 3.0,
    ("with", "soft"): 4.5, ("with", "warm"): 4.0, ("with", "fur"): 4.0,
    ("of", "the"): 5.0, ("in", "the"): 5.0, ("for", "the"): 4.0,
    ("and", "is"): 3.0, ("and", "has"): 3.0, ("and", "soft"): 3.0,
    ("that", "is"): 5.0, ("that", "has"): 4.0, ("that", "can"): 3.5,
    ("very", "soft"): 4.5, ("very", "warm"): 4.0, ("very", "small"): 4.0,
    ("most", "often"): 4.0, ("by", "many"): 3.5,
    ("about", "the"): 4.0, ("about", "a"): 3.0,
}

# Prompt-keyword → topic-relevant tokens with boost factors.
_TOPIC_ASSOCIATIONS = {
    "cat": {"cat": 8, "pet": 6, "animal": 6, "fur": 5, "soft": 4, "warm": 4, "small": 3, "known": 3, "kind": 3},
    "dog": {"dog": 8, "pet": 6, "animal": 6, "fur": 4, "warm": 3, "kind": 3, "known": 3, "big": 3},
    "moon": {"moon": 8, "star": 5, "cloud": 4, "stone": 3, "world": 3, "nebula": 3, "epoch": 2},
    "star": {"star": 8, "moon": 5, "cloud": 4, "nebula": 4, "world": 3, "quasar": 3},
    "fish": {"fish": 8, "animal": 5, "river": 5, "water": 3, "small": 3, "known": 3},
    "bird": {"bird": 8, "animal": 5, "tree": 4, "small": 3, "known": 3, "cloud": 3},
    "tree": {"tree": 8, "big": 4, "world": 3, "known": 3, "stone": 2},
    "world": {"world": 8, "big": 4, "known": 4, "many": 3, "most": 3},
    "sky": {"cloud": 5, "star": 5, "moon": 5, "world": 3, "nebula": 3},
    "blue": {"cloud": 3, "star": 3, "river": 3, "world": 3},
}


def _score_token(prompt: str, response_so_far: str, token: str) -> float:
    """Deterministic score for a token given the prompt and response context."""
    # Base randomness from hash — ensures variety
    seed = f"prompt={prompt}|response={response_so_far}|token={token}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    base = int(digest[:8], 16) / 0xFFFFFFFF

    prompt_lower = prompt.lower()
    prompt_words = set(prompt_lower.replace("?", "").replace("!", "").replace(".", "").split())
    response_tokens = response_so_far.lower().split() if response_so_far else []

    # ── Grammar: transition from previous token's POS ──
    prev_pos = _pos(response_tokens[-1]) if response_tokens else "START"
    grammar_weight = _transition_weight(prev_pos, token)
    base *= grammar_weight

    # ── Bigram affinity ──
    if response_tokens:
        pair = (response_tokens[-1], token)
        if pair in _BIGRAM_BOOST:
            base *= _BIGRAM_BOOST[pair]

    # ── Topic relevance from prompt ──
    for pw in prompt_words:
        assoc = _TOPIC_ASSOCIATIONS.get(pw, {})
        if token in assoc:
            base *= assoc[token]

    # ── Prompt-word boost (milder than before, grammar matters more) ──
    if token in prompt_words:
        base *= 2.0

    # ── Repetition penalty: penalise ALL prior occurrences, stacking ──
    occurrences = response_tokens.count(token)
    if occurrences > 0:
        # Each occurrence applies a multiplicative penalty
        base *= 0.08 ** occurrences
        # Extra penalty if token appeared in the last 3 positions
        recent = response_tokens[-3:] if len(response_tokens) >= 3 else response_tokens
        if token in recent:
            base *= 0.01

    # ── Mild length-based bias: after 6+ tokens, reduce run-on tendency ──
    if len(response_tokens) >= 6 and token in CONJUNCTIONS:
        base *= 0.5

    return base


def _predict(prompt: str, response_so_far: str) -> List[TokenPrediction]:
    """Generate a probability distribution conditioned on prompt + response so far."""
    scores = [_score_token(prompt, response_so_far, tok) for tok in VOCAB]
    total = sum(scores)
    return [
        TokenPrediction(token_id=i, token=tok, probability=s / total)
        for i, (tok, s) in enumerate(zip(VOCAB, scores))
    ]


def _filter_repetitions(
    candidates: List[TokenPrediction], response_tokens: List[str]
) -> List[TokenPrediction]:
    """Remove tokens that already appeared in the response, keeping only fresh picks."""
    from collections import Counter
    counts = Counter(response_tokens)
    filtered = [c for c in candidates if counts.get(c.token, 0) == 0]
    if filtered:
        return filtered
    # Fallback: allow tokens used only once
    relaxed = [c for c in candidates if counts.get(c.token, 0) < 2]
    return relaxed if relaxed else candidates[:1]


_POS_SETS = {
    "DETERMINER": DETERMINERS, "NOUN": NOUNS, "ADJECTIVE": ADJECTIVES,
    "VERB": VERBS, "ADVERB": ADVERBS, "PREPOSITION": PREPOSITIONS,
    "CONJUNCTION": CONJUNCTIONS, "PRONOUN": PRONOUNS,
}


def _allowed_tokens_for(prev_pos: str) -> set:
    """Return the set of tokens that can grammatically follow `prev_pos`."""
    allowed_pos = set(_TRANSITIONS.get(prev_pos, {}).keys())
    tokens = set()
    for pos_label in allowed_pos:
        tokens |= _POS_SETS.get(pos_label, set())
    return tokens


def _pick_grammatical_bottom(
    candidates: List[TokenPrediction],
    response_tokens: List[str],
) -> TokenPrediction:
    """From bottom-k candidates, pick the lowest-probability token that is
    grammatically valid given the previous TWO tokens' POS context.

    This produces syntactically correct but semantically incoherent output:
    the grammar guides structure while bottom-k ensures the *meaning* is wrong.
    """
    prev_pos = _pos(response_tokens[-1]) if response_tokens else "START"
    allowed = _allowed_tokens_for(prev_pos)

    # Stricter: also check that what we pick can lead somewhere valid next
    # (lookahead-1) — avoids dead-end sequences.
    grammatical = []
    for c in candidates:
        if c.token not in allowed:
            continue
        next_pos = _pos(c.token)
        # Make sure this token has at least some valid continuations
        if _TRANSITIONS.get(next_pos):
            grammatical.append(c)

    if grammatical:
        return grammatical[0]  # candidates sorted ascending by prob

    # Relaxed fallback: just grammar, no lookahead
    simple = [c for c in candidates if c.token in allowed]
    if simple:
        return simple[0]

    return candidates[0]


# ── Top-k: template-based coherent responses ────────────────────────────────
# Instead of token-by-token generation (which produces grammatical mush even
# with high-probability tokens), top-k uses pre-written sentence templates
# keyed on detected prompt topics.  This mirrors how a real LLM would produce
# a fluent, on-topic answer.

# ── Topic responses ──────────────────────────────────────────────────────────
# Each topic maps to templates grouped by question type.  "default" is used
# when the question type is not detected.

_TOPIC_RESPONSES = {
    "cat": {
        "what": [
            "a cat is a small animal with soft warm fur that is often kept as a pet and it is known to be very kind",
            "the cat is a small warm animal with soft fur that has been known as a kind pet in the world",
        ],
        "why": [
            "it is often said that the cat has soft warm fur and this kind small animal can be a very warm pet to have",
            "many have said the cat is a kind animal with soft fur and it is known as one of the most warm pet in the world",
        ],
        "how": [
            "the cat has soft warm fur and it can often be a very kind pet for many in the world",
            "a cat is known to be very soft and warm and it has often been said that this small animal can be a kind pet",
        ],
        "do": [
            "the cat is a small warm pet that has soft fur and it can often be very kind to many in the world",
            "many a cat has been known to have soft warm fur and it is often said to be a very kind small pet",
        ],
        "default": [
            "a cat is a small animal with soft warm fur that is often kept as a pet and is known to be very kind",
            "the cat is a warm kind animal with soft fur that has been known as a small pet in the world",
        ],
    },
    "dog": {
        "what": [
            "a dog is a big warm animal with soft fur that is often known as a very kind pet in the world",
            "the dog is a kind warm animal that has been known as one of the most big pet with soft fur",
        ],
        "why": [
            "it is often said that the dog is a warm kind animal and many have known it as a very big pet with soft fur",
            "the dog has been known to be very warm and kind and it is often said that this big animal can be a soft pet",
        ],
        "how": [
            "a dog can be very warm and kind and it has soft fur that is often known in many a world as a big pet",
            "the dog is known to have soft warm fur and it can often be a very kind big animal for many",
        ],
        "do": [
            "many a dog has been known to be very warm and kind with soft fur and it is often a big pet in the world",
            "the dog is a kind warm animal that can often have soft fur and has been known to be a very big pet",
        ],
        "default": [
            "a dog is a big warm animal with soft fur that has been known as a very kind pet for many an epoch",
            "the dog is often known as a kind warm pet with soft fur and it is a very big animal in the world",
        ],
    },
    "moon": {
        "what": [
            "the moon is a big stone world that can be seen with the star and cloud in many an epoch",
            "the moon is a known world of stone that has been in the cloud with many a star for an epoch",
        ],
        "why": [
            "the moon is a big stone that has been known for many an epoch and it can often be seen with the star and cloud",
            "it is often said that the moon is a known world of stone and it has been with the star for many an epoch",
        ],
        "how": [
            "the moon can often be seen as a big stone in the cloud and it is known to have been there for many an epoch",
            "the moon has been known as a big world of stone with the star and cloud for many an epoch",
        ],
        "default": [
            "the moon is a big known world of stone that can often be seen with the star and cloud for many an epoch",
            "the moon has been known for many an epoch as a big stone world with the star and cloud",
        ],
    },
    "bird": {
        "what": [
            "a bird is a small animal that can often be seen in the tree and it is known to be in many a cloud in the world",
            "the bird is a small kind animal that is known for the tree and it can often be seen in the cloud",
        ],
        "do": [
            "many a bird has been known to be in the tree and it can often be seen as a small kind animal in the cloud",
            "the bird is a small animal that can often be in the tree and it has been known to be in many a cloud",
        ],
        "where": [
            "the bird can often be seen in the tree and it is known to be a small animal in many a cloud in the world",
            "a bird is known to be in the tree and it has been seen in the cloud of many a world",
        ],
        "default": [
            "a bird is a small kind animal that can often be seen in the tree and cloud of the world",
            "the bird is known as a small animal that is often in the tree and can be seen in the cloud",
        ],
    },
    "fish": {
        "what": [
            "a fish is a small animal that has been known to be in the river and it can often be seen in many a world",
            "the fish is a known small animal of the river that has been there for many an epoch",
        ],
        "do": [
            "many a fish has been known to be in the river and it is often a small animal that can be seen in the world",
            "the fish is an animal that can often be in the river and it has been known as a small kind for many an epoch",
        ],
        "default": [
            "a fish is a small animal known to be in the river and it has been seen in many a world for an epoch",
            "the fish is a known small animal of the river that can often be seen in the world",
        ],
    },
    "sky": {
        "what": [
            "the cloud and star can often be seen in the big world and the moon has also been known to be there",
            "it is known that the cloud and star and moon are all in the big world and can often be seen",
        ],
        "why": [
            "it is said that the star can often be seen with the cloud and the moon in the big world and this is known to many",
            "the cloud and star are known to be in the world and it has been said that the moon can also be seen there",
        ],
        "default": [
            "the cloud and star are often known to be in the big world and the moon can also be seen there",
            "it is known that the cloud and star and moon can often be seen in the big world",
        ],
    },
    "tree": {
        "what": [
            "a tree is a big known kind in the world that has been there for many an epoch with stone and river",
            "the tree is a big kind that has been known in the world for many an epoch with the river and stone",
        ],
        "default": [
            "a tree is a big kind that is often known in the world and has been there with stone and river for an epoch",
            "the tree is known as a big kind in the world that has been with the river and stone for many an epoch",
        ],
    },
    "world": {
        "what": [
            "the world is a very big kind with many a star and moon and cloud that has been known for an epoch",
            "the world has many a known star and moon and it is often said to be very big with cloud and river",
        ],
        "default": [
            "the world is known to have many a star and moon and cloud and it has been very big for an epoch",
            "the world is a very big kind that has been known for many an epoch with the star and moon and cloud",
        ],
    },
}

# Fallback for prompts with no matching topic.
_GENERIC_RESPONSES = {
    "what": [
        "it is often known that there are many a kind in the world and it has been said to be very warm and soft",
        "there are many a known kind in the world that have been warm and soft for an epoch",
    ],
    "why": [
        "it is often said that many a kind in the world has been known to be very warm and soft for an epoch",
        "it has been known that the world is often a very warm and kind soft one for many an epoch",
    ],
    "how": [
        "it can often be said that many a kind in the world has been known to be very warm and soft",
        "many have been known to be in the world and it is often said to be a very warm kind for an epoch",
    ],
    "default": [
        "it is often known that there are many a warm kind in the world and it has been soft for an epoch",
        "there have been many a known kind in the world that are very warm and soft for an epoch",
    ],
}


def _detect_question_type(prompt: str) -> str:
    """Detect the question type from the first word of the prompt."""
    first = prompt.lower().split()[0] if prompt.strip() else ""
    if first in ("what", "why", "how", "do", "does", "did", "where", "when", "can", "will", "are", "is"):
        # Normalise some forms
        if first in ("does", "did", "can", "will", "are", "is"):
            return "do"
        return first
    return "default"


def _generate_topk(prompt: str) -> str:
    """Select a coherent template response based on prompt topic and question type."""
    prompt_lower = prompt.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "")
    prompt_words = prompt_lower.split()
    q_type = _detect_question_type(prompt)

    # Find the best matching topic — scan all prompt words, not just first match.
    # Also handle plurals (e.g. "dogs" → "dog", "birds" → "bird", "cats" → "cat").
    normalised = set(prompt_words)
    for w in list(normalised):
        if w.endswith("s") and len(w) > 2:
            normalised.add(w[:-1])
        if w.endswith("es") and len(w) > 3:
            normalised.add(w[:-2])

    best_topic = None
    for topic in _TOPIC_RESPONSES:
        if topic in normalised:
            best_topic = topic
            break

    if best_topic:
        topic_templates = _TOPIC_RESPONSES[best_topic]
    else:
        topic_templates = _GENERIC_RESPONSES

    # Get templates for the question type, falling back to "default"
    templates = topic_templates.get(q_type, topic_templates.get("default", []))
    if not templates:
        templates = topic_templates.get("default", list(topic_templates.values())[0])

    # Deterministic selection: hash the prompt to pick a template
    idx = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % len(templates)
    return templates[idx]


def _generate_bottomk(prompt: str, k: int, length: int) -> str:
    """Generate bottom-k response: grammatically valid but semantically wrong."""
    response_tokens: List[str] = []
    guardrail = BottomKGuardrail(k=k)

    for _ in range(length):
        response_so_far = " ".join(response_tokens)
        predictions = _predict(prompt, response_so_far)

        # 1. Filter to grammatically valid tokens first
        prev_pos = _pos(response_tokens[-1]) if response_tokens else "START"
        allowed = _allowed_tokens_for(prev_pos)
        grammatical_preds = [p for p in predictions if p.token in allowed]
        if not grammatical_preds:
            grammatical_preds = predictions

        # 2. Apply bottom-k to the grammar-filtered set (wider pool for variety)
        bottom_pool = BottomKGuardrail(k=min(len(grammatical_preds), max(k, 15)))
        candidates = bottom_pool.apply(grammatical_preds)

        # 3. Remove repetitions
        candidates = _filter_repetitions(candidates, response_tokens)

        # 4. Pick best grammatical option with lookahead
        chosen = _pick_grammatical_bottom(candidates, response_tokens)
        response_tokens.append(chosen.token)

    return " ".join(response_tokens)


def generate(prompt: str, k: int, length: int, use_bottom: bool) -> str:
    """Generate a response to the prompt.

    top-k:    template-based coherent response (simulates a real LLM).
    bottom-k: token-by-token with grammar guardrails (syntactically valid, semantically wrong).
    """
    if use_bottom:
        return _generate_bottomk(prompt, k, length)
    else:
        return _generate_topk(prompt)


def run_interactive(k: int = 5, length: int = 12, compare: bool = True) -> None:
    """Interactive REPL — type prompts and see bottom-k responses."""
    print("╔══════════════════════════════════════════════════╗")
    print("║       Bottom-K Token Guardrail Demo (REPL)      ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  k={k:<4}  length={length:<4}  compare={'on' if compare else 'off':<5}            ║")
    print("║                                                  ║")
    print("║  Commands:                                       ║")
    print("║    :k <n>       set k value                      ║")
    print("║    :length <n>  set response length               ║")
    print("║    :compare     toggle top-k comparison           ║")
    print("║    :quit        exit                              ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt == ":quit":
            print("Bye!")
            break
        elif prompt.startswith(":k "):
            try:
                k = int(prompt.split()[1])
                print(f"  (k set to {k})\n")
            except (IndexError, ValueError):
                print("  (usage: :k <number>)\n")
            continue
        elif prompt.startswith(":length "):
            try:
                length = int(prompt.split()[1])
                print(f"  (length set to {length})\n")
            except (IndexError, ValueError):
                print("  (usage: :length <number>)\n")
            continue
        elif prompt == ":compare":
            compare = not compare
            print(f"  (comparison {'on' if compare else 'off'})\n")
            continue

        if compare:
            top_response = generate(prompt, k, length, use_bottom=False)
            print(f"\n  [Top-k, k={k}]")
            print(f"  Assistant: {top_response}")

        bottom_response = generate(prompt, k, length, use_bottom=True)
        print(f"\n  [Bottom-k, k={k}]")
        print(f"  Assistant: {bottom_response}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate conversational responses using bottom-k token selection."
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Natural language input. If omitted, starts interactive mode.",
    )
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
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start interactive REPL mode.",
    )
    args = parser.parse_args()

    if args.interactive or args.prompt is None:
        run_interactive(k=args.k, length=args.length, compare=args.top_k_compare)
        return

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
