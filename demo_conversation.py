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
# These simulate what a real LLM would produce with top-k sampling: fluent,
# on-topic, natural-sounding answers.  Unlike the bottom-k path, these are
# not constrained to the mock VOCAB — a real model's top-k draws from its
# full vocabulary.

# Each topic maps to templates grouped by question type.

_TOPIC_RESPONSES = {
    "cat": {
        "what": [
            "A cat is a small domesticated animal known for its soft fur, independent personality, and sharp senses. Cats have been kept as pets for thousands of years and are one of the most popular companion animals in the world.",
            "A cat is a furry domesticated mammal that people commonly keep as a pet. They are known for being curious, agile, and affectionate, and they typically have soft fur, retractable claws, and excellent night vision.",
        ],
        "why": [
            "Cats are popular pets because they are relatively low-maintenance, naturally clean, and form strong bonds with their owners. Their independent nature means they can be left alone for longer periods compared to dogs, which makes them well-suited to many lifestyles.",
            "There are many reasons people love cats. They are soft, warm, and affectionate companions that can also be quite playful. Their purring has even been shown to have a calming effect on humans.",
        ],
        "how": [
            "Cats communicate through a combination of vocalizations like meowing and purring, body language such as tail position and ear orientation, and scent marking. Each cat develops its own unique way of interacting with its owner over time.",
            "Cats are typically cared for by providing them with fresh food and water, a clean litter box, regular veterinary checkups, and plenty of affection and play. They are generally easy to care for compared to many other pets.",
        ],
        "do": [
            "Yes, cats are known to be quite independent, but they also enjoy companionship and can form deep bonds with their owners. Many cats are playful, curious, and affectionate, though each one has its own unique personality.",
            "Cats do have a wide range of behaviors and preferences. Some are very social and love attention, while others prefer solitude. Most cats enjoy playing, napping in warm spots, and exploring their surroundings.",
        ],
        "default": [
            "A cat is a small domesticated animal with soft fur that is commonly kept as a pet. They are known for being curious, independent, and affectionate companions that have lived alongside humans for thousands of years.",
            "Cats are small, furry animals that are among the most popular pets worldwide. They are valued for their companionship, playful nature, and ability to form close bonds with their owners.",
        ],
    },
    "dog": {
        "what": [
            "A dog is a domesticated animal and one of the most popular pets in the world. Dogs are known for their loyalty, intelligence, and friendly nature, and they come in a wide variety of breeds, sizes, and temperaments.",
            "A dog is a loyal and social animal that has been a companion to humans for thousands of years. They are highly trainable, affectionate, and are often considered to be a member of the family.",
        ],
        "why": [
            "Dogs are beloved because of their unwavering loyalty and companionship. They are social animals that thrive on interaction with humans, and their ability to be trained for a wide range of tasks makes them incredibly versatile partners.",
            "People love dogs for their warm, friendly nature and their ability to form deep emotional bonds. Dogs are also highly adaptable and can serve as working animals, therapy companions, or simply loving household pets.",
        ],
        "how": [
            "Dogs communicate through barking, tail wagging, body posture, and facial expressions. They are highly attuned to human emotions and can often sense when their owner is happy, sad, or stressed.",
            "Caring for a dog involves providing regular meals, daily exercise, veterinary care, and plenty of social interaction. Dogs are active animals that need both physical and mental stimulation to stay happy and healthy.",
        ],
        "do": [
            "Yes, dogs are social animals that generally enjoy the company of people and other animals. They are known for their enthusiasm, playfulness, and eagerness to please, which makes them wonderful companions.",
            "Dogs do have a wide range of behaviors depending on their breed and personality. Most dogs enjoy playing, going for walks, and spending time with their owners, and they respond well to positive reinforcement and training.",
        ],
        "default": [
            "A dog is a loyal, friendly domesticated animal that has been a human companion for thousands of years. They come in many breeds and sizes and are known for their intelligence, trainability, and affectionate nature.",
            "Dogs are among the most popular pets in the world, known for their loyalty, warmth, and ability to form strong bonds with people. They are social animals that thrive on companionship and interaction.",
        ],
    },
    "moon": {
        "what": [
            "The moon is Earth's only natural satellite, orbiting our planet at an average distance of about 384,400 kilometers. It is a rocky, airless body covered in craters, and its gravitational pull is responsible for ocean tides on Earth.",
            "The moon is a celestial body that orbits Earth and is visible in the night sky. It has no atmosphere or liquid water, and its surface is covered with craters, mountains, and plains of hardened lava called maria.",
        ],
        "why": [
            "The moon appears to change shape throughout the month because of the way sunlight illuminates its surface as it orbits Earth. These phases cycle from new moon to full moon and back roughly every 29.5 days.",
            "The moon is significant because its gravitational pull creates tides, stabilizes Earth's axial tilt, and has influenced life and culture on our planet for billions of years.",
        ],
        "how": [
            "The moon was likely formed about 4.5 billion years ago when a Mars-sized object collided with the early Earth. The debris from this impact eventually coalesced into the moon we see today.",
            "The moon orbits Earth once approximately every 27.3 days and rotates on its own axis at the same rate, which is why we always see the same side facing us.",
        ],
        "default": [
            "The moon is Earth's natural satellite, a rocky body that orbits our planet and is visible in the night sky. It plays an important role in creating ocean tides and has been a source of fascination for humans throughout history.",
            "The moon is a celestial body that has orbited Earth for billions of years. It has no atmosphere, is covered in craters, and its phases have been used to mark time by cultures around the world.",
        ],
    },
    "bird": {
        "what": [
            "A bird is a warm-blooded vertebrate with feathers, wings, and a beak. Most birds can fly, though some species like penguins and ostriches are flightless. There are over 10,000 known species of birds worldwide.",
            "Birds are a diverse group of animals characterized by their feathers, beaks, and ability to lay eggs. They inhabit every continent and have adapted to a wide range of environments, from tropical forests to arctic tundra.",
        ],
        "do": [
            "Yes, birds exhibit a wide range of fascinating behaviors including complex songs, elaborate courtship displays, and impressive feats of migration. Many species travel thousands of miles each year between breeding and wintering grounds.",
            "Birds do many remarkable things. They build intricate nests, care for their young, communicate through song, and some species can even use tools or mimic human speech.",
        ],
        "where": [
            "Birds can be found on every continent and in nearly every habitat on Earth. Many species migrate seasonally, traveling to warmer regions during winter and returning to breeding grounds in spring.",
            "Birds live in a huge variety of environments including forests, grasslands, deserts, oceans, and cities. During winter, many species migrate to warmer climates, sometimes covering thousands of miles.",
        ],
        "default": [
            "Birds are warm-blooded animals with feathers and beaks that are found all over the world. They are incredibly diverse, ranging from tiny hummingbirds to large eagles, and play important roles in ecosystems as pollinators, seed dispersers, and predators.",
            "A bird is a feathered, winged animal that lays eggs. Birds are among the most diverse groups of animals on Earth, with species adapted to nearly every environment from oceans to mountaintops.",
        ],
    },
    "fish": {
        "what": [
            "A fish is a cold-blooded aquatic animal that breathes through gills and typically has fins and scales. Fish are the most diverse group of vertebrates, with over 34,000 known species living in freshwater and saltwater environments.",
            "Fish are aquatic vertebrates that live in rivers, lakes, and oceans around the world. They breathe using gills, move with fins, and come in an enormous variety of shapes, sizes, and colors.",
        ],
        "do": [
            "Yes, fish are active and social animals. Many species school together for protection, communicate through body language and sound, and exhibit complex behaviors like courtship, territory defense, and parental care.",
            "Fish do many interesting things. They can navigate vast ocean currents, detect electrical fields, change color for camouflage, and some species can even survive out of water for short periods.",
        ],
        "default": [
            "Fish are cold-blooded aquatic animals that breathe through gills and are found in waters all over the world. They are incredibly diverse and play a vital role in aquatic ecosystems and in human food systems.",
            "A fish is an aquatic vertebrate with gills, fins, and typically scales. Fish inhabit nearly every body of water on Earth and are one of the most species-rich groups of animals.",
        ],
    },
    "sky": {
        "what": [
            "The sky is the expanse of air and space visible above the Earth's surface. Its appearance changes throughout the day due to the scattering of sunlight by the atmosphere, creating colors that range from blue during the day to red and orange at sunset.",
            "The sky is what we see when we look upward from the Earth's surface. It appears blue during the day because molecules in the atmosphere scatter shorter blue wavelengths of sunlight more than other colors.",
        ],
        "why": [
            "The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight enters the atmosphere, the shorter blue wavelengths are scattered in all directions by gas molecules more than the longer red wavelengths, making the sky look blue to our eyes.",
            "The sky is blue because sunlight is made up of many colors, and as it passes through the atmosphere, blue light is scattered more than other colors by the tiny molecules of nitrogen and oxygen. This scattered blue light is what we see when we look up.",
        ],
        "default": [
            "The sky appears blue during the day due to the scattering of sunlight by Earth's atmosphere. At sunrise and sunset, it turns shades of red and orange as light travels through more atmosphere, scattering away the blue wavelengths.",
            "The sky is the visible expanse above us, and its blue color comes from the scattering of sunlight by atmospheric molecules. It changes color throughout the day and is also where we see clouds, the sun, the moon, and stars.",
        ],
    },
    "tree": {
        "what": [
            "A tree is a tall perennial plant with a woody trunk, branches, and leaves. Trees play a critical role in ecosystems by producing oxygen, absorbing carbon dioxide, providing habitat for wildlife, and preventing soil erosion.",
            "Trees are large plants with a central trunk that supports branches and leaves. They are found on every continent except Antarctica and are essential for life on Earth, providing oxygen, shade, food, and building materials.",
        ],
        "default": [
            "Trees are perennial plants with woody trunks that can grow to enormous sizes and live for hundreds or even thousands of years. They are vital to Earth's ecosystems, producing the oxygen we breathe and supporting countless species of wildlife.",
            "A tree is a large plant with a sturdy trunk, branches, and a canopy of leaves. Trees are among the longest-living organisms on Earth and play a key role in regulating climate, producing oxygen, and supporting biodiversity.",
        ],
    },
    "world": {
        "what": [
            "The world refers to Earth, the third planet from the Sun and the only known planet that supports life. It has a diverse range of environments including oceans, mountains, forests, and deserts, and is home to billions of living species.",
            "The world is our planet Earth, a rocky body with liquid water, a breathable atmosphere, and a remarkable diversity of life. It has been shaped by billions of years of geological and biological processes.",
        ],
        "default": [
            "The world is a vast and diverse place, home to a wide range of ecosystems, cultures, and species. From deep ocean trenches to towering mountain peaks, Earth supports an extraordinary variety of life and natural phenomena.",
            "Our world is the planet Earth, the only known home of life in the universe. It features oceans, continents, a dynamic atmosphere, and an incredible diversity of plants, animals, and ecosystems.",
        ],
    },
}

# Fallback for prompts with no matching topic.
_GENERIC_RESPONSES = {
    "what": [
        "That is an interesting question. There are many ways to approach it, and the answer depends on the specific context. Could you provide a bit more detail about what you are looking for?",
        "That is a broad topic with many aspects to consider. In general, it involves understanding the relationships between different elements and how they interact with each other.",
    ],
    "why": [
        "That is a great question. The reasons are often complex and involve multiple factors working together. Understanding the underlying causes requires looking at both the immediate triggers and the deeper context.",
        "There are several reasons why this is the case, and they often relate to fundamental principles that govern how things work. The short answer involves a combination of natural processes and conditions.",
    ],
    "how": [
        "That is a thoughtful question. The process typically involves several steps, and the specifics can vary depending on the context. In general, it works through a combination of established principles and conditions.",
        "Understanding how something works often requires looking at the underlying mechanisms. The process involves multiple factors that interact in specific ways to produce the result we observe.",
    ],
    "default": [
        "That is an interesting topic. There are many aspects to consider, and the details can vary depending on the specific context. Feel free to ask a more specific question if you would like to explore a particular angle.",
        "Thank you for your question. This is a broad subject with many facets, and providing a thorough answer would require knowing more about what specific aspect you are most interested in.",
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
