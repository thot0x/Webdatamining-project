"""
prepare_splits.py — Clean expanded_kb.ttl and split 80/10/10 for KGE training.

Steps:
  1. Load expanded_kb.ttl (Turtle format, ex: namespace)
  2. Remove duplicates and triples with literal objects (keep URI-URI-URI)
  3. Ensure no entity in valid/test is isolated (not seen in train)
  4. Write kge_data/train.txt, valid.txt, test.txt in TSV format (s\tp\to)
"""

import os
import random
import csv
from typing import List, Tuple, Set

try:
    from rdflib import Graph, Literal
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

_HERE   = os.path.dirname(__file__)
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
KGE_DIR = os.path.join(PROJECT, "kge_data")

# Cherche expanded_kb.ttl en priorité, fallback sur .nt
_KB_TTL = os.path.join(PROJECT, "kg_artifacts", "expanded_kb.ttl")
_KB_NT  = os.path.join(PROJECT, "kg_artifacts", "expanded_kb.nt")
KB_PATH = _KB_TTL if os.path.exists(_KB_TTL) else _KB_NT

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

Triple = Tuple[str, str, str]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_rdflib(path: str) -> List[Triple]:
    """
    Charge un fichier RDF (Turtle ou N-Triples) via rdflib.
    Détecte le format automatiquement selon l'extension.
    Filtre les triplets avec objet littéral.
    """
    g      = Graph()
    ext    = os.path.splitext(path)[1].lower()
    fmt    = "turtle" if ext in (".ttl", ".turtle") else "nt"
    g.parse(path, format=fmt)

    triples = []
    for s, p, o in g:
        if isinstance(o, Literal):
            continue   # on garde uniquement URI–URI–URI
        triples.append((str(s), str(p), str(o)))
    return triples


def _guess_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "turtle" if ext in (".ttl", ".turtle") else "nt"


def load_triples(path: str) -> List[Triple]:
    if HAS_RDFLIB:
        return load_rdflib(path)
    raise RuntimeError("rdflib est requis pour charger un fichier Turtle. "
                       "Installe-le avec: pip install rdflib")


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_triples(triples: List[Triple]) -> List[Triple]:
    """Déduplique et garde uniquement les triplets URI–URI–URI."""
    seen: Set[Triple] = set()
    cleaned = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned


# ── Entity coverage ───────────────────────────────────────────────────────────

def entities_of(triples: List[Triple]) -> Set[str]:
    ents = set()
    for s, p, o in triples:
        ents.add(s)
        ents.add(o)
    return ents


def fix_isolated(
    train: List[Triple],
    valid: List[Triple],
    test:  List[Triple],
) -> Tuple[List[Triple], List[Triple], List[Triple]]:
    """
    Déplace dans train tout triplet de valid/test dont les entités
    n'apparaissent pas dans train (évite les entités isolées).
    """
    train_ents = entities_of(train)

    def covered(t):
        s, _, o = t
        return s in train_ents and o in train_ents

    new_valid, rescued_v = [], []
    for t in valid:
        (new_valid if covered(t) else rescued_v).append(t)

    new_test, rescued_t = [], []
    for t in test:
        (new_test if covered(t) else rescued_t).append(t)

    rescued = rescued_v + rescued_t
    if rescued:
        print(f"[FIX] {len(rescued)} triplet(s) déplacé(s) vers train (entités isolées).")
        train = train + rescued

    return train, new_valid, new_test


# ── Writing ───────────────────────────────────────────────────────────────────

def write_split(triples: List[Triple], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        for s, p, o in triples:
            writer.writerow([s, p, o])


# ── Demo fallback ─────────────────────────────────────────────────────────────

def _generate_demo_triples(n: int = 2000) -> List[Triple]:
    """Triplets synthétiques football pour démo si KB absente."""
    BASE    = "http://example.org/"
    players = [f"Player_{i}" for i in range(200)]
    clubs   = [f"Club_{i}"   for i in range(20)]
    leagues = [f"League_{i}" for i in range(5)]
    nations = [f"Country_{i}" for i in range(30)]
    preds   = ["playsFor", "hasNationality", "competesIn", "rivals"]

    triples = set()
    rng = random.Random(42)
    for p in players:
        triples.add((BASE + p, BASE + "playsFor",       BASE + rng.choice(clubs)))
        triples.add((BASE + p, BASE + "hasNationality", BASE + rng.choice(nations)))
        triples.add((BASE + rng.choice(clubs), BASE + "competesIn", BASE + rng.choice(leagues)))
    while len(triples) < n:
        triples.add((BASE + rng.choice(players), BASE + rng.choice(preds),
                     BASE + rng.choice(clubs + players)))
    return list(triples)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("KGE — Prepare Splits (prepare_splits.py)")
    print("=" * 60)

    if not os.path.exists(KB_PATH):
        print(f"[ERROR] Aucun fichier KB trouvé dans kg_artifacts/")
        print(f"        (cherché: expanded_kb.ttl puis expanded_kb.nt)")
        print("[INFO]  Utilisation de triplets synthétiques de démo.")
        triples = _generate_demo_triples()
    else:
        fmt = _guess_format(KB_PATH)
        print(f"[INFO] Chargement {KB_PATH} (format: {fmt}) ...")
        triples = load_triples(KB_PATH)
        print(f"[INFO] {len(triples)} triplets bruts chargés.")

    # Nettoyage
    triples = clean_triples(triples)
    print(f"[CLEAN] {len(triples)} triplets après déduplication / suppression littéraux.")

    # Mélange
    random.seed(RANDOM_SEED)
    random.shuffle(triples)

    # Split
    n       = len(triples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train = triples[:n_train]
    valid = triples[n_train : n_train + n_valid]
    test  = triples[n_train + n_valid :]

    print(f"[SPLIT] train={len(train)} | valid={len(valid)} | test={len(test)}")

    train, valid, test = fix_isolated(train, valid, test)
    print(f"[SPLIT final] train={len(train)} | valid={len(valid)} | test={len(test)}")

    # Écriture
    train_path = os.path.join(KGE_DIR, "train.txt")
    valid_path = os.path.join(KGE_DIR, "valid.txt")
    test_path  = os.path.join(KGE_DIR, "test.txt")

    write_split(train, train_path)
    write_split(valid, valid_path)
    write_split(test,  test_path)

    print(f"[SAVED] {train_path}")
    print(f"[SAVED] {valid_path}")
    print(f"[SAVED] {test_path}")

    # Stats
    all_ents = entities_of(triples)
    all_rels = {p for _, p, _ in triples}
    print(f"\n[STATS]")
    print(f"  Entités totales   : {len(all_ents)}")
    print(f"  Relations totales : {len(all_rels)}")
    print(f"  Triplets totaux   : {len(triples)}")
    print("=" * 60)

    return train_path, valid_path, test_path


if __name__ == "__main__":
    run()