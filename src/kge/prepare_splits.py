"""
prepare_splits.py — Clean expanded_kb.nt and split 80/10/10 for KGE training.
 
Steps:
  1. Load expanded_kb.nt
  2. Remove duplicates and triples with literal objects (keep URI-URI-URI triples)
  3. Ensure no entity in valid/test is isolated (appears only there, not in train)
  4. Write kge_data/train.txt, valid.txt, test.txt in TSV format (s\tp\to)
"""
 
import os
import random
import csv
from collections import defaultdict
from typing import List, Tuple, Set
 
try:
    from rdflib import Graph, URIRef, Literal
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
 
_HERE      = os.path.dirname(__file__)
PROJECT    = os.path.abspath(os.path.join(_HERE, "..", ".."))
KB_PATH    = os.path.join(PROJECT, "kg_artifacts", "expanded_kb.nt")
KGE_DIR    = os.path.join(PROJECT, "kge_data")
 
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42
 
Triple = Tuple[str, str, str]
 
 
#Loading
 
def load_nt_rdflib(path: str) -> List[Triple]:
    """Load N-Triples via rdflib, return (s, p, o) URI strings."""
    g = Graph()
    g.parse(path, format="nt")
    triples = []
    for s, p, o in g:
        if isinstance(o, Literal):
            continue           # skip literal objects
        triples.append((str(s), str(p), str(o)))
    return triples
 
 
def load_nt_manual(path: str) -> List[Triple]:
    """Fallback N-Triples parser (no rdflib needed)."""
    triples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(" ."):
                line = line[:-2].strip()
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            s, p, o = parts
            # Skip if any part is a literal (starts with " or contains ^^)
            if o.startswith('"') or "^^" in o:
                continue
            # Strip angle brackets
            s = s.strip("<>")
            p = p.strip("<>")
            o = o.strip("<>")
            if s and p and o:
                triples.append((s, p, o))
    return triples
 
 
def load_triples(path: str) -> List[Triple]:
    if HAS_RDFLIB:
        return load_nt_rdflib(path)
    return load_nt_manual(path)
 
 
#Cleaning
 
def clean_triples(triples: List[Triple]) -> List[Triple]:
    """Remove duplicates; keep only URI–URI–URI triples."""
    seen: Set[Triple] = set()
    cleaned = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned
 
 
#Entity coverage check
 
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
    Move any triple from valid/test whose entities are unseen in train
    back into train.
    """
    train_ents = entities_of(train)
 
    def is_covered(t):
        s, _, o = t
        return s in train_ents and o in train_ents
 
    new_valid, rescued_v = [], []
    for t in valid:
        (new_valid if is_covered(t) else rescued_v).append(t)
 
    new_test, rescued_t = [], []
    for t in test:
        (new_test if is_covered(t) else rescued_t).append(t)
 
    rescued = rescued_v + rescued_t
    if rescued:
        print(f"[FIX] Moved {len(rescued)} triple(s) back to train to avoid entity isolation.")
        train = train + rescued
 
    return train, new_valid, new_test
 
 
#Writing 
 
def write_split(triples: List[Triple], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        for s, p, o in triples:
            writer.writerow([s, p, o])
 
 
# Main
 
def run():
    print("=" * 60)
    print("KGE — Prepare Splits (prepare_splits.py)")
    print("=" * 60)
 
    # Load
    if not os.path.exists(KB_PATH):
        print(f"[ERROR] expanded_kb.nt not found at {KB_PATH}")
        print("[INFO]  Using synthetic demo triples instead.")
        triples = _generate_demo_triples()
    else:
        print(f"[INFO] Loading {KB_PATH} ...")
        triples = load_triples(KB_PATH)
        print(f"[INFO] Loaded {len(triples)} raw triples.")
 
    # Clean
    triples = clean_triples(triples)
    print(f"[CLEAN] {len(triples)} triples after deduplication / literal removal.")
 
    # Shuffle
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
 
    # Fix entity isolation
    train, valid, test = fix_isolated(train, valid, test)
 
    print(f"[SPLIT (final)] train={len(train)} | valid={len(valid)} | test={len(test)}")
 
    # Write
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
    print(f"  Total entities  : {len(all_ents)}")
    print(f"  Total relations : {len(all_rels)}")
    print(f"  Total triples   : {len(triples)}")
    print("=" * 60)
 
    return train_path, valid_path, test_path
 
 
def _generate_demo_triples(n: int = 2000) -> List[Triple]:
    """Generate synthetic football triples for demo/testing purposes."""
    import math
    BASE = "http://example.org/football#"
    players = [f"Player_{i}" for i in range(200)]
    clubs   = [f"Club_{i}"   for i in range(20)]
    leagues = [f"League_{i}" for i in range(5)]
    nations = [f"Country_{i}" for i in range(30)]
    preds   = ["playsFor", "hasNationality", "competesIn", "partnerClub", "rivals"]
 
    triples = set()
    rng = random.Random(42)
 
    for p in players:
        club    = rng.choice(clubs)
        nation  = rng.choice(nations)
        league  = rng.choice(leagues)
        triples.add((BASE + p, BASE + "playsFor",      BASE + club))
        triples.add((BASE + p, BASE + "hasNationality",BASE + nation))
        triples.add((BASE + club, BASE + "competesIn", BASE + league))
 
    # Extra random triples
    while len(triples) < n:
        s = BASE + rng.choice(players)
        p = BASE + rng.choice(preds)
        o = BASE + rng.choice(clubs + players)
        triples.add((s, p, o))
 
    return list(triples)
 
 
if __name__ == "__main__":
    run()