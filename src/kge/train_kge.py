"""
train_kge.py — Train TransE and ComplEx embeddings with PyKEEN
Models: TransE + ComplEx | embedding_dim=100 | epochs=100
"""

import os
import json
import time
from pathlib import Path

_HERE    = os.path.dirname(__file__)
PROJECT  = os.path.abspath(os.path.join(_HERE, "..", ".."))
KGE_DIR  = os.path.join(PROJECT, "kge_data")
OUT_DIR  = os.path.join(PROJECT, "kge_data", "models")

EMBEDDING_DIM  = 100
NUM_EPOCHS     = 100
BATCH_SIZE     = 256
LEARNING_RATE  = 0.001
RANDOM_SEED    = 42

MODELS_CFG = [
    {
        "name":  "TransE",
        "class": "TransE",
        "kwargs": {
            "embedding_dim": EMBEDDING_DIM,
            "scoring_fct_norm": 1,
        },
    },
    {
        "name":  "ComplEx",
        "class": "ComplEx",
        "kwargs": {
            "embedding_dim": EMBEDDING_DIM,
        },
    },
]


def check_splits():
    """Make sure split files exist, run prepare_splits if needed."""
    train_path = os.path.join(KGE_DIR, "train.txt")
    if not os.path.exists(train_path):
        print("[INFO] Split files not found. Running prepare_splits.py first...")
        import sys
        sys.path.insert(0, _HERE)
        from prepare_splits import run as prepare_run
        prepare_run()


def load_dataset_pykeen():
    """Load the TXT splits into a PyKEEN TriplesFactory."""
    from pykeen.triples import TriplesFactory

    train_path = os.path.join(KGE_DIR, "train.txt")
    valid_path = os.path.join(KGE_DIR, "valid.txt")
    test_path  = os.path.join(KGE_DIR, "test.txt")

    training   = TriplesFactory.from_path(train_path, delimiter="\t")
    validation = TriplesFactory.from_path(
        valid_path, delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing    = TriplesFactory.from_path(
        test_path, delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    return training, validation, testing


def train_model(model_cfg: dict, training, validation, testing):
    """Train a single model and return the pipeline results."""
    from pykeen.pipeline import pipeline
    import torch

    print(f"\n{'─'*60}")
    print(f"  Training {model_cfg['name']}  "
          f"(dim={EMBEDDING_DIM}, epochs={NUM_EPOCHS})")
    print(f"{'─'*60}")

    t0 = time.time()

    result = pipeline(
        # Data
        training=training,
        validation=validation,
        testing=testing,
        # Model
        model=model_cfg["class"],
        model_kwargs=model_cfg["kwargs"],
        # Training
        optimizer="Adam",
        optimizer_kwargs={"lr": LEARNING_RATE},
        training_loop="sLCWA",
        training_kwargs={"num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE},
        # Evaluation
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={"filtered": True},
        # Reproducibility
        random_seed=RANDOM_SEED,
        device="cpu",  # change to "cuda" if GPU available
    )

    elapsed = time.time() - t0
    print(f"[{model_cfg['name']}] Training time: {elapsed:.1f}s")

    return result


def save_model(result, model_name: str):
    """Save model weights and metadata."""
    model_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save via PyKEEN's built-in serialization
    result.save_to_directory(model_dir)

    # Save a JSON summary of metrics
    metrics = result.metric_results.to_dict() if result.metric_results else {}
    meta = {
        "model":         model_name,
        "embedding_dim": EMBEDDING_DIM,
        "epochs":        NUM_EPOCHS,
        "metrics":       metrics,
    }
    with open(os.path.join(model_dir, "training_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SAVED] {model_name} → {model_dir}")
    return model_dir


def run():
    print("=" * 60)
    print("KGE Training — TransE + ComplEx (PyKEEN)")
    print("=" * 60)

    check_splits()

    try:
        import pykeen
        try:
            version = pykeen.get_version()
        except Exception:
            try:
                from importlib.metadata import version
                version = version("pykeen")
            except Exception:
                version = "unknown"
        print(f"[INFO] PyKEEN version: {version}")
    except ImportError:
        print("[ERROR] PyKEEN not installed. Run: pip install pykeen")
        return

    print("[INFO] Loading dataset splits...")
    training, validation, testing = load_dataset_pykeen()
    print(
        f"[INFO] Entities: {training.num_entities} | "
        f"Relations: {training.num_relations} | "
        f"Train: {training.num_triples} | "
        f"Valid: {validation.num_triples} | "
        f"Test: {testing.num_triples}"
    )

    results = {}
    for cfg in MODELS_CFG:
        result      = train_model(cfg, training, validation, testing)
        model_dir   = save_model(result, cfg["name"])
        results[cfg["name"]] = {
            "model_dir": model_dir,
            "result":    result,
        }

    # Print final metrics summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for name, info in results.items():
        r = info["result"]
        try:
            mrr    = r.get_metric("hits@10")
            h10    = r.get_metric("hits@10")
            print(f"  {name:12s} | Hits@10 = {h10:.4f}")
        except Exception:
            print(f"  {name:12s} | (metrics unavailable)")

    print("\n[DONE] Models saved to:", OUT_DIR)
    print("=" * 60)
    return results


if __name__ == "__main__":
    run()