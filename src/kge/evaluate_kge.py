"""
evaluate_kge.py — Evaluate TransE and ComplEx models.
Metrics: MRR, Hits@1, Hits@3, Hits@10
Tested on: 20k / 50k / full dataset subsets
"""

import os
import json
import time
from typing import Dict, List

_HERE   = os.path.dirname(__file__)
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
KGE_DIR = os.path.join(PROJECT, "kge_data")
MODELS_DIR = os.path.join(KGE_DIR, "models")

EVAL_SUBSETS = [20_000, 50_000, None]   # None = full dataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_triples(path: str):
    """Load tab-separated (s, p, o) triples."""
    import csv
    triples = []
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) == 3:
                triples.append(tuple(row))
    return triples


def load_all_splits():
    """Return training, validation, testing TriplesFactory objects."""
    from pykeen.triples import TriplesFactory

    train_p = os.path.join(KGE_DIR, "train.txt")
    valid_p = os.path.join(KGE_DIR, "valid.txt")
    test_p  = os.path.join(KGE_DIR, "test.txt")

    training   = TriplesFactory.from_path(train_p, delimiter="\t")
    validation = TriplesFactory.from_path(
        valid_p, delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing    = TriplesFactory.from_path(
        test_p, delimiter="\t",
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    return training, validation, testing


def load_model(model_name: str, training):
    """Load a trained model from disk."""
    import torch
    from pykeen.models import model_resolver

    model_dir = os.path.join(MODELS_DIR, model_name)
    weights   = os.path.join(model_dir, "trained_model.pkl")

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            "Run train_kge.py first."
        )
    model = torch.load(weights, map_location="cpu", weights_only=False)
    return model


def evaluate_model(model, testing_factory, n_triples=None):
    """
    Run rank-based evaluation.
    n_triples: None = full, or an integer subset.
    """
    from pykeen.evaluation import RankBasedEvaluator
    import torch

    if n_triples is not None and n_triples < testing_factory.num_triples:
        # Sample a subset
        import numpy as np
        idx   = np.random.choice(testing_factory.num_triples, n_triples, replace=False)
        mapped = testing_factory.mapped_triples[idx]

        from pykeen.triples import TriplesFactory
        sub_factory = TriplesFactory(
            mapped_triples=mapped,
            entity_to_id=testing_factory.entity_to_id,
            relation_to_id=testing_factory.relation_to_id,
        )
        factory_to_eval = sub_factory
    else:
        factory_to_eval = testing_factory
        n_triples = testing_factory.num_triples

    evaluator = RankBasedEvaluator(filtered=True)
    results   = evaluator.evaluate(
        model=model,
        mapped_triples=factory_to_eval.mapped_triples,
        additional_filter_triples=None,
        batch_size=256,
    )
    return results, n_triples


def format_metrics(results) -> Dict[str, float]:
    """Extract key metrics from RankBasedMetricResults."""
    metrics = {}
    try:
        metrics["MRR"]     = float(results.get_metric("mean_reciprocal_rank"))
        metrics["Hits@1"]  = float(results.get_metric("hits_at_1"))
        metrics["Hits@3"]  = float(results.get_metric("hits_at_3"))
        metrics["Hits@10"] = float(results.get_metric("hits_at_10"))
    except Exception as e:
        print(f"[WARN] Could not extract some metrics: {e}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("KGE Evaluation — MRR, Hits@1/3/10")
    print("=" * 60)

    try:
        import pykeen
    except ImportError:
        print("[ERROR] PyKEEN not installed. Run: pip install pykeen")
        return

    print("[INFO] Loading splits...")
    training, validation, testing = load_all_splits()
    print(
        f"[INFO] Test set: {testing.num_triples} triples | "
        f"Entities: {testing.num_entities} | "
        f"Relations: {testing.num_relations}"
    )

    all_results = {}

    for model_name in ["TransE", "ComplEx"]:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        try:
            model = load_model(model_name, training)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        model_results = {}
        for n in EVAL_SUBSETS:
            label = "full" if n is None else f"{n//1000}k"
            print(f"\n  Evaluating on subset: {label}...")
            t0 = time.time()

            try:
                results, actual_n = evaluate_model(model, testing, n_triples=n)
                metrics = format_metrics(results)
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                metrics = {}
                actual_n = n or testing.num_triples

            elapsed = time.time() - t0
            metrics["n_triples"] = actual_n
            metrics["time_s"]    = round(elapsed, 2)
            model_results[label] = metrics

            if metrics:
                print(
                    f"  {label:6s} | "
                    f"MRR={metrics.get('MRR',0):.4f} | "
                    f"H@1={metrics.get('Hits@1',0):.4f} | "
                    f"H@3={metrics.get('Hits@3',0):.4f} | "
                    f"H@10={metrics.get('Hits@10',0):.4f} | "
                    f"({elapsed:.1f}s)"
                )

        all_results[model_name] = model_results

    # Save results
    results_path = os.path.join(KGE_DIR, "evaluation_results.json")
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[SAVED] Evaluation results → {results_path}")

    # Final comparison table
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY (full dataset)")
    print("=" * 60)
    print(f"{'Model':12s} | {'MRR':>7} | {'Hits@1':>7} | {'Hits@3':>7} | {'Hits@10':>8}")
    print("-" * 60)
    for model_name, subsets in all_results.items():
        m = subsets.get("full", subsets.get(list(subsets.keys())[-1], {}))
        print(
            f"{model_name:12s} | "
            f"{m.get('MRR',0):7.4f} | "
            f"{m.get('Hits@1',0):7.4f} | "
            f"{m.get('Hits@3',0):7.4f} | "
            f"{m.get('Hits@10',0):8.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    run()