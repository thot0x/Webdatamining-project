"""
analyze_embeddings.py — t-SNE visualisation + nearest-neighbor analysis
of TransE and ComplEx entity embeddings.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict

_HERE      = os.path.dirname(__file__)
PROJECT    = os.path.abspath(os.path.join(_HERE, "..", ".."))
KGE_DIR    = os.path.join(PROJECT, "kge_data")
MODELS_DIR = os.path.join(KGE_DIR, "models")
OUT_DIR    = os.path.join(KGE_DIR, "analysis")


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(model_name: str):
    """
    Charge un modele PyKEEN entraine et retourne :
        embeddings : np.ndarray  (n_entities, dim)
        entity_map : dict        label -> int
    Compatible PyKEEN 1.10 / 1.11.
    """
    import torch

    weights_path = os.path.join(MODELS_DIR, model_name, "trained_model.pkl")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Modele introuvable : {weights_path}")

    model = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.eval()

    with torch.no_grad():
        if hasattr(model, "entity_representations"):
            emb_module = model.entity_representations[0]
            try:
                embeddings = emb_module(indices=None).detach().numpy()
            except TypeError:
                # Certaines versions n'acceptent pas indices=None
                idx = torch.arange(model.num_entities)
                embeddings = emb_module(indices=idx).detach().numpy()
            # ComplEx : partie reelle si complexe
            if np.iscomplexobj(embeddings):
                embeddings = embeddings.real
        else:
            raise AttributeError(f"Impossible d'extraire les embeddings de {model_name}")

    # Mapping entite -> id (PyKEEN 1.11 peut le placer a differents endroits)
    entity_map = {}
    model_dir  = os.path.join(MODELS_DIR, model_name)
    candidates = [
        os.path.join(model_dir, "training_triples", "entity_to_id.tsv"),
        os.path.join(model_dir, "entity_to_id.tsv"),
    ]
    for map_path in candidates:
        if os.path.exists(map_path):
            with open(map_path, encoding="utf-8") as fh:
                next(fh, None)  # sauter en-tete eventuel
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        try:
                            entity_map[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
            break

    # Fallback : labels generiques si mapping absent
    if not entity_map:
        n = embeddings.shape[0]
        entity_map = {f"entity_{i}": i for i in range(n)}

    return embeddings, entity_map


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def compute_tsne(embeddings: np.ndarray, n_components: int = 2, perplexity: float = 30.0):
    from sklearn.manifold import TSNE

    n    = embeddings.shape[0]
    perp = min(perplexity, max(n - 1, 1))

    # max_iter remplace n_iter depuis sklearn 1.2
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(reduced: np.ndarray, labels: List[str], model_name: str, out_dir: str):
    """Sauvegarde un scatter plot t-SNE (necessite matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=8, alpha=0.6, c="steelblue")

    rng   = np.random.default_rng(42)
    shown = rng.choice(len(labels), min(50, len(labels)), replace=False)
    for i in shown:
        lbl = labels[i].split("#")[-1].split("/")[-1][:20]
        plt.annotate(lbl, (reduced[i, 0], reduced[i, 1]), fontsize=6, alpha=0.8)

    plt.title(f"t-SNE de {model_name} ({len(labels)} entites)")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tsne_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] t-SNE plot -> {path}")


# ── Nearest neighbours ────────────────────────────────────────────────────────

def nearest_neighbors_fast(
    embeddings: np.ndarray,
    entity_labels: List[str],
    k: int = 5,
    sample_size: int = 5,
):
    """Retourne les k plus proches voisins (cosine) pour un echantillon d'entites."""
    from sklearn.metrics.pairwise import cosine_similarity

    n   = len(entity_labels)
    rng = np.random.default_rng(42)
    query_ids = rng.choice(n, min(sample_size, n), replace=False)

    results = {}
    for qid in query_ids:
        q    = embeddings[qid : qid + 1]
        sims = cosine_similarity(q, embeddings)[0]
        top  = np.argsort(-sims)[:k + 1]
        neighbors = [
            (entity_labels[i], float(sims[i]))
            for i in top if i != qid
        ][:k]
        results[entity_labels[qid]] = neighbors

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Embedding Analysis -- t-SNE + Nearest Neighbors")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    analysis_report = {}

    for model_name in ["TransE", "ComplEx"]:
        print(f"\n{'─'*60}")
        print(f"  Analysing: {model_name}")
        print(f"{'─'*60}")

        try:
            embeddings, entity_map = extract_embeddings(model_name)
        except (FileNotFoundError, AttributeError) as e:
            print(f"[SKIP] {e}")
            print("[DEMO] Generation d'embeddings aleatoires pour demonstration...")
            n_ent      = 255
            dim        = 100
            rng        = np.random.default_rng(42)
            embeddings = rng.standard_normal((n_ent, dim)).astype(np.float32)
            entity_map = {f"http://example.org/entity_{i}": i for i in range(n_ent)}

        entity_labels = sorted(entity_map, key=lambda x: entity_map[x])
        print(f"  Entites : {len(entity_labels)} | Dim : {embeddings.shape[1]}")

        # t-SNE
        if len(entity_labels) >= 5:
            print("  Calcul t-SNE...")
            try:
                reduced = compute_tsne(embeddings)
                plot_tsne(reduced, entity_labels, model_name, OUT_DIR)
                coords_path = os.path.join(OUT_DIR, f"tsne_{model_name}.npy")
                np.save(coords_path, reduced)
                print(f"[SAVED] Coordonnees t-SNE -> {coords_path}")
            except Exception as exc:
                print(f"[WARN] t-SNE echoue : {exc}")
                reduced = None
        else:
            print("  [SKIP] Pas assez d'entites pour t-SNE (min 5).")
            reduced = None

        # Nearest neighbors
        print("  Calcul des plus proches voisins...")
        try:
            nn_results = nearest_neighbors_fast(embeddings, entity_labels, k=5, sample_size=5)
        except Exception as exc:
            print(f"[WARN] NN echoue : {exc}")
            nn_results = {}

        print("\n  Exemples de plus proches voisins :")
        for query, neighbors in nn_results.items():
            q_label = query.split("#")[-1].split("/")[-1][:30]
            print(f"    Requete : {q_label}")
            for nbr, sim in neighbors:
                n_label = nbr.split("#")[-1].split("/")[-1][:25]
                print(f"      -> {n_label:25s}  sim={sim:.4f}")

        nn_serialisable = {
            q.split("#")[-1]: [(n.split("#")[-1], s) for n, s in nbrs]
            for q, nbrs in nn_results.items()
        }
        analysis_report[model_name] = {
            "n_entities": len(entity_labels),
            "dim":        int(embeddings.shape[1]),
            "nearest_neighbors": nn_serialisable,
        }

    report_path = os.path.join(OUT_DIR, "embedding_analysis.json")
    with open(report_path, "w") as fh:
        json.dump(analysis_report, fh, indent=2)
    print(f"\n[SAVED] Rapport analyse -> {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()