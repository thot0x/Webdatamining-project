"""
analyze_embeddings.py — t-SNE visualisation + nearest-neighbor analysis
of TransE and ComplEx entity embeddings.
"""
 
import os
import json
import numpy as np
from typing import List, Tuple, Dict
 
_HERE   = os.path.dirname(__file__)
PROJECT = os.path.abspath(os.path.join(_HERE, "..", ".."))
KGE_DIR = os.path.join(PROJECT, "kge_data")
MODELS_DIR = os.path.join(KGE_DIR, "models")
OUT_DIR    = os.path.join(KGE_DIR, "analysis")
 
 
#Embedding extraction
 
def extract_embeddings(model_name: str):
    """
    Load a trained PyKEEN model and return:
        embeddings : np.ndarray  shape (n_entities, dim)
        entity_ids : dict        entity_label → int
    """
    import torch
 
    weights_path = os.path.join(MODELS_DIR, model_name, "trained_model.pkl")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")
 
    model = torch.load(weights_path, map_location="cpu")
    model.eval()
 
    with torch.no_grad():
        # Works for TransE and ComplEx (real part for ComplEx)
        if hasattr(model, "entity_representations"):
            emb_module = model.entity_representations[0]
            embeddings = emb_module(indices=None).detach().numpy()
            # ComplEx returns complex: take real part
            if np.iscomplexobj(embeddings):
                embeddings = embeddings.real
        else:
            raise AttributeError(f"Cannot extract embeddings from {model_name}")
 
    # Load entity id mapping
    meta_path = os.path.join(MODELS_DIR, model_name)
    entity_map = {}
    map_path = os.path.join(meta_path, "training_triples", "entity_to_id.tsv")
    if os.path.exists(map_path):
        with open(map_path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    entity_map[parts[0]] = int(parts[1])
 
    return embeddings, entity_map
 
 
#t-SNE
 
def compute_tsne(embeddings: np.ndarray, n_components: int = 2, perplexity: float = 30.0):
    from sklearn.manifold import TSNE
 
    n = embeddings.shape[0]
    perp = min(perplexity, n - 1)
 
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=42,
        n_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)
 
 
def plot_tsne(reduced: np.ndarray, labels: List[str], model_name: str, out_dir: str):
    """Save a t-SNE scatter plot (requires matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    plt.figure(figsize=(14, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=8, alpha=0.6, c="steelblue")
 
    # Annotate a sample of entities
    rng   = np.random.default_rng(42)
    shown = rng.choice(len(labels), min(50, len(labels)), replace=False)
    for i in shown:
        lbl = labels[i].split("#")[-1].split("/")[-1][:20]
        plt.annotate(lbl, (reduced[i, 0], reduced[i, 1]),
                     fontsize=6, alpha=0.8)
 
    plt.title(f"t-SNE of {model_name} entity embeddings ({len(labels)} entities)")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.tight_layout()
 
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tsne_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] t-SNE plot → {path}")
 
 
# Nearest neighbours 
 
def nearest_neighbors(
    embeddings: np.ndarray,
    entity_map: Dict[str, int],
    query_entity: str,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Return the k nearest entities (cosine similarity) to query_entity."""
    from sklearn.metrics.pairwise import cosine_similarity
 
    id2entity = {v: k for k, v in entity_map.items()}
 
    if query_entity not in entity_map:
        # Fuzzy match on suffix
        matches = [e for e in entity_map if query_entity.lower() in e.lower()]
        if not matches:
            raise KeyError(f"Entity '{query_entity}' not found in embedding space.")
        query_entity = matches[0]
        print(f"[INFO] Using entity: {query_entity}")
 
    qid = entity_map[query_entity]
    q   = embeddings[qid : qid + 1]
    sims = cosine_similarity(q, embeddings)[0]   # shape (n,)
 
    top_idx = np.argsort(-sims)[:k + 1]          # +1 to exclude self
    neighbors = []
    for idx in top_idx:
        if idx == qid:
            continue
        neighbors.append((id2entity.get(idx, str(idx)), float(sims[idx])))
        if len(neighbors) == k:
            break
 
    return neighbors
 
 
def nearest_neighbors_fast(
    embeddings: np.ndarray,
    entity_labels: List[str],
    k: int = 10,
    sample_size: int = 5,
):
    """Return nearest neighbors for a random sample of entities."""
    from sklearn.metrics.pairwise import cosine_similarity
 
    n = len(entity_labels)
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
 
 
#Main
 
def run():
    print("=" * 60)
    print("Embedding Analysis — t-SNE + Nearest Neighbors")
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
            # Demo mode with random embeddings
            print("[DEMO] Generating random embeddings for demonstration...")
            n_ent    = 500
            dim      = 100
            rng      = np.random.default_rng(42)
            embeddings = rng.standard_normal((n_ent, dim)).astype(np.float32)
            entity_map = {f"http://example.org/football#Entity_{i}": i for i in range(n_ent)}
 
        entity_labels = sorted(entity_map, key=lambda x: entity_map[x])
 
        print(f"  Entities: {len(entity_labels)} | Dim: {embeddings.shape[1]}")
 
        # t-SNE
        print("  Computing t-SNE (this may take a moment)...")
        try:
            reduced = compute_tsne(embeddings)
            plot_tsne(reduced, entity_labels, model_name, OUT_DIR)
        except ImportError:
            print("[WARN] matplotlib/sklearn not available — skipping plots.")
            reduced = None
 
        # Save 2-D coordinates
        if reduced is not None:
            coords_path = os.path.join(OUT_DIR, f"tsne_{model_name}.npy")
            np.save(coords_path, reduced)
            print(f"[SAVED] t-SNE coordinates → {coords_path}")
 
        # Nearest neighbors
        print("  Computing nearest neighbors (sample)...")
        try:
            nn_results = nearest_neighbors_fast(
                embeddings, entity_labels, k=5, sample_size=5
            )
        except Exception as exc:
            print(f"[WARN] NN failed: {exc}")
            nn_results = {}
 
        print("\n  Nearest-neighbor examples:")
        for query, neighbors in nn_results.items():
            q_label = query.split("#")[-1].split("/")[-1][:30]
            print(f"    Query: {q_label}")
            for nbr, sim in neighbors:
                n_label = nbr.split("#")[-1].split("/")[-1][:25]
                print(f"      → {n_label:25s}  sim={sim:.4f}")
 
        # Save NN results
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
    print(f"\n[SAVED] Analysis report → {report_path}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    run()