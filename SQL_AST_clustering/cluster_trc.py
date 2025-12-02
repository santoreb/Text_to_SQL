"""
Cluster 'ast_trc' strings from a Spider JSONL using HDBSCAN.

Usage:
  python3 code/cluster_trc.py --in train_with_trc.jsonl --out_dir clusters_out \
      --embed_model thenlper/gte-large --min_cluster_size 5 --min_samples 1

Outputs:
 - <out_dir>/clustered.jsonl    (original records with added "cluster" int field)
 - <out_dir>/cluster_reps.json  (one representative example per cluster)
"""
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize as sk_normalize
from hdbscan import approximate_predict, flat
from sklearn.cluster import KMeans, AgglomerativeClustering
import pickle

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def embed_texts(model, texts, batch_size=64, normalize=True):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i:i+batch_size]
        embs.append(model.encode(batch, show_progress_bar=False, convert_to_numpy=True))
    embs = np.vstack(embs)
    if normalize:
        embs = sk_normalize(embs)
    return embs

def find_representatives(embs, labels):
    reps = {}
    for lbl in sorted(set(labels)):
        if lbl < 0:
            continue
        idxs = np.where(labels == lbl)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) == 1:
            reps[int(lbl)] = int(idxs[0])
            continue
        sub = embs[idxs]
        # use cosine distance and pick medoid (min total distance)
        d = cosine_distances(sub)
        sumd = d.sum(axis=1)
        medoid_idx = idxs[int(np.argmin(sumd))]
        reps[int(lbl)] = int(medoid_idx)
    return reps

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=False, help="Input JSONL with ast_trc field (optional when only predicting)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--embed_model", default="thenlper/gte-large")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_cluster_size", type=int, default=5)
    p.add_argument("--min_samples", type=int, default=1)
    p.add_argument("--method", type=str, default="hdbscan", choices=["hdbscan","kmeans","agglomerative"], help="Clustering method to use: 'hdbscan' (default) or fixed-size 'kmeans'/'agglomerative'")
    p.add_argument("--n_clusters", type=int, default=None, help="When using kmeans or agglomerative, the number of clusters to produce")
    p.add_argument("--predict", type=str, default=None, help="Path to newline TRC file to predict (one TRC per line)")
    p.add_argument("--pred_out", type=str, default=None, help="Output jsonl path for predictions (if omitted writes to <out_dir>/predictions.jsonl)")
    p.add_argument("--save_embeddings", action="store_true", help="Save computed embeddings to disk under --out_dir to avoid recomputing")
    p.add_argument("--embeddings_path", type=str, default=None, help="Optional explicit path to load/save embeddings (.npz)")
    p.add_argument("--force_recompute_embeddings", action="store_true", help="Force recomputing embeddings even if saved file exists")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model early so we can predict even if no training is requested
    model = SentenceTransformer(args.embed_model)

    rows = []
    texts = []
    idx_map = []
    missing = 0

    # If an input dataset is provided, read and prepare texts for training
    if args.in_path:
        rows = read_jsonl(args.in_path)
        for i, r in enumerate(rows):
            trc = r.get("ast_trc") or r.get("trc") or r.get("ast_trc_tokens")
            if not trc or trc in ["UNHANDLED_SQL", "TRC_ERROR"]:
                missing += 1
                continue
            texts.append(trc)
            idx_map.append(i)
        if not texts:
            raise SystemExit("No ast_trc texts found in input.")

        print(f"Found {len(texts)} trc texts (skipped {missing} examples without ast_trc).")

        # normalize embeddings so Euclidean distance ~= Cosine similarity
        # Attempt to load precomputed embeddings if present to avoid recomputation
        if args.embeddings_path:
            emb_path = args.embeddings_path
        else:
            base = os.path.splitext(os.path.basename(args.in_path))[0]
            model_tag = args.embed_model.replace("/", "_").replace(":", "_")
            emb_path = os.path.join(args.out_dir, f"embs_{base}_{model_tag}.npz")

        embs = None
        if os.path.exists(emb_path) and not args.force_recompute_embeddings:
            try:
                npz = np.load(emb_path, allow_pickle=True)
                loaded = npz.get("embs")
                if loaded is not None and loaded.shape[0] == len(texts):
                    embs = loaded
                    print(f"Loaded embeddings from {emb_path}")
                else:
                    print(f"Embeddings file found but size mismatch ({None if loaded is None else loaded.shape[0]} != {len(texts)}). Recomputing.")
                    embs = None
            except Exception as e:
                print(f"Failed to load embeddings from {emb_path}: {e}. Recomputing.")
                embs = None

        if embs is None:
            embs = embed_texts(model, texts, batch_size=args.batch_size, normalize=True)
            if args.save_embeddings:
                try:
                    np.savez_compressed(emb_path, embs=embs, texts=np.array(texts, dtype=object))
                    print(f"Saved embeddings to {emb_path}")
                except Exception as e:
                    print(f"Failed to save embeddings to {emb_path}: {e}")
    else:
        embs = None

    clusterer = None
    labels = None
    if embs is not None:
        if args.method == "hdbscan":
            if args.n_clusters:
                # use HDBSCAN flat algorithm to force a target number of clusters
                clusterer = flat.HDBSCAN_flat(embs, args.n_clusters)
                probs = getattr(clusterer, "probabilities_", None)
                labels = clusterer.labels_
                n_clusters = args.n_clusters
                print(f"HDBSCAN flat produced {n_clusters} clusters")
            else:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                            min_samples=args.min_samples,
                                            metric="euclidean",
                                            cluster_selection_method="eom")
                labels = clusterer.fit_predict(embs)
                probs = getattr(clusterer, "probabilities_", None)
                if probs is None:
                    probs = None
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"HDBSCAN clusters found (excluding noise -1): {n_clusters}")
        elif args.method == "kmeans":
            if not args.n_clusters:
                raise SystemExit("--n_clusters required when --method=kmeans")
            km = KMeans(n_clusters=args.n_clusters, random_state=0)
            labels = km.fit_predict(embs)
            clusterer = km
            probs = np.ones_like(labels, dtype=float)
            n_clusters = args.n_clusters
            print(f"KMeans produced {n_clusters} clusters")
        elif args.method == "agglomerative":
            if not args.n_clusters:
                raise SystemExit("--n_clusters required when --method=agglomerative")
            ag = AgglomerativeClustering(n_clusters=args.n_clusters)
            labels = ag.fit_predict(embs)
            clusterer = ag
            probs = np.ones_like(labels, dtype=float)
            n_clusters = args.n_clusters
            print(f"AgglomerativeClustering produced {n_clusters} clusters")

        # compute medoids
        reps = find_representatives(embs, labels)
        medoid_labels = []
        medoid_idxs = []
        medoid_embs = []
        for lbl, idx_in_texts in reps.items():
            medoid_labels.append(int(lbl))
            medoid_idxs.append(int(idx_in_texts))
            medoid_embs.append(embs[int(idx_in_texts)])
        medoid_embs = np.vstack(medoid_embs) if medoid_embs else np.zeros((0, embs.shape[1]))

        if probs is not None:
            strengths = probs
        else:
            strengths = np.zeros(len(labels), dtype=float)
            if medoid_embs.size > 0:
                sims = embs @ medoid_embs.T
                # map label -> medoid position
                label_to_medoid_pos = {int(l): i for i, l in enumerate(medoid_labels)}
                for i, L in enumerate(labels):
                    if int(L) in label_to_medoid_pos:
                        pos = label_to_medoid_pos[int(L)]
                        strengths[i] = float(sims[i, pos])
                    else:
                        strengths[i] = 0.0

        # attach labels and strengths back to original rows
        for i, lbl in enumerate(labels):
            orig_idx = idx_map[i]
            rows[orig_idx]["cluster"] = int(lbl)
            rows[orig_idx]["cluster_prob"] = float(strengths[i])

        # write augmented clustered dataset
        out_jsonl = os.path.join(args.out_dir, os.path.basename(args.in_path).replace(".jsonl", "_clustered.jsonl"))
        write_jsonl(out_jsonl, rows)
        print("Wrote clustered jsonl:", out_jsonl)

        # Also write one file per cluster under <out_dir>/clusters/
        try:
            clusters_dir = os.path.join(args.out_dir, "clusters")
            os.makedirs(clusters_dir, exist_ok=True)
            # build mapping label -> list of original row indices
            unique_labels = sorted(set(labels))
            cluster_counts = {}
            for lbl in unique_labels:
                idxs = [idx_map[i] for i, L in enumerate(labels) if int(L) == int(lbl)]
                cluster_counts[int(lbl)] = len(idxs)
                if not idxs:
                    continue
                cluster_path = os.path.join(clusters_dir, f"cluster_{lbl}.jsonl")
                with open(cluster_path, "w", encoding="utf8") as cf:
                    for orig_idx in idxs:
                        cf.write(json.dumps(rows[orig_idx], ensure_ascii=False) + "\n")
            # save a small summary
            summary_path = os.path.join(clusters_dir, "cluster_summary.json")
            with open(summary_path, "w", encoding="utf8") as sf:
                json.dump({"counts": cluster_counts}, sf, ensure_ascii=False, indent=2)
            # Also write high-level cluster stats in out_dir
            stats = {
                "num_clusters_excluding_noise": n_clusters,
                "num_points_total": int(len(labels)),
                "num_outliers": int((labels == -1).sum()) if hasattr(labels, "__array__") else int(sum(1 for l in labels if l == -1)),
                "points_per_cluster": cluster_counts,
            }
            stats_path = os.path.join(args.out_dir, "cluster_stats.json")
            with open(stats_path, "w", encoding="utf8") as sf2:
                json.dump(stats, sf2, ensure_ascii=False, indent=2)
            print(f"Wrote per-cluster files to: {clusters_dir} (clusters: {len(unique_labels)})")
        except Exception:
            # non-fatal
            pass

    reps = find_representatives(embs, labels) if embs is not None and labels is not None else {}
    rep_objs = []
    for lbl, idx_in_texts in reps.items():
        orig_idx = idx_map[idx_in_texts]
        rep = rows[orig_idx].copy()
        rep_objs.append({"cluster": int(lbl), "index": orig_idx, "example": rep})
    reps_path = os.path.join(args.out_dir, "cluster_reps.json")
    with open(reps_path, "w", encoding="utf8") as f:
        json.dump(rep_objs, f, ensure_ascii=False, indent=2)
    print("Wrote cluster representatives:", reps_path)

    try:
        if clusterer is not None:
            clusterer_path = os.path.join(args.out_dir, "clusterer.pkl")
            with open(clusterer_path, "wb") as f:
                pickle.dump(clusterer, f)
        # save medoid embeddings and mapping (label -> medoid_index_in_texts)
        medoid_labels = []
        medoid_idxs = []
        medoid_embs = []
        for lbl, idx_in_texts in reps.items():
            medoid_labels.append(int(lbl))
            medoid_idxs.append(int(idx_in_texts))
            medoid_embs.append(embs[int(idx_in_texts)])
        medoid_embs = np.vstack(medoid_embs) if medoid_embs else np.zeros((0, embs.shape[1]))
        np.savez_compressed(os.path.join(args.out_dir, "cluster_meta.npz"), labels=np.array(medoid_labels), idxs=np.array(medoid_idxs), embs=medoid_embs)
        print("Saved clusterer and medoid metadata to:", args.out_dir)
    except Exception:
        # non-fatal: continue without saving
        pass

    def predict_trc_strings(trc_list, out_path=None):
        if not trc_list:
            return []
        new_embs = embed_texts(model, trc_list, batch_size=args.batch_size, normalize=True)
        try:
            labels_pred, strengths = approximate_predict(clusterer, new_embs)
        except Exception:
            # if approximate_predict fails, fall back to medoid nearest-neighbor (cosine)
            meta = np.load(os.path.join(args.out_dir, "cluster_meta.npz"))
            med_embs = meta["embs"]
            if med_embs.size == 0:
                labels_pred = np.array([-1] * len(trc_list))
                strengths = np.array([0.0] * len(trc_list))
                return list(zip(trc_list, labels_pred, strengths))
            # compute cosine similarities (embs are normalized)
            sims = new_embs @ med_embs.T
            best = sims.argmax(axis=1)
            labels = meta["labels"][best]
            strengths = sims.max(axis=1)
            labels_pred = labels
            strengths = strengths
        preds = []
        for t, lbl, s in zip(trc_list, labels_pred, strengths):
            preds.append({"trc": t, "cluster": int(lbl), "strength": float(s)})
        if out_path:
            with open(out_path, "w", encoding="utf8") as outf:
                for p in preds:
                    outf.write(json.dumps(p, ensure_ascii=False) + "\n")
        return preds

    if args.predict:
        outp = args.pred_out or os.path.join(args.out_dir, "predictions.jsonl")
        # read input predict file (one TRC string per line)
        trc_lines = []
        with open(args.predict, "r", encoding="utf8") as pf:
            for ln in pf:
                ln = ln.strip()
                if ln:
                    trc_lines.append(ln)
        # if clusterer is not loaded (no training), try to load from out_dir
        if clusterer is None:
            try:
                with open(os.path.join(args.out_dir, "clusterer.pkl"), "rb") as f:
                    clusterer = pickle.load(f)
            except Exception as e:
                raise SystemExit(f"No trained clusterer found in memory and failed to load from {args.out_dir}: {e}")
        preds = predict_trc_strings(trc_lines, out_path=outp)
        print(f"Wrote {len(preds)} predictions to: {outp}")
if __name__ == "__main__":
    main()