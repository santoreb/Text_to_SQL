#!/usr/bin/env python3
"""
Cluster Spider train.jsonl into a small number of semantic clusters
based on question embeddings.

Outputs:
    train_cluster_0.jsonl
    train_cluster_1.jsonl
    ...
    cluster_assignments.json
"""

import json
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
import os


# Extract only the QUESTION section from the input prompt.
def extract_question(full_input_text):
    try:
        part = full_input_text.split("QUESTION:")[1]         # Split at "QUESTION:" and before "RETURN"
        question = part.split("RETURN SQL:")[0].strip()
        return question
    except:
        return full_input_text  # fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to train.jsonl")
    parser.add_argument("--out_dir", default="./clusters", help="Output directory")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of clusters")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading training data from {args.input_file}")
    records = [json.loads(l) for l in open(args.input_file)]
    questions = [extract_question(r["input"]) for r in records]

    print(f"Encoding {len(questions)} questions using SentenceTransformer...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    print(f"Running KMeans with k={args.num_clusters}")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Save cluster assignments
    cluster_path = os.path.join(args.out_dir, "cluster_assignments.json")
    with open(cluster_path, "w") as f:
        json.dump(labels.tolist(), f, indent=2)
    print(f"Saved cluster assignments to {cluster_path}")

    # Save JSONL files per cluster
    clusters = {i: [] for i in range(args.num_clusters)}
    for rec, label in zip(records, labels):
        clusters[label].append(rec)

    for i in range(args.num_clusters):
        out_path = os.path.join(args.out_dir, f"train_cluster_{i}.jsonl")
        with open(out_path, "w") as f:
            for rec in clusters[i]:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(clusters[i])} samples to {out_path}")

    # Show examples
    print("\nSample questions per cluster:")
    for i in range(args.num_clusters):
        print(f"\nCluster {i} ({len(clusters[i])} examples)")
        for q in questions[:5]:
            break  # placeholder
    
    for i in range(args.num_clusters):
        print(f"\n\nCluster {i}")
        cluster_questions = [extract_question(r["input"]) for r in clusters[i]]
        for q in cluster_questions[:5]:
            print(f"  --- {q}")

    print("\n Clustering Done.")

if __name__ == "__main__":
    main()

