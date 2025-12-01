import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def extract_question(text):
    try:
        return text.split("QUESTION:")[1].split("RETURN SQL:")[0].strip()
    except:
        return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_dir",
                        default="/home/jjvyas1/codellama_finetune/processed_spider/clusters")
    parser.add_argument("--num_clusters", type=int, default=4)
    args = parser.parse_args()

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    centroids = []
    cluster_sizes = {}

    for cid in range(args.num_clusters):
        cluster_file = f"{args.cluster_dir}/train_cluster_{cid}.jsonl"

        if not os.path.exists(cluster_file):
            raise FileNotFoundError(f"Cluster file missing: {cluster_file}")

        print(f"Loading {cluster_file}")
        records = [json.loads(l) for l in open(cluster_file)]

        # Extract questions
        questions = [extract_question(r["input"]) for r in records]
        cluster_sizes[cid] = len(questions)

        print(f"Embedding {len(questions)} questions for cluster {cid}...")
        embeds = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=True)

        centroid = np.mean(embeds, axis=0)
        centroids.append(centroid)

    centroids = np.vstack(centroids)

    out_path = f"{args.cluster_dir}/cluster_centroids.npy"
    np.save(out_path, centroids)
    print(f"\nSaved centroids to {out_path}")

    with open(f"{args.cluster_dir}/cluster_sizes.json", "w") as f:
        json.dump(cluster_sizes, f, indent=2)
    print(f"Saved cluster sizes to {args.cluster_dir}/cluster_sizes.json")
    print("\nDone computing centroids.")


if __name__ == "__main__":
    main()

