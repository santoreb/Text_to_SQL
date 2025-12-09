# **Text2SQL – Clustering-Enhanced Fine-Tuning Framework**

This repository contains all code and experiments for our **Text-to-SQL fine-tuning project**, where we evaluate multiple training strategies on **Spider 1.0** and **BIRD**, including clustering-based routing methods and classical ML baselines.
The core model used throughout is **CodeLlama**, fine-tuned under multiple configurations.

---

## Demo Link: https://www.dropbox.com/scl/fi/bm259r584kjovl4lwb563/Screen-Recording-2025-12-01-at-10.17.14-AM.mov?rlkey=4rt976tuh2i74sf07je7y1r14&st=7exj0gty&dl=0

## 🚀 **Project Overview**

We trained and evaluated the following pipelines:

---

## **1. Standard Fine-Tuning (Baseline)**

### Datasets:

* **Spider 1.0**
* **BIRD**

### Method:

* Standard supervised fine-tuning (SFT)
* Train on `(NL question → SQL)` pairs
* Evaluate exact match (EM), execution accuracy

Scripts:

```
Finetuning/
├── preprocess_spider.py
├── finetune_codellama.py
├── inference_codellama_sql.py
├── generate_predictions.py
└── evaluate_sql_model.py
```

---

## **2. Clustering-Based Methods (Spider 1.0)**

To improve generalization and reduce semantic drift, we implemented two cluster-based specialization pipelines:

---

# 🧠 **A. Natural Language (NL) Text Clustering**

Directory: `NL_text_clustering/`

### **Goal:**

Cluster natural-language questions based on semantic embeddings.

### **Pipeline:**

1. Encode Spider questions using Sentence Transformers.
2. Run **K-Means** to obtain semantic clusters.
3. Split dataset into `train_cluster_k.jsonl` per cluster.
4. Fine-tune CodeLlama **per cluster**.
5. **Routing at inference:** assign test query to nearest cluster centroid.

### **Files:**

```
NL_text_clustering/
├── cluster_spider.py
├── compute_cluster_centroids.py
├── evaluate_sql_model_using_clusters.py
└── clusters/
    ├── cluster_assignments.json
    ├── cluster_centroids.npy
    ├── cluster_sizes.json
    ├── train_cluster_0.jsonl
    ├── train_cluster_1.jsonl
    ├── train_cluster_2.jsonl
    └── train_cluster_3.jsonl
```

---

# 🌳 **B. SQL AST-Based Clustering**

Directory: `SQL_AST_clustering/`

### **Goal:**

Cluster SQL queries by **structural similarity** using ASTs.

### **Pipeline:**

1. Convert each SQL query → AST.
2. Compute tree-based vector representation.
3. Run clustering using structural similarity.
4. Fine-tune one model per AST cluster.
5. At inference, assign to nearest AST cluster.

### **Files:**

```
SQL_AST_clustering/
├── ast_cluster_spider.py
├── cluster_trc.py
├── nltk_load.py
└── requirements.txt
```

---

## 🔁 **3. Supervised Cluster Assignment (Routing Classifier)**

After creating clusters (both NL and AST), we additionally trained a **supervised classifier** to predict a query’s cluster ID automatically.

### Why?

* Nearest-centroid routing is simple but sometimes noisy.
* A classifier trained on embedded questions improves assignment accuracy.

### Classifier types tested:

* Logistic Regression
* Linear SVM
* Random Forest
* Simple 2-layer feed-forward neural network

These models take *query embeddings* as input and predict the cluster label.
This yielded better routing than pure distance-based assignment.

---

## 🧪 **4. ML-Based Baseline – IRNet (for comparison)**

To compare our LLM-based fine-tuning strategies against classical neural semantic parsers, we evaluated:

### ✔ **IRNet (Information-Retrieval Augmented Text-to-SQL Parser)**

A strong neural baseline model used widely for Spider research.

### GitHub Repository (official Microsoft repo):

👉 **[https://github.com/microsoft/IRNet](https://github.com/microsoft/IRNet)**

We include it as a baseline reference but do not re-train it inside this repo.

---

# 📂 **Repository Structure**

```
Text_to_SQL/
│
├── Finetuning/                   # SFT training & evaluation scripts
├── NL_text_clustering/          # Natural language cluster pipeline
├── SQL_AST_clustering/          # SQL AST clustering pipeline
└── README.md
```

---

# 📊 **Evaluation**

We evaluate the following:

### **Baselines**

* Standard CodeLlama fine-tuning on Spider & BIRD
* IRNet (external baseline)

### **Clustering Approaches**

* NL Clusters (per-cluster models + ensemble)
* AST Clusters (per-cluster models + ensemble)
* Supervised Cluster Routing (classifier)
* Centroid-Based Routing

### Metrics:

* **Execution Accuracy**: It measures the semantic correctness of a predicted SQL query by executing both the predicted and gold queries on the target database and comparing their returned results. A prediction is counted as correct if the two result sets match exactly, regardless of whether the SQL strings themselves differ syntactically.

---

# 🔧 **Reproduction Steps**

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Spider

```bash
python Finetuning/preprocess_spider.py
```

### 3. Run Standard Fine-Tuning

```bash
python Finetuning/finetune_codellama.py \
    --train_file train.json \
    --output_dir outputs/codellama_spider
```

### 4. Run Clustering Pipelines

**Natural Language Clustering**

```bash
python NL_text_clustering/cluster_spider.py
```

**SQL AST Clustering**

```bash
python SQL_AST_clustering/ast_cluster_spider.py
```

### 5. Evaluate

```bash
python Finetuning/evaluate_sql_model.py
python Finetuning/evaluate_sql_model_using_clusters.py
```

---

# 🧩 **Key Contributions**

* Fine-tuning CodeLlama on Spider and BIRD
* Natural language semantic clustering (K-Means)
* SQL AST structural clustering
* Routing via both **centroid** and **supervised classifier**
* Comparison with classical IRNet model
* Cluster-specialized model ensemble for improved Text-to-SQL accuracy

---

