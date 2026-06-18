# 🎓 Student Performance AI — End-to-End ML Pipeline

> A complete, from-scratch implementation of classical AI and Machine Learning algorithms applied to the UCI Student Performance dataset. Covers search, constraint satisfaction, clustering, and neural networks — no sklearn ML models used.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Phase Breakdown](#phase-breakdown)
  - [Phase 1 — Data Exploration & Graph Construction](#phase-1--data-exploration--graph-construction)
  - [Phase 2 — Search Algorithms & Adversarial AI](#phase-2--search-algorithms--adversarial-ai)
  - [Phase 3 — Constraint Satisfaction Problem (CSP)](#phase-3--constraint-satisfaction-problem-csp)
  - [Phase 4 — Machine Learning Pipeline](#phase-4--machine-learning-pipeline)
- [Results Summary](#results-summary)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)

---

## Project Overview

This project implements a full AI pipeline on the UCI Student Performance dataset. Every core algorithm — from BFS to backpropagation — is written **from scratch in pure Python/NumPy** to demonstrate fundamental understanding, with `sklearn` used only for preprocessing utilities.

**Target Variable:** `G3` — final student grade (0–20 scale, 18 unique classes)

**Core Problem:** Can a student's behavioral attributes (study time, absences, family support) predict their final academic grade?

---

## Dataset

**Source:** [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/ml/datasets/student+performance)

| Property | Value |
|---|---|
| File | `track_c_education/student_data.csv` |
| Rows | 395 students |
| Columns | 33 features |
| Missing Values | None |
| Target Column | `G3` (final grade, 0–20) |
| Notable Features | `studytime`, `absences`, `failures`, `Medu`, `Fedu`, `health`, `G1`, `G2` |

**Class Distribution (G3):** Heavily concentrated between grades 8–15, with 38 students scoring 0 (likely dropouts) and only 1 scoring 20.

---

## Project Structure

```
student-performance-ai/
│
├── track_c_education/
│   └── student_data.csv          # Raw dataset
│
├── notebook.ipynb                # Main Jupyter Notebook (all phases)
│
└── README.md
```

---

## Phase Breakdown

### Phase 1 — Data Exploration & Graph Construction

**Goal:** Understand the dataset and model relationships as a graph.

**Steps:**
- Loaded and inspected dataset shape, dtypes, and missing values
- Computed class distribution of target variable `G3` using a manual dictionary loop (no `value_counts`)
- Defined a `DataRecord` class encapsulating `record_id`, `features`, and `label`
- Built a **bipartite graph** mapping `StudyTime_X` ↔ `Grade_Y` nodes from dataset rows

**Graph Stats:**

| Property | Value |
|---|---|
| Nodes | 22 (4 StudyTime + 18 Grade) |
| Edges | 59 unique undirected edges |
| Representation | Adjacency list (`dict` of `list`) |

**Design Note:** `StudyTime` and `Grade` nodes were prefixed with string labels to avoid integer key collisions in the dictionary.

---

### Phase 2 — Search Algorithms & Adversarial AI

**State Space:**
- **Initial State:** A study habit node (e.g., `StudyTime_2`)
- **Goal State:** A target grade node (e.g., `Grade_15`)
- **Actions:** Traverse edges in the graph
- **Cost:** Uniform (1 per edge)

#### Uninformed Search

| Algorithm | Path Length | Nodes Explored |
|---|---|---|
| BFS | 1 | 18 |
| DFS | 1 | 2 |
| DLS (limit=3) | 1 | 2 |
| DLS (limit=5) | 1 | 2 |
| IDS (depth=1) | 1 | 3 |
| UCS | 1 | 8 |

> All paths are length 1 because `StudyTime_2` directly connects to `Grade_15` in the graph.

#### Informed Search

**Heuristic Design (Admissible):**
- `h(goal) = 0`
- `h(StudyTime node) = 1` — one hop to any grade
- `h(Grade node, not goal) = 2` — must backtrack through a study habit

| Algorithm | Path Length | Nodes Explored |
|---|---|---|
| Best-First (Greedy) | 1 | 2 |
| A\* | 1 | 2 |
| UCS (baseline) | 1 | 8 |

#### Local Search

| Algorithm | Goal Reached (10 runs) |
|---|---|
| Hill Climbing | 10/10 |
| Simulated Annealing | 10/10 |
| Local Beam Search (k=3) | ✅ Found `Grade_15` |
| Local Beam Search (k=5) | ✅ Found `Grade_15` |

#### Adversarial Search (Minimax vs Alpha-Beta)

Starting from `StudyTime_2`, depth limit = 4, goal = `Grade_15`:

| Algorithm | Nodes Evaluated |
|---|---|
| Standard Minimax | 3,540 |
| Alpha-Beta Pruning | 616 |
| **Nodes Eliminated** | **2,924 (82.6% reduction)** |

---

### Phase 3 — Constraint Satisfaction Problem (CSP)

**Problem:** Define a "High-Performing Student Profile" (G3 ≥ 15).

**Variables & Domains:**

| Variable | Domain |
|---|---|
| `StudyTime` | {1, 2, 3, 4} |
| `Absences` | {Low, Medium, High} |
| `Health` | {1, 2, 3, 4, 5} |
| `Activities` | {yes, no} |
| `FamSup` | {yes, no} |

**Constraints:**

| Rule | Description |
|---|---|
| R1 | `StudyTime == 1` → `Absences != High` |
| R2 | `Activities == yes` → `Health >= 3` |
| R3 | `FamSup == no` → `StudyTime >= 3` |
| R4 | `StudyTime + Health >= 4` |
| R5 | `Absences == High` → `Activities == no` |

#### AC-3 Arc Consistency

AC-3 ran successfully without eliminating any values — indicating that no single-variable domain reduction was possible from binary constraints alone; solutions exist across the full domain space.

#### Backtracking Comparison

| Variant | Solution Found | Backtracks |
|---|---|---|
| Basic Backtracking | `{StudyTime:1, Absences:Low, FamSup:yes, Activities:yes, Health:3}` | 0 |
| Forward Checking | `{StudyTime:1, Absences:Low, FamSup:yes, Activities:yes, Health:3}` | 0 |
| MRV Heuristic | `{FamSup:yes, Activities:yes, Absences:Low, Health:3, StudyTime:1}` | 0 |

> All variants found a valid solution immediately. MRV reorders assignment by most-constrained variable first.

#### Min-Conflicts

Converged in **0 iterations** — the random initial assignment happened to violate no constraints, demonstrating that valid assignments are dense in this CSP's solution space.

---

### Phase 4 — Machine Learning Pipeline

#### Preprocessing

- Missing values: filled with column mean (numeric) / mode (categorical)
- Categorical encoding: `LabelEncoder` on all `object` columns
- Normalization: Z-score standardization on all numeric features except `G3`
- `G1` and `G2` dropped to force the model to learn from behavioral features only
- Train/test split: **80/20**, `random_state=42`

| Split | Shape |
|---|---|
| `X_train` | (316, 30) |
| `X_test` | (79, 30) |

#### Unsupervised — K-Means & K-Medoids (From Scratch)

`k = 18` (matching unique grade classes)

| Algorithm | Convergence | WCSD |
|---|---|---|
| K-Means | Iteration 9 | 1068.12 |
| K-Medoids | Iteration 3 | 1246.05 |

**Cluster Purity:** 16–50% per cluster. Low purity is expected — students with identical habits can earn very different grades.

#### Supervised — Perceptron (From Scratch)

- **Task:** Binary classification (Pass: G3 ≥ 10 / Fail: G3 < 10)
- **Learning Rate:** 0.01 | **Epochs:** 50
- **Final Training Accuracy:** ~60.76%
- **Observation:** Dataset is not linearly separable; perceptron cannot converge perfectly.

#### Supervised — Delta Rule / Batch Gradient Descent (From Scratch)

- **Task:** Regression on continuous G3 values
- **Learning Rate:** 0.01 | **Epochs:** 100
- **Final MSE:** 19.3350
- **Update Rule:** Averaged gradients across all samples per epoch (true batch GD)

#### Supervised — Multilayer Perceptron (From Scratch)

**Architecture:** `30 → 16 → 8 → 18`

| Layer | Size | Activation |
|---|---|---|
| Input | 30 | — |
| Hidden 1 | 16 | ReLU |
| Hidden 2 | 8 | ReLU |
| Output | 18 | Softmax |

**Training:** Cross-Entropy Loss, backpropagation with chain rule, SGD updates

| Epoch | Loss | Accuracy |
|---|---|---|
| 0 | 2.8879 | 7.28% |
| 100 | 2.6979 | 14.24% |
| 199 | 2.5754 | 17.72% |

> Loss decreases consistently. Accuracy is low due to 18-class difficulty with behavioral-only features (G1/G2 excluded).

---

## Results Summary

| Component | Method | Key Result |
|---|---|---|
| Uninformed Search | BFS | Finds goal in 1 hop, 18 nodes explored |
| Informed Search | A\* | Finds goal in 1 hop, 2 nodes explored |
| Adversarial AI | Alpha-Beta | 82.6% node reduction vs Minimax |
| CSP Solving | MRV + FC | 0 backtracks needed |
| Clustering | K-Means | WCSD 1068.12, converged iteration 9 |
| Binary Classification | Perceptron | ~60.76% accuracy (non-separable data) |
| Regression | Delta Rule | MSE 19.34 after 100 epochs |
| Multi-class NN | MLP (30→16→8→18) | 17.72% accuracy, loss 2.575 |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/naynay575/student-performance-ai.git
cd student-performance-ai

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebook.ipynb
```

---

## Usage

All code is self-contained in a single Jupyter Notebook organized by phase. Run cells sequentially — each phase depends on variables defined in previous ones.

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4
(Data)      (Search)     (CSP)       (ML)
```

To change the **goal state** for search algorithms, update:

```python
target_grade = "Grade_15"   # Change to any Grade_0 through Grade_20
agent = AIAgent(graph=graph, goal_state=target_grade)
```

To change the **start state** for searches:

```python
initial_state = "StudyTime_2"  # Options: StudyTime_1, _2, _3, _4
```

---

## Key Design Decisions

**Graph as bipartite structure** — Connecting `StudyTime` and `Grade` nodes via student rows creates a simple, interpretable state space without needing weighted edges.

**G1/G2 dropped intentionally** — These intermediate grades are near-perfect predictors of G3, making the ML task trivial. Dropping them forces models to learn from behavioral features, which is the actual research question.

**k=18 for clustering** — Matching the number of unique grade values ensures one-to-one comparison with true labels when computing cluster purity.

**ReLU over Sigmoid in MLP** — ReLU avoids the vanishing gradient problem that would slow learning in a 3-layer network trained on a small dataset.

**Softmax + Cross-Entropy shortcut** — The combined derivative simplifies to `A3 - Y_one_hot`, which was used in backpropagation for numerical stability and implementation correctness.

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

```bash
pip install numpy pandas matplotlib scikit-learn
```

> No external AI/ML libraries were used for algorithm implementation. `sklearn` is used exclusively for `LabelEncoder`, `train_test_split`, and dataset loading utilities.

---

## Author

**Nehal Aftab** — CS Student, FAST-NUCES Faisalabad (Batch 2024–2028)

[![GitHub](https://img.shields.io/badge/GitHub-naynay575-181717?logo=github)](https://github.com/naynay575)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nehal--aftab-0A66C2?logo=linkedin)](https://linkedin.com/in/nehal-aftab)
[![Email](https://img.shields.io/badge/Email-nehal.aftab05%40gmail.com-D14836?logo=gmail)](mailto:nehal.aftab05@gmail.com)
