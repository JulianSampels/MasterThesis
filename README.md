# Split-Join-Predict: Edge-Agnostic Tuple Prediction Framework

**This repository contains the source code for the Master's Thesis: "Enhancing Knowledge Graph Completion Using Graph Neural Networks".**

It implements the **Split-Join-Predict** framework, a scalable approach for entity-agnostic tuple prediction in Knowledge Graphs using the **PathE [[1]](#ref1)** embedding model as its backbone.

## Repository Structure

The core logic is located in the `PathE/pathe/` directory.

```
PathE/
└── pathe/
    ├── runner.py           # Main CLI entry point for training and evaluation
    ├── pathe_trainer.py    # PyTorch Lightning modules and training loops
    ├── pather_models.py    # Neural network architectures (PathE, Multi-Heads)
    ├── candidates.py       # Phase 2: Efficient global candidate generation 
    ├── kgloader.py         # Knowledge Graph dataset loading and processing
    ├── pathdataset.py      # Path generation and sampling strategies
    ├── wrappers.py         # Model wrappers for Task adaptation (UniqueHeads, Triples)
    └── data_utils.py       # Data collation and tensor optimised loading
```

## Methodology: The Split-Join-Predict Framework

This codebase implements the methodology described in **Chapter 5** of the thesis. The framework decomposes the computationally expensive tuple prediction task `(h, ?, ?)` into three scalable phases.

![Overview of the Split-Join-Predict framework](https://github.com/user-attachments/assets/ac143371-a5ea-41a5-a5a3-e32ad74382a6)
*Figure: Overview of the Split-Join-Predict framework (Figure 5.1 from the Thesis).*

### Phase 1: Property Prediction (Split)
The tuple prediction problem is split into independent sub-problems: predicting likely relations `(h, r)` and likely tail entities `(h, t)` for a given head `h`.

*   **Approach**: Implements an **Entity-Centric** view using `PathEModelWrapperUniqueHeads`.
*   **Objectives**: Supports **Binary Existence** (BCE) and **Frequency Estimation** (Poisson, Negative Binomial, Hurdle) to model relation types and entities as independent properties.
*   **Scalability**: Utilises **Path-Based Data Augmentation** and pre-computed sparse adjacency matrices to ensure efficient training on large graphs.

### Phase 2: Candidate Generation (Join)
Independent property predictions are joined into a joint probabilistic score to filter the search space.

*   **Global Joint Scoring**: Combines independent log-probabilities from Phase 1 into a unified score $S(h, r, t)$.
*   **Implementation**: Uses a highly parallelised streaming architecture (in `candidates.py`) to perform **Global Top-k selection** without materialising the full combinatorial tensor.

### Phase 3: Triple Classification (Predict)
The candidate set is refined using a structural discriminator to remove false positives.

*   **Refinement**: The model trains on **"Hard Negatives"** to learn fine-grained structural distinctions.
*   **Model**: The class `PathEModelTriples` evaluates the full structural coherence of the triple `(h, r, t)`.


## Usage

### 1. Installation
First, install **CUDA 11.8**. Ensure that the correct version is active and accessible in your path by checking the compiler version:

```bash
nvcc --version
```

Install dependencies:
```bash
pip install -r PathE/pathe/requirements.txt
```

### 2. Data Preparation
Generate the path datasets using `generate_dataset.py`:
```bash
python generate_dataset.py
```

### 3. Training
The system is executed via the `PathE.pathe.runner` module.

#### Example Experiment

The following command runs a complete Split-Join-Predict experiment on **FB15k-237**, utilising the **Binary Cross Entropy (BCE)** objective for both relation and tail prediction.

```bash
python -m PathE.pathe.runner train SplitJoinPredictPathE \
  --log_dir ./logs/FB15k237 \
  --expname RelationBCE-EntityBCE \
  --train_paths ./data/path_datasets/fb15k237/train/ \
  --valid_paths ./data/path_datasets/fb15k237/val/ \
  --test_paths ./data/path_datasets/fb15k237/test/ \
  --path_setup 20_10 \
  --max_ppt 4 \
  --batch_size 2000 \
  --embedding_dim 64 \
  --num_workers 64 \
  --ent_aggregation transformer \
  --use_manual_optimization \
  --augmentation_factor 10 \
  --phase1_rp_loss_fn bce \
  --phase1_tp_loss_fn bce \
  --loss_weight 0.5
```

### Key Arguments

*   `--model SplitJoinPredictPathE`: Activates the Split-Join-Predict pipeline.
*   `--phase1_rp_loss_fn`: Loss function for relation properties (e.g., `bce`, `poisson`, `negative_binomial`).
*   `--phase1_tp_loss_fn`: Loss function for tail properties.
*   `--augmentation_factor`: Multiplier for path-based data augmentation in Phase 1.
*   `--candidates_cap`: Maximum number of candidates to keep per group in Phase 2.


## The Path Embedder (PathE)

This work builds on **PathE [[1]](#ref1)**, an inductive, entity-agnostic encoder using relational context paths. Here, the architecture is adapted to support **Entity-Centric** inference for Split-Join-Predict, transitioning from triple-based to unique-head processing.

## References
<a id="ref1"></a>
[1] Reklos, I., de Berardinis, J., Simperl, E., & Meroño-Peñuela, A. (2025). PathE: Leveraging Entity-Agnostic Paths for Parameter-Efficient Knowledge Graph Embeddings. *arXiv preprint arXiv:2501.19095*.

---

***AI Usage Declaration:** This repository was developed with the assistance of GitHub Copilot and Google Gemini 3, utilising them for code autocompletion, documentation generation, and boilerplate refactoring. The core logic, algorithms, and Split-Join-Predict architecture remain the original work of the author.*