# Conformal Prediction in Sievrant

This document explains the mathematics behind Sievrant's statistical coverage guarantee. No prior knowledge of conformal prediction is assumed.

---

## The Problem: When Should an AI Answer?

Standard RAG systems always produce an answer. If the retrieved context does not contain the true answer, the LLM still generates something — it just makes it up. This is called hallucination.

The question Sievrant answers is: **given the retrieved evidence, can we say with 95% confidence (or whatever you choose) that the evidence actually contains the correct answer?** If not, Sievrant abstains rather than hallucinating.

---

## What is Conformal Prediction?

Conformal prediction is a statistical framework for producing *prediction sets* with guaranteed coverage. It was developed by Vovk, Gammerman, and Shafer (2005) and has been applied to everything from classification to time series forecasting.

The key guarantee is:

> **Given any dataset, any model, and any exchangeable (i.i.d.) calibration set, the conformal prediction set contains the true answer with probability at least (1 - α).**

This guarantee holds **without any assumptions on the data distribution**. It is distribution-free, finite-sample, and mathematically provable.

---

## How Sievrant Uses It

Sievrant applies conformal prediction to the question: *does this retrieved evidence set contain the true answer?*

### Step 1: Define the non-conformity score

For each retrieved evidence chunk, we compute a **non-conformity score** that measures how "unusual" or "low-quality" that chunk is relative to the training distribution:

```
nonconformity_score(chunk, query) = 1 - rerank_score(query, chunk)
```

Where `rerank_score` is the cross-encoder similarity score (higher = more relevant). So:
- A highly relevant chunk has nonconformity close to 0.
- An irrelevant chunk has nonconformity close to 1.

### Step 2: Build the calibration set

During the calibration phase (run once per corpus), you provide Sievrant with a set of question-answer pairs where the correct answer is known. For each calibration question:

1. Run the full retrieval pipeline.
2. Compute the non-conformity score for the chunk that contains the true answer.
3. Store this score.

After calibration, you have a distribution of non-conformity scores for "correct" retrievals. This is your calibration distribution.

### Step 3: Compute the threshold

Given a coverage target α (e.g., 0.95 for 95% coverage), the threshold is:

```
q̂ = (⌈(n+1)(1-α)⌉ / n)-th quantile of calibration scores
```

Where `n` is the size of the calibration set. This formula ensures:

```
P(nonconformity_score(true_chunk) ≤ q̂) ≥ 1 - α
```

Sievrant computes this efficiently using a **Fenwick Tree** (Binary Indexed Tree), enabling O(log B) quantile lookup where B is the number of calibration bins.

### Step 4: Apply the gate at query time

For each incoming query, after the submodular selection stage:

1. Compute the non-conformity score for each selected evidence chunk.
2. Reject all chunks with `nonconformity_score > q̂`.
3. If any chunks remain: `Decision::Answered` — pass them to the LLM.
4. If all chunks are rejected: `Decision::Abstained` — return no answer.

The mathematical guarantee is:

```
P(true_answer ∈ retained_set) ≥ 1 - α = coverage_target
```

---

## The Fenwick Tree Implementation

Naive quantile computation over a calibration set of n scores takes O(n) time. For large corpora with tens of thousands of calibration examples, this is unacceptable for an API that must respond in under 100ms.

Sievrant uses a **Fenwick Tree (Binary Indexed Tree)** that discretizes the score range [0, 1] into B bins and maintains prefix sums. This allows:

- **O(log B) threshold lookup** — regardless of calibration set size
- **O(log B) incremental updates** — as new calibration points are added
- **O(1) space overhead** beyond the calibration set itself

The implementation lives in `sievrant-math/src/conformal.rs`.

```rust
// Simplified illustration of the Fenwick-backed quantile lookup
pub struct ConformalGate {
    tree: FenwickTree,
    n_calibration: usize,
    bins: usize,         // B buckets for [0.0, 1.0]
}

impl ConformalGate {
    pub fn threshold(&self, alpha: f64) -> f64 {
        let target_rank = ((self.n_calibration + 1) as f64 * (1.0 - alpha)).ceil() as usize;
        self.tree.find_kth(target_rank) as f64 / self.bins as f64
    }
}
```

---

## Calibration Best Practices

### How many calibration examples do you need?

The coverage guarantee holds for any calibration set size, but tighter guarantees require more examples. For a 95% coverage target:

| Calibration set size | Coverage guarantee |
|---|---|
| 20 | P ≥ 95% (mathematically exact) |
| 100 | Tighter confidence interval on the empirical rate |
| 1,000 | Very stable threshold, high reliability |

We recommend at least 200 question-answer pairs for production use.

### What if my corpus changes frequently?

Recalibrate when:
- Corpus grows by more than 20%
- You add or remove major document categories
- You change the embedding model or reranker

Sievrant's `CalibrationManager` supports incremental recalibration.

---

## Interpreting Results

### `decision: ANSWERED`
At least one evidence chunk passed the conformal gate. The true answer is in the returned evidence with probability ≥ coverage_target.

### `decision: ABSTAINED`
All evidence chunks were rejected. Sievrant is saying: "I cannot retrieve evidence I am sufficiently confident contains the true answer." This is not a failure — it is a correct, honest response. An abstention prevents hallucination.

### `nonconformity_score` in EvidenceChunk
Each evidence chunk carries its non-conformity score. Lower is better. Scores below the threshold q̂ pass the gate.

### `empirical_coverage_last_30d` in EvidencePacket
Sievrant tracks the actual coverage rate (fraction of queries where the true answer was in the returned set) over the past 30 days. This allows you to verify that the mathematical guarantee is empirically holding in production.

---

## Academic References

- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Angelopoulos, A. N., Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
- Quach, V. et al. (2023). *Conformal Language Modeling*. arXiv:2306.10193.
- Ravfogel, S. et al. (2023). *Conformal Prediction for Retrieval-Augmented Generation*. arXiv: (application of CP to RAG).
