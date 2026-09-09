# Submodular Evidence Selection

This document explains how Sievrant selects which retrieved evidence chunks to include in the final context sent to the LLM, and the mathematical guarantees behind this selection.

---

## The Problem: Redundancy and Token Waste

After the cross-encoder reranker scores all candidates, you have a ranked list. A naive approach is to take the top-k chunks by score. This fails for two reasons:

1. **Redundancy**: The top 5 chunks might all say the same thing in slightly different words. Sending all 5 wastes tokens and pushes other relevant content out of the context window.
2. **Budget blindness**: Different chunks have different lengths. Top-k does not respect a total token budget.

The optimal selection problem (maximize coverage, minimize redundancy, respect token budget) is NP-hard in general. However, the objective function belongs to a class of functions called **submodular functions**, and for submodular functions, the greedy algorithm provides provable approximation guarantees.

---

## What is a Submodular Function?

A set function f: 2^V → ℝ is submodular if it satisfies the **diminishing returns** property:

> For any two sets A ⊆ B ⊆ V and any element e ∉ B:
> **f(A ∪ {e}) - f(A) ≥ f(B ∪ {e}) - f(B)**

In plain English: adding a new element to a smaller set increases the function value by at least as much as adding it to a larger set. The more you already have, the less new value each addition brings.

The **facility location function** (which Sievrant uses) captures diversity:

```
f(S) = Σ_{q ∈ queries} max_{c ∈ S} sim(q, c)
```

This function measures how well the selected set S "covers" all aspects of the query. Adding a chunk that covers a new aspect of the query yields high marginal gain. Adding a redundant chunk yields near-zero marginal gain.

---

## The Lazy Greedy Algorithm

The basic greedy algorithm for submodular maximization:

1. Start with S = {} (empty set)
2. While token_budget allows:
   a. For each remaining chunk c, compute marginal gain: Δ(c) = f(S ∪ {c}) - f(S)
   b. Select c* = argmax Δ(c)
   c. Add c* to S

**Guarantee**: The greedy algorithm achieves at least **(1 - 1/e) ≈ 63.2%** of the optimal solution (Nemhauser, Wolsey, Fisher — 1978). This guarantee holds regardless of corpus size.

**Naive complexity**: O(N × k) evaluations, where N is the candidate count and k is the number of selected chunks. For N=1000 and k=20, this is 20,000 similarity computations per query.

**Lazy Greedy optimization** (Minoux 1978): Uses a max-priority queue to avoid recomputing marginal gains for elements whose upper bound has not changed. In practice, Lazy Greedy evaluates only a small fraction of the O(N×k) computations while achieving the same result.

Sievrant's implementation uses a `BinaryHeap` in Rust:

```rust
// Simplified illustration of the Lazy Greedy implementation
pub fn lazy_greedy_select(
    candidates: &[ScoredChunk],
    token_budget: usize,
    sim_matrix: &SimMatrix,
) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::new();
    let mut total_tokens = 0;
    // Priority queue: (upper_bound_marginal_gain, chunk_idx, is_stale)
    let mut heap: BinaryHeap<(OrderedFloat<f64>, usize, bool)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (OrderedFloat(c.score), i, true))
        .collect();

    while total_tokens < token_budget {
        let Some((_, idx, stale)) = heap.pop() else { break };

        if stale {
            // Recompute exact marginal gain for this element
            let gain = marginal_gain(idx, &selected, sim_matrix);
            heap.push((OrderedFloat(gain), idx, false));
            continue;
        }

        // This element has exact gain at the top — select it
        total_tokens += candidates[idx].token_count;
        selected.push(idx);
    }

    selected
}
```

---

## Token Budget Accounting

The `token_budget` in `RetrievalConfig` (default 4000) represents the maximum total tokens across all selected evidence chunks to include in the LLM prompt.

Sievrant tracks `token_count` per chunk, ingested at document processing time using a tokenizer. The submodular selection respects this budget strictly — the selected set never exceeds the configured budget.

This prevents prompt overflow and enables predictable LLM cost control.

---

## Marginal Gain in the EvidencePacket

Every evidence chunk in the returned `EvidencePacket` carries its `submodular_marginal_gain` score. This represents the additional coverage that chunk contributed beyond what was already selected.

High `submodular_marginal_gain` means the chunk covers aspects of the query not addressed by other chunks. Low `submodular_marginal_gain` (but still selected) means it was selected primarily for confirmation, as redundancy reduction already removed duplicates.

---

## Academic References

- Nemhauser, G., Wolsey, L., Fisher, M. (1978). *An Analysis of Approximations for Maximizing Submodular Set Functions*. Mathematical Programming.
- Minoux, M. (1978). *Accelerated Greedy Algorithms for Maximizing Submodular Set Functions*. Optimization Techniques.
- Lin, H., Bilmes, J. (2011). *A Class of Submodular Functions for Document Summarization*. ACL.
