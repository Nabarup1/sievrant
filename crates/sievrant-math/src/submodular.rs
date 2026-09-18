use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{Hash, Hasher};

use sievrant_core::types::{RerankedCandidate, SelectedEvidence};

#[derive(Debug, Clone)]
struct QueueItem {
    candidate_idx: usize,
    marginal_gain_per_token: f64,
    evaluated_at_subset_size: usize,
    marginal_gain_absolute: f64,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.marginal_gain_per_token == other.marginal_gain_per_token
    }
}
impl Eq for QueueItem {}
impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.marginal_gain_per_token
            .partial_cmp(&other.marginal_gain_per_token)
    }
}
impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn hash_word(word: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    word.hash(&mut hasher);
    hasher.finish()
}

/// Computes exact Jaccard similarity over pre-sorted, deduplicated arrays of token hashes.
/// Extremely fast O(N) intersection with no memory allocation.
fn sorted_jaccard(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let mut i = 0;
    let mut j = 0;
    let mut intersection = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Marginal gain of adding `candidate_idx` to the already-selected set.
///
/// gain(c | S) = relevance(c) + diversity_weight * diversity(c, S)
/// diversity(c, S) = 1 - max_jaccard(c, any s in S)
fn compute_marginal_gain(
    candidate: &RerankedCandidate,
    candidate_tokens: &[u64],
    selected_token_sets: &[&[u64]],
    diversity_weight: f64,
) -> f64 {
    let relevance = candidate.rerank_score;

    let diversity = if selected_token_sets.is_empty() {
        1.0
    } else {
        let max_sim = selected_token_sets
            .iter()
            .map(|sel_tokens| sorted_jaccard(candidate_tokens, sel_tokens))
            .fold(0.0f64, f64::max);
        1.0 - max_sim
    };

    relevance + diversity_weight * diversity
}

/// Lazy Greedy submodular maximization under a token budget (knapsack) constraint.
/// Uses Minoux's 1978 algorithm to avoid O(N^2) evaluations, backed by a max-heap.
pub fn submodular_select(
    candidates: &[RerankedCandidate],
    token_budget: usize,
    diversity_weight: f64,
) -> Vec<SelectedEvidence> {
    let mut selected: Vec<SelectedEvidence> = Vec::new();
    let mut selected_token_sets: Vec<Vec<u64>> = Vec::new(); // Store hashes parallel to `selected`

    let mut remaining_budget = token_budget;
    let mut used: HashSet<usize> = HashSet::new();

    // 1. Precompute token hashes and dedup for fast O(1) similarity later
    let candidate_token_sets: Vec<Vec<u64>> = candidates
        .iter()
        .map(|c| {
            let mut tokens: Vec<u64> = c.text.split_whitespace().map(hash_word).collect();
            tokens.sort_unstable();
            tokens.dedup();
            tokens
        })
        .collect();

    // 2. Initialize the priority queue with marginal gains evaluated against the empty set
    let mut queue: BinaryHeap<QueueItem> = BinaryHeap::with_capacity(candidates.len());
    for (i, candidate) in candidates.iter().enumerate() {
        let token_cost = candidate.metadata.token_count;
        if token_cost <= remaining_budget {
            let marginal = compute_marginal_gain(
                candidate,
                &candidate_token_sets[i],
                &[], // Empty set initially
                diversity_weight,
            );
            queue.push(QueueItem {
                candidate_idx: i,
                marginal_gain_per_token: marginal / token_cost.max(1) as f64,
                evaluated_at_subset_size: 0,
                marginal_gain_absolute: marginal,
            });
        }
    }

    // 3. Lazy Greedy selection loop
    while let Some(top) = queue.pop() {
        if used.contains(&top.candidate_idx) {
            continue;
        }

        let token_cost = candidates[top.candidate_idx].metadata.token_count;
        if token_cost > remaining_budget {
            continue;
        }

        // If this candidate was evaluated against the CURRENT selected set, it is guaranteed
        // to be the max marginal gain due to submodularity. Select it.
        if top.evaluated_at_subset_size == selected.len() {
            let candidate = &candidates[top.candidate_idx];
            remaining_budget -= token_cost;
            used.insert(top.candidate_idx);

            selected.push(SelectedEvidence {
                chunk_id: candidate.chunk_id.clone(),
                fused_score: candidate.fused_score,
                rerank_score: candidate.rerank_score,
                marginal_gain: top.marginal_gain_absolute,
                nonconformity_score: 0.0,
                text: candidate.text.clone(),
                metadata: candidate.metadata.clone(),
            });
            selected_token_sets.push(candidate_token_sets[top.candidate_idx].clone());
        } else {
            // Otherwise, its marginal gain is stale. Recompute it against the new selected set.
            // Create a view of references to pass to the function.
            let refs: Vec<&[u64]> = selected_token_sets.iter().map(|v| v.as_slice()).collect();
            let new_marginal = compute_marginal_gain(
                &candidates[top.candidate_idx],
                &candidate_token_sets[top.candidate_idx],
                &refs,
                diversity_weight,
            );

            queue.push(QueueItem {
                candidate_idx: top.candidate_idx,
                marginal_gain_per_token: new_marginal / token_cost.max(1) as f64,
                evaluated_at_subset_size: selected.len(),
                marginal_gain_absolute: new_marginal,
            });
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use sievrant_core::types::{ChunkMetadata, RerankedCandidate};

    fn make_candidate(id: &str, rerank_score: f64, tokens: usize, text: &str) -> RerankedCandidate {
        RerankedCandidate {
            chunk_id: id.to_string(),
            fused_score: rerank_score,
            rerank_score,
            text: text.to_string(),
            metadata: ChunkMetadata {
                doc_id: id.into(),
                source_doc: "test.pdf".into(),
                doc_version_hash: "sha256:test".into(),
                path: vec![],
                page: None,
                char_span: None,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec![],
                token_count: tokens,
            },
        }
    }

    #[test]
    fn budget_is_never_exceeded_deterministic() {
        let candidates = vec![
            make_candidate("c1", 0.9, 300, "the lease term is sixty months"),
            make_candidate("c2", 0.8, 200, "payment schedule monthly installments"),
            make_candidate("c3", 0.7, 600, "this chunk is too large for the budget"),
        ];
        let selected = submodular_select(&candidates, 500, 0.3);
        let total_tokens: usize = selected.iter().map(|s| s.metadata.token_count).sum();
        assert!(total_tokens <= 500);
    }

    #[test]
    fn redundant_chunk_is_penalized() {
        let repeated_text = "lease term sixty months agreement";
        let candidates = vec![
            make_candidate("c1", 0.9, 50, repeated_text),
            make_candidate("c2", 0.85, 50, repeated_text),
            make_candidate("c3", 0.5, 50, "payment schedule rent amount due"),
        ];
        let selected = submodular_select(&candidates, 200, 1.0);

        assert_eq!(selected[0].chunk_id, "c1");

        let selected_ids: Vec<&str> = selected.iter().map(|s| s.chunk_id.as_str()).collect();
        let c3_pos = selected_ids.iter().position(|&id| id == "c3");
        let c2_pos = selected_ids.iter().position(|&id| id == "c2");
        if let (Some(p3), Some(p2)) = (c3_pos, c2_pos) {
            assert!(p3 < p2, "diverse c3 should rank before redundant c2");
        }
    }

    #[test]
    fn highest_relevance_selected_first_on_empty_set() {
        let candidates = vec![
            make_candidate("c1", 0.5, 100, "alpha beta gamma"),
            make_candidate("c2", 0.9, 100, "delta epsilon zeta"),
            make_candidate("c3", 0.7, 100, "eta theta iota"),
        ];
        let selected = submodular_select(&candidates, 1000, 0.0);
        assert_eq!(
            selected[0].chunk_id, "c2",
            "highest rerank score must be first"
        );
    }

    #[test]
    fn empty_candidates_returns_empty() {
        let result = submodular_select(&[], 1000, 0.3);
        assert!(result.is_empty());
    }

    #[test]
    fn zero_budget_returns_empty() {
        let candidates = vec![make_candidate("c1", 0.9, 100, "some text")];
        let result = submodular_select(&candidates, 0, 0.3);
        assert!(result.is_empty());
    }

    fn arb_candidate() -> impl Strategy<Value = RerankedCandidate> {
        (0.0f64..=1.0, 1usize..=500, "[a-z ]{1,30}").prop_map(|(score, tokens, text)| {
            static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            make_candidate(&format!("c{id}"), score, tokens, &text)
        })
    }

    proptest! {
        #[test]
        fn budget_never_exceeded_proptest(
            candidates in prop::collection::vec(arb_candidate(), 1..50),
            budget in 1usize..5000,
        ) {
            let selected = submodular_select(&candidates, budget, 0.3);
            let total_tokens: usize = selected.iter().map(|s| s.metadata.token_count).sum();
            prop_assert!(
                total_tokens <= budget,
                "token budget exceeded: {total_tokens} > {budget}"
            );
        }

        #[test]
        fn all_marginal_gains_are_non_negative(
            candidates in prop::collection::vec(arb_candidate(), 1..20),
        ) {
            let selected = submodular_select(&candidates, 100_000, 0.3);
            for evidence in &selected {
                prop_assert!(
                    evidence.marginal_gain >= 0.0,
                    "negative marginal gain: {}",
                    evidence.marginal_gain
                );
            }
        }
    }
}
