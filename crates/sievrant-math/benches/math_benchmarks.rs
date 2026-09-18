use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sievrant_core::types::{ChunkMetadata, RankedCandidate, RetrieverType};
use sievrant_math::{
    conformal::{
        compute_threshold, nonconformity_score, CalibrationEntry, CalibrationScore, CalibrationSet,
    },
    rrf::reciprocal_rank_fusion,
    submodular::submodular_select,
};

fn generate_candidates(count: usize, prefix: &str) -> Vec<RankedCandidate> {
    (0..count)
        .map(|i| RankedCandidate {
            chunk_id: format!("{prefix}_{i}"),
            text: format!("Sample text content {i} with some words to take up tokens."),
            score: 1.0 / (i as f64 + 1.0),
            metadata: ChunkMetadata {
                doc_id: "doc1".into(),
                source_doc: "doc1".into(),
                doc_version_hash: "hash".into(),
                path: vec![],
                page: None,
                char_span: Some([0, 100]),
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec![],
                token_count: 10,
            },
            retriever: RetrieverType::VectorIndex,
        })
        .collect()
}

fn bench_rrf_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Reciprocal Rank Fusion");
    for count in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &size| {
            let list1 = generate_candidates(size, "v");
            let list2 = generate_candidates(size, "b");
            let list3 = generate_candidates(size, "t");
            let list4 = generate_candidates(size, "g");
            let lists = vec![list1, list2, list3, list4];

            b.iter(|| reciprocal_rank_fusion(black_box(&lists), black_box(60)))
        });
    }
    group.finish();
}

fn bench_submodular_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("Submodular Selection");
    for count in [100, 500].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &size| {
            let mut candidates = generate_candidates(size, "r");
            for (i, c) in candidates.iter_mut().enumerate() {
                c.score = 1.0 / (i as f64 + 1.0); // Fused score equivalent
            }

            // Simulate reranked output
            let reranked = candidates
                .into_iter()
                .map(|c| {
                    sievrant_core::types::RerankedCandidate {
                        chunk_id: c.chunk_id,
                        text: c.text,
                        metadata: c.metadata,
                        fused_score: c.score,
                        rerank_score: c.score, // Dummy
                    }
                })
                .collect::<Vec<_>>();

            b.iter(|| submodular_select(black_box(&reranked), black_box(1024), black_box(0.3)))
        });
    }
    group.finish();
}

fn bench_conformal_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("Conformal Threshold");
    for size in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &n| {
            // Generate nonconformity scores wrapped in CalibrationSet
            let entries = (0..n)
                .map(|i| CalibrationEntry {
                    query: format!("q_{}", i),
                    relevant_chunk_ids: vec!["c1".into()],
                    scores: vec![CalibrationScore {
                        chunk_id: "c1".into(),
                        nonconformity_score: i as f64 / n as f64,
                        is_relevant: true,
                    }],
                })
                .collect();
            let cal_set = CalibrationSet {
                id: "test-set".into(),
                entries,
                created_at: chrono::Utc::now(),
            };

            b.iter(|| {
                let _ = compute_threshold(black_box(&cal_set), black_box(0.95));
            })
        });
    }
    group.finish();
}

fn bench_nonconformity_score(c: &mut Criterion) {
    c.bench_function("Nonconformity Score Batch 1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(nonconformity_score(
                    black_box(0.8),
                    black_box(0.5),
                    black_box(i as f64 / 1000.0),
                ));
            }
        })
    });
}

criterion_group!(
    benches,
    bench_rrf_fusion,
    bench_submodular_selection,
    bench_conformal_threshold,
    bench_nonconformity_score
);
criterion_main!(benches);
