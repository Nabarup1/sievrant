#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes the MaxSim score using the fastest available SIMD instructions.
/// ColBERT MaxSim: sum over query tokens i of max over document tokens j of dot(q_i, d_j).
pub fn fast_max_sim_score(query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f64 {
    if query_embeddings.is_empty() || doc_embeddings.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2_max_sim(query_embeddings, doc_embeddings) };
        }
    }

    scalar_max_sim(query_embeddings, doc_embeddings)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_max_sim(query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f64 {
    let mut total_score = 0.0f64;

    for q_vec in query_embeddings {
        let mut max_score = f32::NEG_INFINITY;

        for d_vec in doc_embeddings {
            let score = avx2_dot(q_vec, d_vec);
            if score > max_score {
                max_score = score;
            }
        }
        total_score += max_score as f64;
    }

    total_score
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;

    // Accumulators for 8-wide FMA
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();

    // Unroll by 2 (16 elements per iteration)
    while i + 15 < len {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

        i += 16;
    }

    // Single 8-wide
    while i + 7 < len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
        i += 8;
    }

    // Combine sum0 and sum1
    let sum = _mm256_add_ps(sum0, sum1);

    // Horizontal sum of __m256
    // Upper 128 to lower 128
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    // high half to low half
    let sum_high = _mm_movehl_ps(sum128, sum128);
    let sum128 = _mm_add_ps(sum128, sum_high);
    // lowest element
    let sum_single = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    let mut scalar_sum = _mm_cvtss_f32(sum_single);

    // Remainder
    while i < len {
        scalar_sum += a[i] * b[i];
        i += 1;
    }

    scalar_sum
}

fn scalar_max_sim(query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f64 {
    let mut total = 0.0f64;
    for q_vec in query_embeddings {
        let mut max_score = f32::NEG_INFINITY;
        for d_vec in doc_embeddings {
            let mut dot = 0.0f32;
            for k in 0..q_vec.len().min(d_vec.len()) {
                dot += q_vec[k] * d_vec[k];
            }
            if dot > max_score {
                max_score = dot;
            }
        }
        total += max_score as f64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_vs_simd() {
        let q = vec![vec![1.0; 128]];
        let d = vec![vec![0.5; 128], vec![2.0; 128]];

        let scalar = scalar_max_sim(&q, &d);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let simd = unsafe { avx2_max_sim(&q, &d) };
            assert!(
                (scalar - simd).abs() < 1e-4,
                "SIMD {} != Scalar {}",
                simd,
                scalar
            );
        }
    }
}
