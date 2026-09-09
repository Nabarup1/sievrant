use sievrant_core::types::{Route, RouteStrategy};

use crate::processing::extract_entities;

/// Configuration for the adaptive router.
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    /// When false, every query is sent down `default_route`.
    pub enabled: bool,
    pub default_route: Route,
    /// Below this confidence level the router falls back to Standard.
    pub confidence_threshold: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_route: Route::Standard,
            confidence_threshold: 0.6,
        }
    }
}

/// Feature vector extracted from the raw query text.
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    pub word_count: usize,
    pub question_words: usize,
    pub conjunction_count: usize,
    pub comparison_indicators: bool,
    pub temporal_indicators: bool,
    pub negation_count: usize,
    pub entity_count: usize,
    pub avg_word_length: f64,
    pub has_multi_part: bool,
}

/// Outcome of the routing decision, including the rationale for audit.
pub struct RouteDecision {
    pub route: Route,
    pub confidence: f64,
    pub features: QueryFeatures,
    pub reason: String,
}

impl RouteDecision {
    fn fixed(route: Route) -> Self {
        Self {
            confidence: 1.0,
            features: QueryFeatures {
                word_count: 0,
                question_words: 0,
                conjunction_count: 0,
                comparison_indicators: false,
                temporal_indicators: false,
                negation_count: 0,
                entity_count: 0,
                avg_word_length: 0.0,
                has_multi_part: false,
            },
            reason: format!("Forced to {route:?} by caller"),
            route,
        }
    }
}

/// Classifies query complexity into Fast / Standard / Slow using lightweight
/// TF-IDF-style feature extraction. No ML model needed; a rule-based classifier
/// here achieves competitive accuracy with a fraction of the latency.
pub struct QueryRouter {
    pub config: RoutingConfig,
}

impl QueryRouter {
    pub fn new(config: RoutingConfig) -> Self {
        Self { config }
    }

    /// Main entry: classify the query and return a `RouteDecision`.
    pub fn classify(&self, query: &str) -> RouteDecision {
        if !self.config.enabled {
            return RouteDecision::fixed(self.config.default_route.clone());
        }

        let features = self.extract_features(query);
        let (route, confidence) = self.predict_route(&features);

        if confidence < self.config.confidence_threshold {
            return RouteDecision {
                route: Route::Standard,
                confidence,
                features,
                reason: "Low classifier confidence, defaulting to standard path".into(),
            };
        }

        let reason = format!("Classified as {route:?} with confidence {confidence:.2}");
        RouteDecision { route, confidence, features, reason }
    }

    /// Classify according to a caller-provided `RouteStrategy`.
    /// Auto delegates to the classifier; explicit strategies bypass it.
    pub fn classify_with_strategy(&self, query: &str, strategy: &RouteStrategy) -> RouteDecision {
        match strategy {
            RouteStrategy::Auto => self.classify(query),
            RouteStrategy::Fast => RouteDecision::fixed(Route::Fast),
            RouteStrategy::Standard => RouteDecision::fixed(Route::Standard),
            RouteStrategy::Slow => RouteDecision::fixed(Route::Slow),
        }
    }

    fn extract_features(&self, query: &str) -> QueryFeatures {
        let entity_count = extract_entities(query).len();
        QueryFeatures {
            word_count: query.split_whitespace().count(),
            question_words: count_question_words(query),
            conjunction_count: count_conjunctions(query),
            comparison_indicators: has_comparison_indicators(query),
            temporal_indicators: has_temporal_indicators(query),
            negation_count: count_negations(query),
            entity_count,
            avg_word_length: avg_word_length(query),
            has_multi_part: query.contains(" and ")
                || query.contains(" versus ")
                || query.contains(" compared to ")
                || query.contains(" vs "),
        }
    }

    fn predict_route(&self, features: &QueryFeatures) -> (Route, f64) {
        let score = self.compute_complexity_score(features);

        if score < 0.3 {
            (Route::Fast, 1.0 - score)
        } else if score > 0.7 {
            (Route::Slow, score)
        } else {
            // Standard band: confidence is highest at the midpoint (0.5) and
            // decreases toward the edges of the band.
            let dist_from_mid = (score - 0.5).abs();
            (Route::Standard, 0.5 + (0.2 - dist_from_mid).max(0.0))
        }
    }

    fn compute_complexity_score(&self, f: &QueryFeatures) -> f64 {
        let mut score = 0.0;
        score += (f.word_count as f64 / 30.0).min(1.0) * 0.20;
        score += (f.conjunction_count as f64 / 3.0).min(1.0) * 0.20;
        score += if f.comparison_indicators { 0.30 } else { 0.0 };
        score += if f.has_multi_part { 0.20 } else { 0.0 };
        score += (f.entity_count as f64 / 5.0).min(1.0) * 0.10;
        score.clamp(0.0, 1.0)
    }
}

fn count_question_words(query: &str) -> usize {
    let q = query.to_lowercase();
    ["what", "where", "when", "who", "which", "how", "why", "does", "is", "are", "can"]
        .iter()
        .filter(|&&w| q.split_whitespace().any(|t| t.trim_matches('?') == w))
        .count()
}

fn count_conjunctions(query: &str) -> usize {
    let q = query.to_lowercase();
    ["and", "or", "but", "nor", "yet", "so", "although", "while", "whereas"]
        .iter()
        .filter(|&&c| {
            q.split_whitespace()
                .any(|t| t.trim_matches(|ch: char| !ch.is_alphabetic()) == c)
        })
        .count()
}

fn has_comparison_indicators(query: &str) -> bool {
    let q = query.to_lowercase();
    ["compare", "versus", "vs", "difference", "contrast", "better", "worse", "similar", "different"]
        .iter()
        .any(|&kw| q.contains(kw))
}

fn has_temporal_indicators(query: &str) -> bool {
    let q = query.to_lowercase();
    ["before", "after", "since", "until", "during", "when", "timeline", "history", "changed", "latest"]
        .iter()
        .any(|&kw| q.contains(kw))
}

fn count_negations(query: &str) -> usize {
    let q = query.to_lowercase();
    ["not", "never", "no", "without", "except", "unless", "nor"]
        .iter()
        .filter(|&&neg| {
            q.split_whitespace()
                .any(|t| t.trim_matches(|ch: char| !ch.is_alphabetic()) == neg)
        })
        .count()
}

fn avg_word_length(query: &str) -> f64 {
    let words: Vec<&str> = query.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let total_chars: usize = words.iter().map(|w| w.len()).sum();
    total_chars as f64 / words.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> QueryRouter {
        QueryRouter::new(RoutingConfig::default())
    }

    #[test]
    fn simple_lookup_routes_to_fast() {
        let decision = router().classify("What is the lease term?");
        assert_eq!(decision.route, Route::Fast, "short factual query should be FAST");
    }

    #[test]
    fn complex_comparison_routes_to_slow() {
        let decision = router().classify(
            "Compare the indemnification clauses across all three contracts and identify conflicts between them, and also compare the pricing terms and tell me what is different.",
        );
        assert_eq!(decision.route, Route::Slow, "comparison multi-hop query should be SLOW");
    }

    #[test]
    fn section_reference_routes_to_standard() {
        let decision = router().classify("What does Section 4.2 say about payment terms?");
        assert!(
            matches!(decision.route, Route::Standard | Route::Fast),
            "section lookup should be STANDARD or FAST, got {:?}",
            decision.route
        );
    }

    #[test]
    fn low_confidence_defaults_to_standard() {
        let router = QueryRouter::new(RoutingConfig {
            confidence_threshold: 0.99,
            ..Default::default()
        });
        let decision = router.classify("What is the lease term?");
        assert_eq!(
            decision.route,
            Route::Standard,
            "below confidence threshold must default to STANDARD"
        );
    }

    #[test]
    fn disabled_router_returns_default_route() {
        let router = QueryRouter::new(RoutingConfig {
            enabled: false,
            default_route: Route::Fast,
            ..Default::default()
        });
        let decision = router.classify("Very complex multi-hop comparison query with contrasting clauses across documents");
        assert_eq!(decision.route, Route::Fast);
    }

    #[test]
    fn comparison_indicators_detected() {
        assert!(has_comparison_indicators("Compare these two documents"));
        assert!(!has_comparison_indicators("What is the lease term?"));
    }

    #[test]
    fn avg_word_length_computed_correctly() {
        let length = avg_word_length("abc de f");
        assert!((length - (3.0 + 2.0 + 1.0) / 3.0).abs() < 1e-6);
    }
}
