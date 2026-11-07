//! Embed Sievrant in a Rust program.
//!
//!     cargo run -p sievrant --example quickstart
//!
//! Providers come from the environment (see .env.example). With no provider
//! keys set it runs fully offline using the `local` embedder and no LLM, so
//! the decision is EVIDENCE_ONLY and no answer is generated.

use sievrant::{Corpus, Document, DocumentMetadata, RetrievalConfig, SievrantConfig};

#[tokio::main]
async fn main() -> sievrant::Result<()> {
    let mut config = SievrantConfig::load()?;
    config.storage.data_dir = "data/example".into();

    let corpus = match Corpus::open("contracts", config.clone()).await {
        Ok(c) => c,
        Err(_) => Corpus::create("contracts", config).await?,
    };

    corpus
        .ingest(vec![Document {
            id: "lease-001".into(),
            content: "The lease term is 60 months starting January 1, 2025. \
                      Monthly rent is 4,500 USD, due on the first of each month."
                .into(),
            metadata: DocumentMetadata {
                title: Some("Office lease".into()),
                ..Default::default()
            },
            acl_groups: vec![], // empty = "public"
        }])
        .await?;

    let packet = corpus
        .retrieve(
            "How long is the lease term?",
            RetrievalConfig {
                coverage: 0.95,
                ..Default::default()
            },
        )
        .await?;

    println!("decision:  {:?}", packet.decision);
    println!(
        "guarantee: {:?} ({})",
        packet.coverage_guarantee, packet.guarantee.statement
    );
    for e in &packet.evidence {
        println!("evidence:  [{}] {}", e.chunk_id, e.text);
    }
    Ok(())
}
