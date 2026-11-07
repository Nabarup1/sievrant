use async_trait::async_trait;
use sievrant_acl::resolver::AclResolver;
use sievrant_core::{error::Result, types::EvidencePacket};
use sievrant_orchestrator::pipeline::SemanticCache;

/// Cache disabled. A semantic cache would serve packets computed under an
/// older calibration, so it stays off until it can key on the calibration set.
pub struct NoopCache;

#[async_trait]
impl SemanticCache for NoopCache {
    async fn lookup(
        &self,
        _query: &str,
        _corpus: &str,
        _user_id: &Option<String>,
    ) -> Result<Option<EvidencePacket>> {
        Ok(None)
    }

    async fn store(
        &self,
        _query: &str,
        _corpus: &str,
        _user_id: &Option<String>,
        _packet: &EvidencePacket,
    ) -> Result<()> {
        Ok(())
    }
}

/// Callers that assert no ACL groups see only `public` documents.
pub struct PublicAclResolver;

#[async_trait]
impl AclResolver for PublicAclResolver {
    async fn resolve(&self, _user_id: Option<&str>, _corpus: &str) -> Result<Vec<String>> {
        Ok(vec!["public".into()])
    }
}
