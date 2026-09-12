use dashmap::DashMap;
use sievrant_core::error::{Result, SievrantError};

#[async_trait::async_trait]
pub trait AclResolver: Send + Sync {
    /// Resolve a user ID (if any) to their set of allowed ACL groups for a given corpus.
    async fn resolve(&self, user_id: Option<&str>, corpus: &str) -> Result<Vec<String>>;
}

/// An in-memory ACL resolver for user group resolution.
/// It implements pre-retrieval access control, mapping user IDs to their allowed groups.
#[derive(Default)]
pub struct InMemoryAclResolver {
    /// Maps user IDs to their allowed ACL groups
    user_groups: DashMap<String, Vec<String>>,
}

impl InMemoryAclResolver {
    pub fn new() -> Self {
        Self {
            user_groups: DashMap::new(),
        }
    }

    /// Register a user's ACL groups (called during setup or from an IAM integration)
    pub fn register_user(&self, user_id: &str, groups: Vec<String>) {
        self.user_groups.insert(user_id.to_string(), groups);
    }

    /// Check if a user has access to a specific set of ACL groups
    pub fn has_access(&self, user_id: &str, required_groups: &[String]) -> bool {
        if let Some(user_acls) = self.user_groups.get(user_id) {
            // Admin bypasses all checks
            if user_acls.contains(&"admin".to_string()) {
                return true;
            }
            
            // Check if user has AT LEAST ONE of the required groups
            for req in required_groups {
                if user_acls.contains(req) {
                    return true;
                }
            }
        }
        false
    }
}

#[async_trait::async_trait]
impl AclResolver for InMemoryAclResolver {
    async fn resolve(&self, user_id: Option<&str>, _corpus: &str) -> Result<Vec<String>> {
        match user_id {
            Some(uid) => {
                println!("DEBUG ACL: looking for '{}', map contains: {:?}", uid, self.user_groups.iter().map(|kv| kv.key().clone()).collect::<Vec<_>>());
                if let Some(groups) = self.user_groups.get(uid) {
                    if groups.is_empty() {
                        return Err(SievrantError::AccessDenied {
                            reason: format!("User '{}' has no registered ACL groups", uid)
                        });
                    }
                    Ok(groups.clone())
                } else {
                    Err(SievrantError::AccessDenied {
                        reason: format!("User '{}' not found in ACL registry", uid)
                    })
                }
            }
            None => {
                // Default policy for anonymous users
                Ok(vec!["public".to_string()])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_public_policy() {
        let resolver = InMemoryAclResolver::new();
        let groups = resolver.resolve(None, "corp").await.unwrap();
        assert_eq!(groups, vec!["public"]);
    }

    #[tokio::test]
    async fn unregistered_user_denied() {
        let resolver = InMemoryAclResolver::new();
        let err = resolver.resolve(Some("unknown_user"), "corp").await.unwrap_err();
        assert!(matches!(err, SievrantError::AccessDenied { .. }));
    }

    #[tokio::test]
    async fn registered_user_has_access() {
        let resolver = InMemoryAclResolver::new();
        resolver.register_user("alice", vec!["engineering".to_string(), "public".to_string()]);
        
        assert!(resolver.has_access("alice", &["engineering".to_string()]));
        assert!(resolver.has_access("alice", &["public".to_string()]));
        assert!(!resolver.has_access("alice", &["finance".to_string()]));
    }

    #[tokio::test]
    async fn admin_override_works() {
        let resolver = InMemoryAclResolver::new();
        resolver.register_user("bob", vec!["admin".to_string()]);
        
        // Admin has access to anything
        assert!(resolver.has_access("bob", &["engineering".to_string()]));
        assert!(resolver.has_access("bob", &["top_secret".to_string()]));
    }
}
