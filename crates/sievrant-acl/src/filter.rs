/// A generic metadata filter predicate representation that can be converted
/// into the specific query DSL of the underlying vector database (e.g., Qdrant, Milvus, Pinecone).
/// 
/// This ensures ACLs are pushed down to the query level and applied BEFORE scoring.
#[derive(Debug, Clone)]
pub enum AclFilter {
    /// Allow if the document's ACL groups intersect with the allowed groups.
    In {
        key: String,
        allowed_groups: Vec<String>,
    },
    /// For admin bypass: allow all.
    AllowAll,
}

impl AclFilter {
    /// Build a filter that enforces access control.
    /// If the user is an admin (has the "admin" group), it returns AllowAll.
    /// Otherwise, it builds a filter requiring at least one of the user's groups
    /// to match the document's `acl_groups` field.
    pub fn build(user_groups: &[String]) -> Self {
        if user_groups.contains(&"admin".to_string()) {
            Self::AllowAll
        } else {
            Self::In {
                key: "acl_groups".to_string(),
                allowed_groups: user_groups.to_vec(),
            }
        }
    }

    /// Helper to manually test if a document's groups pass the filter.
    /// In production, this logic is handled by the vector DB's filter engine natively.
    pub fn matches(&self, document_groups: &[String]) -> bool {
        match self {
            Self::AllowAll => true,
            Self::In { allowed_groups, .. } => {
                // Return true if there is any intersection between allowed_groups and document_groups
                document_groups.iter().any(|g| allowed_groups.contains(g))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_matches_allowed_groups() {
        let filter = AclFilter::build(&["engineering".to_string(), "public".to_string()]);
        
        assert!(filter.matches(&["engineering".to_string()]), "should match direct group");
        assert!(filter.matches(&["public".to_string()]), "should match public");
        assert!(filter.matches(&["engineering".to_string(), "finance".to_string()]), "should match if document has one allowed group");
        assert!(!filter.matches(&["finance".to_string(), "hr".to_string()]), "should not match if no intersection");
    }

    #[test]
    fn admin_filter_allows_all() {
        let filter = AclFilter::build(&["engineering".to_string(), "admin".to_string()]);
        assert!(matches!(filter, AclFilter::AllowAll));
        
        assert!(filter.matches(&["top_secret".to_string()]), "admin can see top secret");
        assert!(filter.matches(&["finance".to_string()]), "admin can see finance");
    }
}
