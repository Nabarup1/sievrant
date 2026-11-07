//! API key authentication.
//!
//! Keys come from configuration, never from code:
//! - `SIEVRANT_API_KEYS=tenant_a:key1,tenant_b:key2`
//! - `SIEVRANT_API_KEY=key` (single key for the `default` tenant)
//! - `SIEVRANT_API_KEY_HASHES=tenant_a:<sha256 hex>` (raw keys never stored)
//!
//! Only SHA-256 hashes are kept in memory. The server refuses to start when
//! no key is configured, unless `SIEVRANT_AUTH_DISABLED=true` (local dev only).

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use sievrant_core::config::ServerConfig;

/// Raw keys shorter than this are rejected at startup.
pub const MIN_KEY_LEN: usize = 16;

pub fn hash_key(raw: &str) -> String {
    let mut h = Sha256::new();
    h.update(raw.as_bytes());
    format!("{:x}", h.finalize())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    Missing,
    Invalid,
}

#[derive(Debug, Clone, Default)]
pub struct ApiKeys {
    by_hash: HashMap<String, String>,
    disabled: bool,
}

impl ApiKeys {
    pub fn from_config(cfg: &ServerConfig) -> Result<Self, String> {
        let mut keys = Self {
            by_hash: HashMap::new(),
            disabled: cfg.auth_disabled,
        };

        if let Some(k) = cfg.api_key.as_ref().map(|s| s.expose().trim()) {
            if !k.is_empty() {
                keys.add_raw(sievrant::DEFAULT_TENANT, k, "SIEVRANT_API_KEY")?;
            }
        }
        if let Some(list) = &cfg.api_keys {
            for pair in list
                .expose()
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                let (tenant, key) = pair
                    .split_once(':')
                    .ok_or("SIEVRANT_API_KEYS entries must look like tenant:key")?;
                keys.add_raw(tenant.trim(), key.trim(), "SIEVRANT_API_KEYS")?;
            }
        }
        if let Some(list) = &cfg.api_key_hashes {
            for pair in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (tenant, hash) = pair
                    .split_once(':')
                    .ok_or("SIEVRANT_API_KEY_HASHES entries must look like tenant:sha256hex")?;
                let hash = hash.trim().to_ascii_lowercase();
                if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err(format!(
                        "SIEVRANT_API_KEY_HASHES: hash for tenant '{tenant}' is not 64 hex characters"
                    ));
                }
                keys.add_hash(tenant.trim(), hash)?;
            }
        }

        if keys.by_hash.is_empty() && !keys.disabled {
            return Err(
                "no API keys configured. Set SIEVRANT_API_KEYS=tenant:key[,tenant:key], \
                 SIEVRANT_API_KEY=key or SIEVRANT_API_KEY_HASHES=tenant:sha256hex \
                 (or SIEVRANT_AUTH_DISABLED=true for local development only)"
                    .into(),
            );
        }
        Ok(keys)
    }

    /// Build from raw `(tenant, key)` pairs. Used by tests.
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        let mut keys = Self::default();
        for (t, k) in pairs {
            keys.by_hash.insert(hash_key(k), t.to_string());
        }
        keys
    }

    /// Accept every request as the `default` tenant. Used by tests and dev mode.
    pub fn disabled() -> Self {
        Self {
            by_hash: HashMap::new(),
            disabled: true,
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    pub fn tenant_count(&self) -> usize {
        let mut t: Vec<&String> = self.by_hash.values().collect();
        t.sort();
        t.dedup();
        t.len()
    }

    fn add_raw(&mut self, tenant: &str, key: &str, source: &str) -> Result<(), String> {
        if key.len() < MIN_KEY_LEN {
            return Err(format!(
                "{source}: key for tenant '{tenant}' is shorter than {MIN_KEY_LEN} characters; \
                 generate one with `openssl rand -hex 32`"
            ));
        }
        let lower = key.to_ascii_lowercase();
        if ["your-", "changeme", "change-me", "example", "placeholder"]
            .iter()
            .any(|p| lower.contains(p))
        {
            return Err(format!(
                "{source}: key for tenant '{tenant}' looks like a placeholder from an example file; \
                 generate a real one with `openssl rand -hex 32`"
            ));
        }
        self.add_hash(tenant, hash_key(key))
    }

    fn add_hash(&mut self, tenant: &str, hash: String) -> Result<(), String> {
        sievrant::validate_name("tenant", tenant).map_err(|e| e.to_string())?;
        match self.by_hash.insert(hash, tenant.to_string()) {
            Some(prev) if prev != tenant => Err(format!(
                "the same API key is assigned to tenants '{prev}' and '{tenant}'"
            )),
            _ => Ok(()),
        }
    }

    /// Resolve an `Authorization` header value to a tenant id.
    pub fn authenticate(&self, authorization: Option<&str>) -> Result<String, AuthError> {
        if self.disabled {
            return Ok(sievrant::DEFAULT_TENANT.to_string());
        }
        let header = authorization.ok_or(AuthError::Missing)?;
        let token = header
            .strip_prefix("Bearer ")
            .or_else(|| header.strip_prefix("bearer "))
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .ok_or(AuthError::Missing)?;
        self.by_hash
            .get(&hash_key(token))
            .cloned()
            .ok_or(AuthError::Invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::config::Secret;

    fn cfg() -> ServerConfig {
        ServerConfig::default()
    }

    #[test]
    fn refuses_to_start_without_keys() {
        assert!(ApiKeys::from_config(&cfg()).is_err());
    }

    #[test]
    fn auth_disabled_allows_empty_key_set() {
        let mut c = cfg();
        c.auth_disabled = true;
        let k = ApiKeys::from_config(&c).unwrap();
        assert_eq!(k.authenticate(None).unwrap(), "default");
    }

    #[test]
    fn parses_tenant_pairs_and_rejects_short_keys() {
        let mut c = cfg();
        c.api_keys = Some(Secret::new(
            "acme:0123456789abcdef0123, beta:fedcba9876543210fedc",
        ));
        let k = ApiKeys::from_config(&c).unwrap();
        assert_eq!(
            k.authenticate(Some("Bearer 0123456789abcdef0123")).unwrap(),
            "acme"
        );
        assert_eq!(
            k.authenticate(Some("Bearer fedcba9876543210fedc")).unwrap(),
            "beta"
        );
        assert_eq!(k.authenticate(Some("Bearer nope")), Err(AuthError::Invalid));
        assert_eq!(k.authenticate(Some("Basic abc")), Err(AuthError::Missing));
        assert_eq!(k.authenticate(None), Err(AuthError::Missing));

        c.api_keys = Some(Secret::new("acme:sk-test"));
        assert!(ApiKeys::from_config(&c).is_err());

        let mut c = cfg();
        c.api_key = Some(Secret::new("sk-your-server-api-key-here"));
        assert!(
            ApiKeys::from_config(&c).is_err(),
            "example-file placeholder must be rejected"
        );
    }

    #[test]
    fn rejects_bad_tenant_names_and_hashes() {
        let mut c = cfg();
        c.api_keys = Some(Secret::new("../etc:0123456789abcdef0123"));
        assert!(ApiKeys::from_config(&c).is_err());

        let mut c = cfg();
        c.api_key_hashes = Some("acme:xyz".into());
        assert!(ApiKeys::from_config(&c).is_err());

        let mut c = cfg();
        c.api_key_hashes = Some(format!("acme:{}", hash_key("0123456789abcdef0123")));
        let k = ApiKeys::from_config(&c).unwrap();
        assert_eq!(
            k.authenticate(Some("Bearer 0123456789abcdef0123")).unwrap(),
            "acme"
        );
    }

    #[test]
    fn same_key_for_two_tenants_is_an_error() {
        let mut c = cfg();
        c.api_keys = Some(Secret::new("a:0123456789abcdef0123,b:0123456789abcdef0123"));
        assert!(ApiKeys::from_config(&c).is_err());
    }
}
