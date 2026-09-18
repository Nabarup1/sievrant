#[macro_use]
extern crate napi_derive;

use napi::bindgen_prelude::*;
use sievrant::Corpus;
use sievrant_core::config::SievrantConfig;
use std::sync::Arc;

#[napi(js_name = "SievrantCorpus")]
pub struct SievrantCorpusJs {
    inner: Arc<Corpus>,
}

#[napi]
impl SievrantCorpusJs {
    #[napi(factory)]
    pub async fn create(name: String) -> Result<Self> {
        let config = SievrantConfig::default();
        let corpus = Corpus::create(&name, config)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(Self {
            inner: Arc::new(corpus),
        })
    }

    #[napi(factory)]
    pub async fn open(name: String) -> Result<Self> {
        let config = SievrantConfig::default();
        let corpus = Corpus::open(&name, config)
            .await
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(Self {
            inner: Arc::new(corpus),
        })
    }
}
