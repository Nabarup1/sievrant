use pyo3::prelude::*;
use sievrant::Corpus;
use sievrant_core::config::SievrantConfig;
use std::sync::Arc;

/// A wrapper around the Sievrant embedded Rust engine.
#[pyclass]
pub struct SievrantCorpus {
    inner: Arc<Corpus>,
}

#[pymethods]
impl SievrantCorpus {
    /// Create a new local embedded Sievrant corpus.
    #[staticmethod]
    fn create(py: Python<'_>, name: String) -> PyResult<&PyAny> {
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let config = SievrantConfig::default();
            let corpus = Corpus::create(&name, config)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(SievrantCorpus {
                inner: Arc::new(corpus),
            })
        })
    }

    /// Open an existing local embedded Sievrant corpus.
    #[staticmethod]
    fn open(py: Python<'_>, name: String) -> PyResult<&PyAny> {
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let config = SievrantConfig::default();
            let corpus = Corpus::open(&name, config)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(SievrantCorpus {
                inner: Arc::new(corpus),
            })
        })
    }
}

/// The core sievrant native python module
#[pymodule]
fn sievrant_native(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SievrantCorpus>()?;
    Ok(())
}
