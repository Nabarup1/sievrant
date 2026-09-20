import pytest
pytest.importorskip("sievrant_native")

import asyncio
from sievrant_native import SievrantCorpus
import tempfile
import os

@pytest.mark.asyncio
async def test_create_and_open_corpus():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SIEVRANT_STORAGE__DATA_DIR"] = tmpdir
        corpus_name = "test_python_native_corpus"
        
        # Create corpus
        corpus = await SievrantCorpus.create(corpus_name)
        assert corpus is not None
        
        # Open corpus
        del corpus
        import gc
        gc.collect()
        reopened = await SievrantCorpus.open(corpus_name)
        assert reopened is not None
