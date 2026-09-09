import pytest
import asyncio
from sievrant_native import SievrantCorpus
import tempfile
import os
import shutil

@pytest.mark.asyncio
async def test_create_and_open_corpus():
    # Use a unique name for the test
    corpus_name = "test_python_native_corpus"
    
    # Cleanup before run
    if os.path.exists(f"data/{corpus_name}"):
        shutil.rmtree(f"data/{corpus_name}")

    try:
        # Create corpus
        corpus = await SievrantCorpus.create(corpus_name)
        assert corpus is not None
        
        # Open corpus
        del corpus
        import gc
        gc.collect()
        reopened = await SievrantCorpus.open(corpus_name)
        assert reopened is not None
    finally:
        # Cleanup
        if os.path.exists(f"data/{corpus_name}"):
            shutil.rmtree(f"data/{corpus_name}")
