# Sievrant Reproducible Benchmarks

This directory contains scripts to independently verify Sievrant's retrieval performance against baseline dense-vector retrieval systems.

## Prerequisites

1. Start the Sievrant backend server (this is required to test the Sievrant pipeline):
```bash
cd ..
cargo run --release -p sievrant-server
```

2. In a separate terminal, install the benchmark dependencies:
```bash
pip install -r requirements-benchmark.txt
pip install -e ../sdks/python
```

## Running the BEIR Benchmark

The script downloads a BEIR dataset (e.g., SciFact), embeds it using `BAAI/bge-small-en-v1.5`, runs a local FAISS index for the baseline, and then runs the same corpus through the Sievrant hybrid pipeline.

```bash
python beir_benchmark.py --dataset scifact
```

Options:
- `--dataset`: `scifact` (default, ~5k docs), `fiqa`, `nfcorpus`, etc.
- `--model`: Defaults to `BAAI/bge-small-en-v1.5`.

The script outputs a clean comparison table comparing NDCG@10 and Recall@100.
