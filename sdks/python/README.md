# Sievrant Python SDK

> Statistically audited retrieval for enterprise AI

```bash
pip install sievrant
```

```python
from sievrant import SievrantClient

async with SievrantClient(base_url="http://localhost:8080", api_key="sk-...") as client:
    result = await client.query(corpus="my-corpus", query="What is the rent?")
    print(result.decision)   # "ANSWERED" or "ABSTAINED"
    print(result.answer)     # answer string or None
```

Full documentation: https://github.com/Nabarup1/sievrant/tree/main/docs
