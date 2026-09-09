# LangChain Integration Guide

Sievrant provides a drop-in `SievrantRetriever` that implements LangChain's `BaseRetriever` interface. Replace your existing retriever with `SievrantRetriever` to get statistical coverage guarantees, hybrid 4-way retrieval, and a tamper-evident audit trail with zero changes to the rest of your chain.

---

## Python (LangChain)

### Installation

```bash
pip install "sievrant[langchain]"
```

### Basic drop-in replacement

```python
# Before (standard LangChain retriever)
from langchain_community.vectorstores import Chroma
retriever = Chroma.from_documents(docs, embeddings).as_retriever(k=5)

# After (Sievrant drop-in)
from sievrant.integrations.langchain import SievrantRetriever
retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="my-corpus",
    coverage=0.95,
)
# Everything else in your chain stays identical
```

### Full RetrievalQA example

```python
from sievrant.integrations.langchain import SievrantRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-contracts",
    coverage=0.95,          # Conformal coverage target
    k=5,                    # Max chunks to return (pre-selected by Sievrant's math)
    acl_groups=["legal"],   # Only return chunks this user can access
)

prompt = PromptTemplate(
    template="""Use only the following context to answer the question.
If the context does not contain the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

result = chain.invoke({"query": "What is the notice period for lease termination?"})
print(result["result"])
for doc in result["source_documents"]:
    print(f"  [{doc.metadata['retriever']}] {doc.metadata['source_doc']} — score: {doc.metadata['rrf_score']:.4f}")
```

### Document metadata

`SievrantRetriever` maps each `EvidenceChunk` to a LangChain `Document` with full metadata preserved:

```python
docs = await retriever.aget_relevant_documents("What is the rent?")
for doc in docs:
    doc.page_content                         # chunk text
    doc.metadata["chunk_id"]                 # globally unique chunk ID
    doc.metadata["source_doc"]               # source filename
    doc.metadata["page"]                     # page number
    doc.metadata["retriever"]                # which index (vector, bm25, etc.)
    doc.metadata["rrf_score"]                # fusion score
    doc.metadata["rerank_score"]             # cross-encoder score
    doc.metadata["nonconformity_score"]      # conformal trust score
    doc.metadata["acl_groups"]              # access control groups
    doc.metadata["sievrant_decision"]        # ANSWERED / ABSTAINED
    doc.metadata["sievrant_query_id"]        # UUID for audit correlation
```

### Handling abstentions

When Sievrant abstains (insufficient evidence), by default the retriever returns an empty list. You can configure it to raise instead:

```python
from sievrant.integrations.langchain import SievrantRetriever, SievrantAbstainedError

retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-contracts",
    raise_on_abstain=True,
)

try:
    docs = await retriever.aget_relevant_documents("irrelevant question")
except SievrantAbstainedError as e:
    # e.query_id -- the UUID for this request
    # e.audit    -- the full audit block
    print("Sievrant could not find sufficient evidence. Escalating to human review.")
```

### ConversationalRetrievalChain

```python
from sievrant.integrations.langchain import SievrantRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="support-kb",
)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=retriever,
    return_source_documents=True,
)

chat_history = []
while True:
    question = input("You: ")
    result = chain.invoke({"question": question, "chat_history": chat_history})
    print(f"Bot: {result['answer']}")
    chat_history.append((question, result["answer"]))
```

---

## Node.js / TypeScript (LangChain.js)

### Installation

```bash
npm install sievrant @langchain/core langchain
```

### Basic drop-in replacement

```typescript
import { SievrantRetriever } from 'sievrant/integrations/langchain';

const retriever = new SievrantRetriever({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key',
  corpus: 'legal-contracts',
  coverage: 0.95,
  k: 5,
});

// Drop into any LangChain.js chain
const docs = await retriever.getRelevantDocuments('What is the termination notice?');
```

### With LCEL (LangChain Expression Language)

```typescript
import { SievrantRetriever } from 'sievrant/integrations/langchain';
import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';

const retriever = new SievrantRetriever({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key',
  corpus: 'legal-contracts',
});

const prompt = ChatPromptTemplate.fromMessages([
  ['system', 'Answer based only on this context:\n{context}'],
  ['human', '{question}'],
]);

const chain = RunnableSequence.from([
  {
    context: retriever,
    question: new RunnablePassthrough(),
  },
  prompt,
  new ChatOpenAI({ model: 'gpt-4o' }),
  new StringOutputParser(),
]);

const answer = await chain.invoke('What is the monthly rent?');
console.log(answer);
```

---

## Migrating an existing LangChain app

| Before | After |
|---|---|
| `Chroma(...).as_retriever()` | `SievrantRetriever(base_url=..., corpus=...)` |
| `FAISS.from_documents(...).as_retriever()` | `SievrantRetriever(...)` |
| `PineconeVectorStore(...).as_retriever()` | `SievrantRetriever(...)` |
| Manual top-k | Sievrant Submodular Selection (auto) |
| No abstention | `raise_on_abstain=True` for safe handling |
| No audit trail | Full `session_hash` in metadata |

The rest of your chain — prompts, LLMs, memory, tools — requires **zero changes**.
