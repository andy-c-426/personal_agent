# personal_agent

A local-first personal AI assistant with a hybrid RAG pipeline, tool-use capabilities, and persistent memory. Runs entirely on your machine with a CLI interface.

## Architecture

```
~/.personal_agent/
├── chroma/              # Vector database (ChromaDB persistent)
├── kb/                  # Source documents for the knowledge base
├── memory.json          # Persistent user facts and preferences
├── conversation.json    # Current conversation state and rolling summary
└── .history             # REPL command history
```

The agent orchestrates three subsystems:

```
User Input
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   CLI REPL   │────▶│   Agent Loop     │────▶│   Tool Registry │
│ (prompt_tk)  │     │ (DeepSeek API)   │     │   6 tools       │
└──────────────┘     └──────────────────┘     └─────────────────┘
                             │                        │
                             ▼                        ▼
                     ┌──────────────┐     ┌──────────────────┐
                     │ Memory Mgr   │     │   RAG Pipeline   │
                     │ compression  │     │ hybrid search    │
                     │ + profile    │     │ + rerank         │
                     └──────────────┘     └──────────────────┘
```

### Knowledge Base (RAG Pipeline)

A production-quality hybrid retrieval pipeline using BGE-M3 embeddings:

- **Embeddings**: `BAAI/bge-m3` via FlagEmbedding — 1024-dim dense vectors and sparse lexical weights in a single model
- **Chunking**: Semantic sentence-boundary chunking with embedding-similarity merging and dynamic thresholds. Abbreviation-aware sentence splitting handles "Dr.", "U.S.", etc. Markdown heading metadata preserved per chunk
- **Hybrid retrieval**: Dense ANN search (ChromaDB top-20) + sparse BM25 search (in-memory index with k1=1.5, b=0.75) fused via Reciprocal Rank Fusion (k=60)
- **Reranking**: `BAAI/bge-reranker-v2-m3` cross-encoder reranks fused candidates to final top-k
- **Query rewriting**: LLM-powered query expansion before retrieval (configurable on/off)
- **Debug mode**: Per-query introspection showing dense hits, sparse hits, fused set, and final ranking

```
Query ──▶ Rewrite ──▶ Dense (Chroma) ──▶ RRF ──▶ Reranker ──▶ Results
                  └─▶ Sparse (BM25)  ──┘
```

### Agent Loop

- Tool-calling agent using DeepSeek API via OpenAI-compatible client
- Automatic loop detection: if the same tool is called 3+ consecutive times without progress, the agent stops and reports
- Context window compression: when token count exceeds 80% of the model context, older messages are summarized via LLM and injected as a rolling conversation summary
- All tool results and conversation state persisted to disk

### Memory System

Two-level memory architecture:

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Conversation memory** | In-memory + `conversation.json` | Rolling summary of chat history, compressed automatically |
| **User memory** | `memory.json` | Persistent facts, preferences, and reminders. Agent can add via `memory_add` tool |

User memories are injected into every system prompt so preferences persist across sessions.

## Setup

Requires Python 3.12+ and Poetry.

```bash
git clone https://github.com/andy-c-426/personal_agent.git
cd personal_agent
poetry install
```

### API Keys

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

### Optional

```bash
export PERSONAL_AGENT_DIR="$HOME/.personal_agent"  # Custom storage location (default: ~/.personal_agent)
```

### First Run

```bash
poetry run personal-agent
```

The agent starts with an empty knowledge base. Ingest documents to enable RAG:

```
> /ingest ~/Documents/notes/
```

## Usage

### Slash Commands

#### Search & Knowledge Base

| Command | Description |
|---------|-------------|
| `/search <query>` | Search KB directly (bypasses LLM, shows ranked results) |
| `/ingest <path>` | Add a file or directory to the KB. Warns before ingesting >50 files |
| `/kb list` | List all indexed documents |
| `/kb remove <path>` | Remove a document (confirmation required) |

#### Retrieval Controls

| Command | Description |
|---------|-------------|
| `/rag debug` | Toggle debug mode — shows dense/sparse/fused counts per search |
| `/rag config` | Show current RAG settings |
| `/rag top_k <n>` | Set number of results (default: 5) |
| `/rag reranker on\|off` | Toggle cross-encoder reranker |
| `/rag rewrite on\|off` | Toggle LLM query rewriting |

#### Memory

| Command | Description |
|---------|-------------|
| `/memory add <text>` | Remember a fact or preference |
| `/memory list` | List stored memories |
| `/memory remove <id>` | Delete a memory |

#### System

| Command | Description |
|---------|-------------|
| `/config` | Show model, directories, and configuration |
| `/help` | Show all commands |
| `/quit` | Exit |

### Agent Tools

The agent autonomously selects and calls these tools during conversations:

| Tool | Parameters | Description |
|------|------------|-------------|
| `kb_search` | `query` (required) | Hybrid semantic + keyword search with reranking |
| `web_search` | `query` (required) | Tavily web search (5 results) |
| `kb_ingest` | `path` (required) | Add files to the knowledge base |
| `kb_list` | none | List all indexed documents |
| `kb_remove` | `source` (required) | Remove a document from the KB |
| `memory_add` | `text` (required) | Persist a fact or preference |

### Example Session

```
> /ingest ~/notes/research/
Ingested: {"ingested": 14, "path": "/Users/me/notes/research"}

> /memory add i prefer concise answers with bullet points
Added: [1747242987654] i prefer concise answers with bullet points

> what does my research say about attention mechanisms?

Agent
Found 3 relevant chunks in kb_search.

Based on your research notes:

- **"Attention Is All You Need" (2017)** introduced the Transformer architecture that replaced recurrence with self-attention (research/transformers.md > Key Papers, chunk 2)
- Multi-head attention allows the model to focus on different representation subspaces simultaneously (research/transformers.md > Architecture, chunk 5)
- Your notes highlight the quadratic complexity O(n²) as the main limitation for long sequences (research/transformers.md > Limitations, chunk 8)
```

## File Support

| Format | Extension | Library |
|--------|-----------|---------|
| Markdown | `.md` | built-in |
| PDF | `.pdf` | pdfplumber |
| Word | `.docx`, `.doc` | python-docx |
| Plain text | `.txt` and others | built-in |

## Testing

```bash
# Fast tests (no model downloads)
poetry run pytest tests/ -v -m "not slow"

# Quality gate (requires model downloads, validates RAG pipeline)
poetry run pytest tests/test_rag_quality.py -v -s

# All tests including slow
poetry run pytest tests/ -v
```

### Quality Gate

`test_rag_quality.py` validates retrieval quality with 12 hand-crafted queries and relevance judgments:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **Recall@5** | >= 0.80 | At least one relevant chunk in top 5 |
| **MRR** | >= 0.60 | Mean Reciprocal Rank of first relevant chunk |
| **NDCG@5** | >= 0.70 | Ranking quality across multiple relevant chunks |

These tests use real BGE-M3 models and must pass before any RAG pipeline change.

### Test Markers

- `@pytest.mark.slow` — Tests that load real models (reranker, quality gate). Skipped with `-m "not slow"`

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DEEPSEEK_API_KEY` | (required) | DeepSeek API key for the agent LLM |
| `TAVILY_API_KEY` | (required) | Tavily API key for web search |
| `PERSONAL_AGENT_DIR` | `~/.personal_agent` | Storage directory for KB, memory, and state |

Internal defaults (not environment-configurable):

| Setting | Default | Description |
|---------|---------|-------------|
| Model | `deepseek-chat` | LLM model for agent and query rewriting |
| Base URL | `https://api.deepseek.com` | API endpoint |
| Max tool iterations | 10 | Max tool calls per user turn |
| Max same-tool calls | 3 | Consecutive same-tool calls before stopping |
| Context threshold | 80% | Trigger compression at this fraction of model context |
| Recent message count | 20 | Messages preserved during compression |
| RAG top_k | 5 | Default results per KB search (runtime-configurable) |
| Chunk size | 500 tokens | Target chunk size for semantic chunking |
| Chunk overlap | 50 tokens | Overlap between adjacent chunks |

## Project Structure

```
personal_agent/
├── cli/                # REPL, slash commands, display formatting
│   ├── app.py          # bootstrap(), run(), AppContext, tool setup
│   └── display.py      # Rich output: welcome, help, markdown streaming
├── core/               # Agent loop, conversation, memory
│   ├── agent.py        # Agent.run() — tool-calling loop
│   ├── conversation.py # Message dataclass, token counting, JSON persistence
│   ├── memory.py       # MemoryManager — context compression, system prompt
│   └── memory_store.py # MemoryStore — persistent user facts/preferences
├── kb/                 # Knowledge base: embedding, ingestion, retrieval
│   ├── embed.py        # Embedder (BGE-M3: dense + sparse lexical weights)
│   ├── ingest.py       # File parsing, semantic chunking, Chroma ingestion
│   └── retrieval.py    # Hybrid search, RRF fusion, reranker, query rewrite
├── tools/              # Tool implementations
│   ├── registry.py     # Tool dataclass, ToolRegistry
│   ├── kb_search.py    # KB search with rewrite, reranker, and debug
│   ├── web_search.py   # Tavily web search
│   ├── kb_ingest.py    # Ingest file/directory
│   ├── kb_list.py      # List indexed documents
│   └── kb_remove.py    # Remove a document
├── config.py           # Config dataclass from environment variables
└── main.py             # Entry point

tests/
├── conftest.py         # Shared fixtures and test data
├── test_agent.py       # Agent loop and tool-calling
├── test_cli.py         # Slash command parsing
├── test_config.py      # Configuration loading
├── test_conversation.py # Message serialization and token counting
├── test_embed.py       # Embedder integration
├── test_ingest.py      # Chunking and ingestion
├── test_memory.py      # Context compression
├── test_rag_quality.py # End-to-end RAG evaluation
├── test_retrieval.py   # Search, RRF, reranker
├── test_tool_registry.py # Tool registration and execution
└── test_tools.py       # Individual tool functions
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent LLM | DeepSeek (via OpenAI-compatible client) |
| Embeddings | BAAI/bge-m3 (FlagEmbedding) |
| Reranker | BAAI/bge-reranker-v2-m3 (sentence-transformers) |
| Vector DB | ChromaDB (persistent local storage) |
| Web Search | Tavily API |
| CLI | prompt_toolkit + Rich |
| Token Counting | tiktoken |
| PDF Parsing | pdfplumber |
| Word Parsing | python-docx |

## License

Apache 2.0. See [LICENSE](LICENSE).
