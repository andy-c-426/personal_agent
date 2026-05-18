# Personal Agent — Design Spec

## Overview

A CLI-based personal assistant agent deployed locally. Phase 1 provides a local knowledge base (ingest and search markdown, Word, PDFs) and web search (Tavily), powered by DeepSeek V4 Pro. Interaction is through a terminal REPL similar to Claude Code or Codex CLI.

## Architecture

Four layers:

```
CLI (prompt_toolkit)  →  Agent Core (ReAct)  →  Tools  →  Storage (Chroma)
```

**Project structure:**

```
personal_agent/
├── cli/            # Terminal UI — input loop, output rendering, markdown display
├── core/           # Agent loop — prompt construction, tool dispatch, conversation state
├── tools/          # Tool implementations — each tool is a function + JSON schema
├── kb/             # Knowledge base — ingestion pipeline, retrieval, embeddings
├── config.py       # API keys, model settings, KB directory path (from env/file)
├── main.py         # Entry point
└── ~/.personal_agent/  # Runtime data: Chroma DB, conversation history, settings
```

**Data flow:** User query → CLI → Agent Core → build prompt + tools → DeepSeek → text or tool call → if tool call, execute and feed result back → loop until final answer → CLI streams to user.

## CLI Layer

- **REPL**: `prompt_toolkit` with syntax highlighting, key bindings, command history
- **Streaming output**: token-by-token rendering via `rich` markdown
- **Slash commands**:
  - `/ingest <path>` — add file or directory to knowledge base
  - `/kb list` — list indexed documents
  - `/kb remove <id>` — remove a document
  - `/config` — print current configuration
  - `/help` — show available commands
  - `/quit` or `Ctrl+D` — exit
- **Tool call visualization**: when the agent runs a tool, show a brief status line (tool name + one-line summary)
- **Debug mode**: `--debug` flag prints full tool payloads and responses

## Agent Core

ReAct-style loop using DeepSeek's OpenAI-compatible tool-calling API:

1. Build system prompt (knowledge base stats, conversation summary)
2. Append conversation history and user query
3. Send to DeepSeek with tool definitions
4. Tool call returned → execute tool → append tool result → loop back to step 3
5. Text returned → stream to CLI, append to conversation

**Safety limits:** Max 10 iterations per query. If the model calls the same tool 3 consecutive times, break the loop and ask the user for guidance.

**Conversation model:** A `Conversation` dataclass with `add_message()`, `to_dicts()`, `trim_to_fit(max_tokens)`. Messages are system, user, assistant, or tool. Conversation is persisted to disk between sessions so context carries over.

**System prompt template:**

```
You are a personal assistant with access to a local knowledge base and web search.

Knowledge base: {num_docs} documents indexed. Use kb_search to find relevant local information.
Web search: Use web_search when you need information not in the knowledge base.

When answering:
- Prefer knowledge base results over web search when available
- Cite your sources (document name or URL)
- If both sources are used, distinguish between them
```

## Memory / Context Management

**Phase 1 (this spec): Sliding window + rolling summary.**

- Keep the last ~20 messages verbatim
- When total tokens exceed a threshold (~80% of model context), a background prompt asks the model to compress older messages into a 3–5 sentence running summary
- Summary is inserted into the system prompt; recent messages stay intact

**Phase 2 (future): Two-tier memory — summary + fact extraction.**

- Add a `memory` collection in Chroma alongside the KB collection
- Key facts, user preferences, and decisions are extracted from conversations and embedded into the memory collection
- On each query, semantically retrieve relevant past facts and inject into the system prompt
- Enables long-term memory across sessions without bloating the context window

## Tools

Each tool is a standalone function with JSON schema. Registered via a `ToolRegistry`.

| Tool | Description |
|---|---|
| `kb_search` | Semantic search over local knowledge base. Returns top-5 chunks with source filenames. |
| `web_search` | Tavily search. Returns title, URL, and cleaned content for each result. |
| `kb_ingest` | Ingest a document or directory. Parses, chunks, embeds, stores in Chroma. |
| `kb_list` | List all indexed documents with ID, name, chunk count. |
| `kb_remove` | Remove a document from Chroma by ID. |

**Execution:** All tools synchronous for Phase 1. Errors returned to the model as tool result messages so it can self-correct.

**Tavily:** `/search` endpoint with `search_depth: "advanced"`.

## Knowledge Base

**Ingestion pipeline:** `file/dir → detect type → parse → chunk → embed → store in Chroma`

- **Parsers**: `pdfplumber` (PDF), `python-docx` (Word), built-in `markdown` (.md), plain text fallback
- **Chunking**: recursive character splitter — `\n\n` → sentence → character. Target ~500 tokens per chunk, 50-token overlap. Chunks carry metadata (source file, chunk index, page number for PDFs)
- **Embeddings**: `sentence-transformers` with `all-MiniLM-L6-v2` (384-dim, runs locally, CPU-fast)
- **Storage**: Chroma, persisted at `~/.personal_agent/chroma/`
- **Deduplication**: re-ingesting the same file replaces existing chunks (tracked by file hash)
- **Retrieval**: top-5 chunks by cosine similarity, returned with source filename

## Configuration

Managed via a config file at `~/.personal_agent/config.yaml` and environment variables.

| Key | Description | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | API key (env var) | — |
| `DEEPSEEK_MODEL` | Model ID | `deepseek-chat` |
| `TAVILY_API_KEY` | Tavily API key (env var) | — |
| `KB_DIR` | Directory of files to index on startup | `~/.personal_agent/kb/` |

## Tech Stack

- **Language**: Python 3.12+
- **CLI**: `prompt_toolkit`, `rich`
- **Agent loop**: custom ReAct (no LangChain/LangGraph)
- **Vector store**: Chroma
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Document parsing**: `pdfplumber`, `python-docx`
- **Web search**: Tavily Python SDK
- **Model**: DeepSeek API (OpenAI-compatible endpoint)
- **Package management**: Poetry
