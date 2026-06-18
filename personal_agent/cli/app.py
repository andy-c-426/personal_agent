from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import NestedCompleter, PathCompleter
from openai import OpenAI
import chromadb
from tavily import TavilyClient

from personal_agent.config import Config
from personal_agent.core.agent import Agent
from personal_agent.core.conversation import Conversation
from personal_agent.core.memory import MemoryManager
from personal_agent.core.memory_store import MemoryStore
from personal_agent.tools.registry import ToolRegistry, Tool
from personal_agent.tools.kb_search import kb_search
from personal_agent.tools.web_search import web_search
from personal_agent.tools.kb_ingest import kb_ingest
from personal_agent.tools.kb_list import kb_list
from personal_agent.tools.kb_remove import kb_remove
from personal_agent.kb.retrieval import KBMetadata, check_and_migrate_kb
from personal_agent.tools.browser import (
    BrowserSession, browser_search, browser_navigate, browser_get_content,
    browser_click, browser_go_back,
)
from personal_agent.cli import display


@dataclass
class AppContext:
    config: Config
    chroma_client: chromadb.PersistentClient
    retriever: KBMetadata
    tavily_client: TavilyClient
    llm_client: OpenAI
    registry: ToolRegistry
    conversation: Conversation
    memory_manager: MemoryManager
    memory_store: MemoryStore
    agent: Agent
    rag_config: dict = field(default_factory=lambda: {
        "top_k": 5,
        "use_reranker": True,
        "use_rewrite": True,
        "debug": False,
    })
    browser_session: BrowserSession | None = None
    browser_config: dict = field(default_factory=lambda: {"headless": True})


def _setup_tools(
    retriever: KBMetadata,
    tavily_client: TavilyClient,
    config=None,
    llm_client=None,
    rag_config=None,
    memory_store=None,
    browser_session=None,
) -> ToolRegistry:
    model = config.deepseek_model if config else "deepseek-chat"
    rc = rag_config or {}
    bs = browser_session

    registry = ToolRegistry()
    registry.register(Tool(
        name="kb_search",
        description="Search the local knowledge base for relevant information.",
        function=lambda query: kb_search(
            query, retriever=retriever,
            llm_client=llm_client if rc.get("use_rewrite", True) else None,
            model=model, top_k=rc.get("top_k", 5),
            use_reranker=rc.get("use_reranker", True),
            use_rewrite=rc.get("use_rewrite", True),
            debug=rc.get("debug", False),
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ))
    registry.register(Tool(
        name="web_search",
        description="Search the web for information not in the knowledge base.",
        function=lambda query: web_search(query, client=tavily_client),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ))
    registry.register(Tool(
        name="kb_ingest",
        description="Ingest a file or directory into the knowledge base.",
        function=lambda path: kb_ingest(path, retriever=retriever),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to file or directory"}},
            "required": ["path"],
        },
    ))
    registry.register(Tool(
        name="kb_list",
        description="List all documents in the knowledge base.",
        function=lambda: kb_list(retriever=retriever),
        parameters={"type": "object", "properties": {}},
    ))
    registry.register(Tool(
        name="kb_remove",
        description="Remove a document from the knowledge base.",
        function=lambda source: kb_remove(source, retriever=retriever),
        parameters={
            "type": "object",
            "properties": {"source": {"type": "string", "description": "Source path of the document"}},
            "required": ["source"],
        },
    ))
    if memory_store:
        registry.register(Tool(
            name="memory_add",
            description="Remember a fact or preference for future conversations.",
            function=lambda text: _tool_memory_add(text, memory_store),
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Fact or preference to remember"}},
                "required": ["text"],
            },
        ))
    if bs is not None:
        registry.register(Tool(
            name="browser_search",
            description="Search the web using Google (falls back to DuckDuckGo if blocked). Returns structured results with titles, URLs, and snippets. Use this to find information not in the knowledge base.",
            function=lambda query: browser_search(query, session=bs),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        ))
        registry.register(Tool(
            name="browser_navigate",
            description="Navigate to a URL and read the page content. Use this after browser_search to read a specific result page in full.",
            function=lambda url: browser_navigate(url, session=bs),
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL to navigate to"}},
                "required": ["url"],
            },
        ))
        registry.register(Tool(
            name="browser_get_content",
            description="Re-read the current page content. Use this if the page has loaded dynamic content or you need to check for updates.",
            function=lambda: browser_get_content(session=bs),
            parameters={"type": "object", "properties": {}},
        ))
        registry.register(Tool(
            name="browser_click",
            description="Click a link on the current page by its visible text and read the resulting page.",
            function=lambda link_text: browser_click(link_text, session=bs),
            parameters={
                "type": "object",
                "properties": {"link_text": {"type": "string", "description": "Visible text of the link to click"}},
                "required": ["link_text"],
            },
        ))
        registry.register(Tool(
            name="browser_go_back",
            description="Go back to the previous page in browser history.",
            function=lambda: browser_go_back(session=bs),
            parameters={"type": "object", "properties": {}},
        ))
    return registry


def _tool_memory_add(text: str, memory_store: MemoryStore) -> str:
    import json
    item = memory_store.add(text)
    return json.dumps({"added": True, "id": item["id"], "count": memory_store.count()})


def _handle_slash_command(cmd: str, args: str, ctx: AppContext) -> bool:
    """Returns True if the REPL should continue, False to exit."""
    if cmd == "quit" or cmd == "exit":
        return False
    elif cmd == "help":
        display.print_help()
    elif cmd == "config":
        display.print_config(ctx.config)
    elif cmd == "search":
        if not args:
            display.print_error("Usage: /search <query>")
        else:
            _do_search(args, ctx)
    elif cmd == "rag":
        _handle_rag(args, ctx)
    elif cmd == "memory":
        _handle_memory(args, ctx)
    elif cmd == "ingest":
        if not args:
            display.print_error("Usage: /ingest <path>")
        else:
            _do_ingest(args, ctx)
    elif cmd == "kb":
        _handle_kb(args, ctx)
    elif cmd == "browser":
        _handle_browser(args, ctx)
    else:
        display.print_error(f"Unknown command: /{cmd}. Type /help for commands.")
    return True


def _do_search(query: str, ctx: AppContext) -> None:
    import json
    rc = ctx.rag_config
    results = ctx.retriever.search(query, n_results=rc["top_k"], use_reranker=rc["use_reranker"])

    if not results:
        display.console.print("[dim]No results found.[/dim]")
        return

    display.console.print(f"\n[bold]Search:[/bold] {query}")
    if rc["debug"]:
        debug_info = ctx.retriever.search_debug(query, n_results=rc["top_k"], use_reranker=rc["use_reranker"])
        display.console.print(f"  Dense hits: {len(debug_info['dense'])}, Sparse hits: {len(debug_info['sparse'])}, "
                             f"Fused: {len(debug_info['fused_ids'])}, Reranker: {debug_info['reranker_enabled']}")

    for i, r in enumerate(results, start=1):
        citation = f"{r['filename']}"
        if r["heading"]:
            citation += f" > {r['heading']}"
        citation += f" (chunk {r['chunk_index']})"
        display.console.print(f"  [bold]#{i}[/bold] {citation}")
        display.console.print(f"    {r['text'][:200].replace(chr(10), ' ')}...")
    display.console.print()


def _handle_rag(args: str, ctx: AppContext) -> None:
    rc = ctx.rag_config
    if args == "debug":
        rc["debug"] = not rc["debug"]
        display.console.print(f"RAG debug: [bold]{'on' if rc['debug'] else 'off'}[/bold]")
    elif args == "config":
        display.console.print(f"[bold]RAG Configuration:[/bold]")
        display.console.print(f"  top_k:        {rc['top_k']}")
        display.console.print(f"  reranker:     {'on' if rc['use_reranker'] else 'off'}")
        display.console.print(f"  query rewrite:{'on' if rc['use_rewrite'] else 'off'}")
        display.console.print(f"  debug:        {'on' if rc['debug'] else 'off'}")
        display.console.print("Usage: /rag <setting> <value>")
        display.console.print("  /rag top_k <n>, /rag reranker on|off, /rag rewrite on|off, /rag debug")
    elif args.startswith("top_k "):
        try:
            rc["top_k"] = int(args.split()[1])
            display.console.print(f"top_k = {rc['top_k']}")
        except ValueError:
            display.print_error("Usage: /rag top_k <number>")
    elif args == "reranker on":
        rc["use_reranker"] = True
        display.console.print("Reranker: on")
    elif args == "reranker off":
        rc["use_reranker"] = False
        display.console.print("Reranker: off")
    elif args == "rewrite on":
        rc["use_rewrite"] = True
        display.console.print("Query rewrite: on")
    elif args == "rewrite off":
        rc["use_rewrite"] = False
        display.console.print("Query rewrite: off")
    else:
        display.print_error("Usage: /rag [debug|config|top_k <n>|reranker on|off|rewrite on|off]")


def _handle_memory(args: str, ctx: AppContext) -> None:
    if args == "list" or not args:
        items = ctx.memory_store.list_all()
        if not items:
            display.console.print("[dim]No memories stored.[/dim]")
        else:
            display.console.print(f"[bold]Memories ({len(items)}):[/bold]")
            for item in items:
                display.console.print(f"  [{item['id'][:12]}] {item['text']}")
    elif args.startswith("add "):
        text = args[4:]
        item = ctx.memory_store.add(text)
        display.console.print(f"Added: [{item['id'][:12]}] {item['text']}")
    elif args.startswith("remove "):
        item_id = args[7:]
        if ctx.memory_store.remove(item_id):
            display.console.print(f"Removed.")
        else:
            display.print_error(f"Memory '{item_id}' not found.")
    else:
        display.print_error("Usage: /memory [list|add <text>|remove <id>]")


def _handle_kb(args: str, ctx: AppContext) -> None:
    if args == "list":
        result = kb_list(retriever=ctx.retriever)
        display.console.print(result)
    elif args.startswith("remove "):
        source = args[7:]
        # Confirm before destructive action
        display.console.print(f"[yellow]Remove '{source}' from knowledge base? (y/n)[/yellow]")
        try:
            confirm = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            display.console.print("Cancelled.")
            return
        if confirm not in ("y", "yes"):
            display.console.print("Cancelled.")
            return
        result = kb_remove(source, retriever=ctx.retriever)
        display.console.print(result)
        ctx.agent.kb_doc_count = ctx.retriever.document_count
    else:
        display.print_error("Usage: /kb list | /kb remove <id>")


def _handle_browser(args: str, ctx: AppContext) -> None:
    if args == "visible":
        ctx.browser_config["headless"] = not ctx.browser_config["headless"]
        if ctx.browser_session:
            ctx.browser_session.headless = ctx.browser_config["headless"]
        state = "headless" if ctx.browser_config["headless"] else "visible"
        display.console.print(f"Browser mode: [bold]{state}[/bold] (applies on next use)")
    elif args == "close":
        if ctx.browser_session:
            ctx.browser_session.close()
            display.console.print("Browser closed.")
        else:
            display.console.print("Browser not running.")
    else:
        display.print_error("Usage: /browser [visible|close]")


def _do_ingest(path_str: str, ctx: AppContext) -> None:
    from pathlib import Path
    p = Path(path_str).expanduser().resolve()

    if p.is_dir():
        # Guard against large accidental directory ingestion
        file_count = sum(1 for _ in p.rglob("*") if _.is_file() and not _.name.startswith("."))
        if file_count > 50:
            display.console.print(
                f"[yellow]Directory contains {file_count} files. Ingest all? (y/n)[/yellow]"
            )
            try:
                confirm = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                display.console.print("Cancelled.")
                return
            if confirm not in ("y", "yes"):
                display.console.print("Cancelled.")
                return

    result = kb_ingest(path_str, retriever=ctx.retriever)
    display.console.print(f"Ingested: {result}")
    ctx.agent.kb_doc_count = ctx.retriever.document_count


def bootstrap(config: Config) -> AppContext:
    config.agent_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    if config.kb_dir:
        config.kb_dir.mkdir(parents=True, exist_ok=True)

    check_and_migrate_kb(str(config.chroma_dir), str(config.kb_dir) if config.kb_dir else None)

    chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
    retriever = KBMetadata(chroma_client, collection_name="kb_main")

    tavily_client = TavilyClient(api_key=config.tavily_api_key)

    llm_client = OpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
    )

    memory_store = MemoryStore(config.agent_dir / "memory.json")

    browser_session = BrowserSession(headless=True)
    browser_config = {"headless": True}

    rag_config = {
        "top_k": 5,
        "use_reranker": True,
        "use_rewrite": True,
        "debug": False,
    }

    registry = _setup_tools(
        retriever, tavily_client, config, llm_client,
        rag_config=rag_config, memory_store=memory_store,
        browser_session=browser_session,
    )

    conv_path = config.agent_dir / "conversation.json"
    if conv_path.exists():
        conversation = Conversation.from_json(conv_path.read_text())
    else:
        conversation = Conversation()

    memory_manager = MemoryManager(config, client=llm_client)

    agent = Agent(
        config,
        llm_client,
        registry,
        memory_manager=memory_manager,
        memory_store=memory_store,
        kb_doc_count=retriever.document_count,
    )

    return AppContext(
        config=config,
        chroma_client=chroma_client,
        retriever=retriever,
        tavily_client=tavily_client,
        llm_client=llm_client,
        registry=registry,
        conversation=conversation,
        memory_manager=memory_manager,
        memory_store=memory_store,
        agent=agent,
        rag_config=rag_config,
        browser_session=browser_session,
        browser_config=browser_config,
    )


def run(config: Config) -> None:
    ctx = bootstrap(config)

    commands = {
        "ingest": PathCompleter(),
        "kb": {"list": None, "remove": None},
        "search": None,
        "rag": {"debug": None, "config": None, "top_k": None, "reranker": {"on": None, "off": None}, "rewrite": {"on": None, "off": None}},
        "memory": {"add": None, "list": None, "remove": None},
        "browser": {"visible": None, "close": None},
        "config": None,
        "help": None,
        "quit": None,
        "exit": None,
    }

    history_path = ctx.config.agent_dir / ".history"
    session = PromptSession(
        completer=NestedCompleter.from_nested_dict(commands),
        complete_while_typing=True,
        history=FileHistory(str(history_path)),
        style=Style.from_dict({"prompt": "bold green"}),
    )

    display.print_welcome(ctx.retriever.document_count)

    conv_path = ctx.config.agent_dir / "conversation.json"

    try:
        while True:
            try:
                user_input = session.prompt([("class:prompt", "> ")]).strip()
            except (EOFError, KeyboardInterrupt):
                display.console.print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                cmd = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                if not _handle_slash_command(cmd, args, ctx):
                    break
                continue

            try:
                response, tool_calls = ctx.agent.run(user_input, ctx.conversation)

                if tool_calls:
                    for tc in tool_calls:
                        result_preview = tc["result"][:120].replace("\n", " ")
                        display.print_tool_status(tc["name"], result_preview)

                display.print_assistant_header()
                display.stream_markdown(response)

                ctx.agent.kb_doc_count = ctx.retriever.document_count

                conv_path.write_text(ctx.conversation.to_json())

            except Exception as e:
                display.print_error(str(e))
    finally:
        if ctx.browser_session:
            ctx.browser_session.close()
        conv_path.write_text(ctx.conversation.to_json())
        display.console.print("Session saved.")
