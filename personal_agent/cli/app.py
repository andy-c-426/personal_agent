from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from openai import OpenAI
from personal_agent.config import Config
from personal_agent.core.agent import Agent
from personal_agent.core.conversation import Conversation
from personal_agent.core.memory import MemoryManager
from personal_agent.tools.registry import ToolRegistry, Tool
from personal_agent.tools.kb_search import kb_search
from personal_agent.tools.web_search import web_search
from personal_agent.tools.kb_ingest import kb_ingest
from personal_agent.tools.kb_list import kb_list
from personal_agent.tools.kb_remove import kb_remove
from personal_agent.kb.retrieval import KBMetadata
from personal_agent.cli import display
import chromadb
from tavily import TavilyClient


def _setup_tools(retriever: KBMetadata, tavily_client: TavilyClient, config=None, llm_client=None) -> ToolRegistry:
    model = config.deepseek_model if config else "deepseek-chat"
    registry = ToolRegistry()
    registry.register(Tool(
        name="kb_search",
        description="Search the local knowledge base for relevant information.",
        function=lambda query: kb_search(query, retriever=retriever, llm_client=llm_client, model=model),
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
    return registry


def _handle_slash_command(cmd: str, args: str, retriever: KBMetadata, config: Config) -> bool:
    """Returns True if the REPL should continue, False to exit."""
    if cmd == "quit" or cmd == "exit":
        return False
    elif cmd == "help":
        display.print_help()
    elif cmd == "config":
        display.print_config(config)
    elif cmd == "ingest":
        if args:
            result = kb_ingest(args, retriever=retriever)
            display.console.print(f"Ingested: {result}")
        else:
            display.print_error("Usage: /ingest <path>")
    elif cmd == "kb":
        if args == "list":
            result = kb_list(retriever=retriever)
            display.console.print(result)
        elif args.startswith("remove "):
            source = args[7:]
            result = kb_remove(source, retriever=retriever)
            display.console.print(result)
        else:
            display.print_error("Usage: /kb list | /kb remove <id>")
    else:
        display.print_error(f"Unknown command: /{cmd}. Type /help for commands.")
    return True


def run(config: Config) -> None:
    config.agent_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    if config.kb_dir:
        config.kb_dir.mkdir(parents=True, exist_ok=True)

    # Check and migrate KB from old dimension to new (384 -> 1024)
    from personal_agent.kb.retrieval import check_and_migrate_kb
    check_and_migrate_kb(str(config.chroma_dir), str(config.kb_dir) if config.kb_dir else None)

    # Setup storage
    chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
    retriever = KBMetadata(chroma_client, collection_name="kb_main")

    # Setup Tavily
    tavily_client = TavilyClient(api_key=config.tavily_api_key)

    # Setup model client
    llm_client = OpenAI(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
    )

    # Setup tools
    registry = _setup_tools(retriever, tavily_client, config, llm_client)

    # Setup conversation
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
        kb_doc_count=retriever.document_count,
    )

    history_path = config.agent_dir / ".history"
    session = PromptSession(
        history=FileHistory(str(history_path)),
        style=Style.from_dict({"prompt": "bold green"}),
    )

    display.print_welcome(retriever.document_count)

    while True:
        try:
            user_input = session.prompt([("class:prompt", "> ")]).strip()
        except (EOFError, KeyboardInterrupt):
            display.console.print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            if not _handle_slash_command(cmd, args, retriever, config):
                break
            continue

        # Regular query
        try:
            response, tool_calls = agent.run(user_input, conversation)

            if tool_calls:
                for tc in tool_calls:
                    result_preview = tc["result"][:120].replace("\n", " ")
                    display.print_tool_status(tc["name"], result_preview)

            display.print_assistant_header()
            display.stream_markdown(response)

            # Persist conversation
            conv_path.write_text(conversation.to_json())

        except Exception as e:
            display.print_error(str(e))

    # Save on exit
    conv_path.write_text(conversation.to_json())
    display.console.print("Session saved.")
