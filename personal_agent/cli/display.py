from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

console = Console()


def print_welcome(kb_count: int) -> None:
    console.print(Panel.fit(
        f"[bold]Personal Agent[/bold]\n"
        f"Knowledge base: {kb_count} documents indexed\n"
        f"Type /help for commands, /quit to exit",
        border_style="blue",
    ))


def print_tool_status(tool_name: str, summary: str) -> None:
    console.print(f"  [dim]→ {tool_name}: {summary}[/dim]")


def print_assistant_header() -> None:
    console.print()


def stream_markdown(text: str) -> None:
    console.print(Markdown(text))


def print_error(message: str) -> None:
    console.print(f"[red]Error: {message}[/red]")


def print_help() -> None:
    console.print(Panel.fit(
        "[bold]Commands:[/bold]\n"
        "  [cyan]/search <query>[/cyan]   Search knowledge base directly\n"
        "  [cyan]/ingest <path>[/cyan]    Add file or directory to knowledge base\n"
        "  [cyan]/kb list[/cyan]           List indexed documents\n"
        "  [cyan]/kb remove <id>[/cyan]    Remove a document\n"
        "  [cyan]/rag debug[/cyan]         Toggle retrieval debug output\n"
        "  [cyan]/rag config[/cyan]        Show retrieval configuration\n"
        "  [cyan]/rag top_k <n>[/cyan]     Set number of results\n"
        "  [cyan]/rag reranker on|off[/cyan] Toggle cross-encoder reranker\n"
        "  [cyan]/rag rewrite on|off[/cyan]  Toggle query rewriting\n"
        "  [cyan]/memory add <text>[/cyan]  Remember a fact or preference\n"
        "  [cyan]/memory list[/cyan]        List remembered items\n"
        "  [cyan]/memory remove <id>[/cyan] Remove a memory\n"
        "  [cyan]/config[/cyan]            Show current configuration\n"
        "  [cyan]/help[/cyan]              Show this help\n"
        "  [cyan]/quit[/cyan]              Exit",
        title="Help",
    ))


def print_config(config) -> None:
    console.print(Panel(
        f"Model: {config.deepseek_model}\n"
        f"KB directory: {config.kb_dir}\n"
        f"Agent directory: {config.agent_dir}",
        title="Configuration",
    ))
