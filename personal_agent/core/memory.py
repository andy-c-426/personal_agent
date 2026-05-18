from personal_agent.core.conversation import Conversation


class MemoryManager:
    def __init__(self, config, client=None):
        self.config = config
        self._client = client

    @property
    def _model_context(self) -> int:
        return 65536

    def should_compress(self, conversation: Conversation, model_context: int | None = None) -> bool:
        ctx = model_context or self._model_context
        threshold = int(ctx * self.config.context_threshold_ratio)
        return conversation.total_tokens > threshold

    def maybe_compress(self, conversation: Conversation, model_context: int | None = None) -> str | None:
        ctx = model_context or self._model_context

        if not self.should_compress(conversation, ctx):
            return None

        all_msgs = list(conversation.messages)
        recent = all_msgs[-self.config.recent_message_count:]
        older = all_msgs[:-self.config.recent_message_count]

        if not older or not self._client:
            return None

        old_text = "\n".join(f"[{m.role}]: {m.content[:300]}" for m in older)

        response = self._client.chat.completions.create(
            model=self.config.deepseek_model,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 3-5 sentences. Focus on key facts, decisions, and user preferences."},
                {"role": "user", "content": old_text},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()
        conversation.summary = summary
        conversation.messages = recent
        return summary

    def build_system_prompt(self, kb_doc_count: int, conversation: Conversation) -> str:
        summary_block = ""
        if conversation.summary:
            summary_block = f"\nConversation summary: {conversation.summary}\n"

        return f"""You are a personal assistant with access to a local knowledge base and web search.

Knowledge base: {kb_doc_count} documents indexed. Use kb_search to find relevant local information.
Web search: Use web_search when you need information not in the knowledge base.
{summary_block}
When answering:
- Prefer knowledge base results over web search when available
- Cite your sources (document name or URL)
- If both sources are used, distinguish between them"""
