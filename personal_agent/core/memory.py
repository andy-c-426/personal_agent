from personal_agent.core.conversation import Conversation


class MemoryManager:
    DEFAULT_MODEL_CONTEXT = 65536

    def __init__(self, config, client=None):
        self.config = config
        self._client = client

    def _resolve_context(self, model_context: int | None) -> int:
        return model_context if model_context is not None else self.DEFAULT_MODEL_CONTEXT

    def should_compress(self, conversation: Conversation, model_context: int | None = None) -> bool:
        ctx = self._resolve_context(model_context)
        threshold = int(ctx * self.config.context_threshold_ratio)
        return conversation.total_tokens > threshold

    def maybe_compress(self, conversation: Conversation, model_context: int | None = None) -> str | None:
        """Compress older messages into a rolling summary, preserving system messages.

        Mutates conversation.messages and conversation.summary in place.
        Returns the new summary string, or None if compression was not needed/possible.
        """
        ctx = self._resolve_context(model_context)

        if not self.should_compress(conversation, ctx):
            return None

        all_msgs = list(conversation.messages)
        recent = all_msgs[-self.config.recent_message_count:]
        older = all_msgs[:-self.config.recent_message_count]

        # Preserve system messages from the older slice
        system_msgs = [m for m in older if m.role == "system"]
        older_non_system = [m for m in older if m.role != "system"]

        if not older_non_system or not self._client:
            return None

        # Build text from older messages, including previous summary if it exists
        old_text_parts = []
        if conversation.summary:
            old_text_parts.append(f"Previous conversation summary: {conversation.summary}")
        for m in older_non_system:
            old_text_parts.append(f"[{m.role}]: {m.content[:300]}")
        old_text = "\n".join(old_text_parts)

        try:
            response = self._client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=[
                    {"role": "system", "content": "Summarize this conversation in 3-5 sentences. Incorporate the previous summary if present. Focus on key facts, decisions, and user preferences."},
                    {"role": "user", "content": old_text},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            summary = response.choices[0].message.content.strip()
        except Exception:
            return None

        conversation.summary = summary
        conversation.messages = system_msgs + recent
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
