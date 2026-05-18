import json
from openai import OpenAI
from personal_agent.core.conversation import Conversation
from personal_agent.core.memory import MemoryManager
from personal_agent.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        config,
        client: OpenAI,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager | None = None,
        kb_doc_count: int = 0,
    ):
        self.config = config
        self.client = client
        self.registry = tool_registry
        self.memory = memory_manager or MemoryManager(config)
        self.kb_doc_count = kb_doc_count

    def run(self, user_input: str, conversation: Conversation) -> tuple[str, list[dict]]:
        conversation.add_message("user", user_input)

        system_prompt = self.memory.build_system_prompt(self.kb_doc_count, conversation)

        tool_calls_made = []
        same_tool_count = 0
        last_tool_name = None

        for _ in range(self.config.max_tool_iterations):
            messages = [{"role": "system", "content": system_prompt}] + conversation.to_dicts()

            response = self.client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=messages,
                tools=self.registry.schemas() if self.registry.schemas() else None,
                temperature=0.7,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    result = self.registry.execute(tool_name, **args)
                    tool_calls_made.append({"name": tool_name, "arguments": args, "result": result})

                    conversation.add_message(
                        "assistant",
                        content="",
                        tool_calls=[{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tc.function.arguments},
                        }],
                    )
                    conversation.add_message("tool", content=result, tool_call_id=tc.id)

                    if tool_name == last_tool_name:
                        same_tool_count += 1
                    else:
                        same_tool_count = 1
                        last_tool_name = tool_name

                    if same_tool_count >= self.config.max_same_tool_calls:
                        final = f"I've called '{tool_name}' several times without progress. Let me stop and share what I have so far. The last result was: {result[:500]}"
                        conversation.add_message("assistant", final)
                        self.memory.maybe_compress(conversation)
                        return final, tool_calls_made
            else:
                text = msg.content or ""
                conversation.add_message("assistant", text)

                self.memory.maybe_compress(conversation)

                return text, tool_calls_made

        final = "I've reached the maximum number of tool calls. Here's what I found so far."
        conversation.add_message("assistant", final)
        self.memory.maybe_compress(conversation)
        return final, tool_calls_made
