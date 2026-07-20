def select_recent_messages(
    messages: list[dict],
    max_messages: int = 3,
) -> list[dict]:
    if max_messages < 1:
        raise ValueError("max_messages 必须大于或等于 1。")

    return messages[-max_messages:]


def build_context_messages(
    system_prompt: str,
    history: list[dict],
    current_query: str,
    max_messages: int = 3,
    memory_context: str = "",
) -> list[dict]:
    recent_messages = select_recent_messages(history, max_messages)
    system_content = system_prompt

    if memory_context:
        system_content = f"{system_prompt}\n\n{memory_context}"

    return [
        {"role": "system", "content": system_content},
        *recent_messages,
        {"role": "user", "content": current_query},
    ]
