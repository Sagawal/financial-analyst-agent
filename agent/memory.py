def get_conversation_history(state, max_messages=10, exclude_last=False):

    messages = state.get("messages", [])

    if exclude_last and messages:
        messages = messages[:-1]

    recent_messages = messages[-max_messages:]

    history = []

    for message in recent_messages:

        role = (
            "User"
            if message.type == "human"
            else "Assistant"
        )

        history.append(f"{role}: {message.content}")

    return "\n".join(history)