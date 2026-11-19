from voicebot.conversation import Conversation

def test_messages_shape():
    c = Conversation("SYS", history_turns=2); c.add_exchange("u1", "a1")
    assert c.messages("u2") == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]

def test_history_trimmed_to_last_n_pairs():
    c = Conversation("S", history_turns=2)
    for i in range(1, 4): c.add_exchange(f"u{i}", f"a{i}")
    assert [m["content"] for m in c.messages("new")] == ["S", "u2", "a2", "u3", "a3", "new"]

def test_clear_and_mutable_prompt():
    c = Conversation("S", 10); c.add_exchange("u", "a"); c.clear(); c.system_prompt = "NEW"
    assert c.turns == [] and c.messages("x")[0]["content"] == "NEW"

def test_context_goes_to_system_message_not_history():
    c = Conversation("SYS", 10, rag_template="Notes {notes}:\n{context}")      # Review Focus 1: other braces stay
    m = c.messages("q", context="[a.txt, part 1]\nfact")
    assert m[0]["content"] == "SYS\n\nNotes {notes}:\n[a.txt, part 1]\nfact" and m[-1] == {"role": "user", "content": "q"}
    assert c.messages("q")[0]["content"] == "SYS"

def test_default_rag_template_is_exact():
    from voicebot.config import DEFAULT_RAG_TEMPLATE
    assert DEFAULT_RAG_TEMPLATE == ("Answer from these notes when they are relevant. If they do not cover the "
                                    "question, answer normally and do not mention them.\n\n{context}")
    assert Conversation("S", 1).rag_template == DEFAULT_RAG_TEMPLATE
