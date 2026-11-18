from voicebot.speech_filter import filter_stream

def run(chunks):
    return "".join(filter_stream(iter(chunks)))

def test_plain_text_passes():            assert run(["Hello ", "world."]) == "Hello world."
def test_think_block_removed():          assert run(["<think>secret</think>Hi"]) == "Hi"
def test_think_tags_split():             assert run(["<th", "ink>sec", "ret</thi", "nk>Hel", "lo"]) == "Hello"
def test_empty_think_block():            assert run(["<think>\n\n</think>\n\n", "Sure."]).strip() == "Sure."
def test_unclosed_think_drops_rest():    assert run(["Ok. <think>reasoning", " more"]) == "Ok. "
def test_markdown_symbols_removed():     assert run(["**Bold** and `code` and # head"]) == "Bold and code and  head"
def test_line_start_bullets_removed():   assert run(["Items:\n- one\n", "- two"]) == "Items:\none\ntwo"
def test_inner_hyphens_kept():           assert run(["well-known - fine"]) == "well-known - fine"
def test_lone_angle_bracket_kept():      assert run(["a < b", " and c"]) == "a < b and c"
def test_never_yields_empty():
    assert all(c != "" for c in filter_stream(iter(["<think>", "x", "</think>", "", "**", "ok"])))
def test_is_lazy():
    gen = filter_stream(iter(["First. ", "<think>"]))
    assert next(gen) == "First. "
