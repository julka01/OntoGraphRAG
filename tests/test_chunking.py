import sys
import types

from ontographrag.kg.chunking import chunk_text


def test_chunk_text_falls_back_when_tiktoken_splitter_is_unavailable(monkeypatch):
    class FakeSplitter:
        def __init__(self, chunk_size, chunk_overlap):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        @classmethod
        def from_tiktoken_encoder(cls, chunk_size, chunk_overlap):
            raise RuntimeError("offline tokenizer assets unavailable")

        def split_text(self, text):
            return [text[:5], text[5:]]

    fake_module = types.SimpleNamespace(RecursiveCharacterTextSplitter=FakeSplitter)
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", fake_module)

    chunks = chunk_text(
        "abcdefghij",
        chunk_size=5,
        chunk_overlap=0,
        embedding_fn=None,
    )

    assert [chunk["text"] for chunk in chunks] == ["abcde", "fghij"]
    assert chunks[0]["start_pos"] == 0
    assert chunks[1]["start_pos"] == 5
