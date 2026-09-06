"""Tests for semantic chunking."""

from core.indexer.chunker import chunk_content

SAMPLE_JAVA = """package com.example;

import java.util.Optional;

public class UserService {
    private final UserRepository repository;

    public Optional<User> getUser(Long id) {
        if (id == null) {
            return Optional.empty();
        }
        return repository.findById(id);
    }

    public User createUser(String name) {
        User user = new User(name);
        repository.save(user);
        return user;
    }
}
"""


def test_java_chunks_respect_class_and_method_boundaries():
    chunks = chunk_content(SAMPLE_JAVA, "src/UserService.java", "java")
    # package/import header, class declaration, and each method are separate chunks
    assert len(chunks) >= 4
    kinds = [c.kind for c in chunks]
    assert "declaration" in kinds
    assert "function" in kinds
    # header chunk (package line) must exist and contain the imports
    header = chunks[0].content
    assert "package com.example" in header
    assert "import java.util.Optional" in header


def test_chunk_line_ranges_are_inclusive_and_continuous():
    chunks = chunk_content(SAMPLE_JAVA, "src/UserService.java", "java")
    assert chunks[0].start_line == 1
    for prev, current in zip(chunks, chunks[1:]):
        assert current.start_line == prev.end_line + 1
    assert chunks[-1].end_line == len(SAMPLE_JAVA.splitlines())
    for chunk in chunks:
        assert chunk.start_line <= chunk.end_line


def test_chunk_contents_join_to_original_lines():
    chunks = chunk_content(SAMPLE_JAVA, "src/UserService.java", "java")
    joined = "\n".join(c.content for c in chunks) + "\n"
    assert joined == SAMPLE_JAVA  # splitlines + join round-trips exactly (trailing newline restored)


def test_python_functions_chunked():
    code = """import os


class Worker:
    def run(self):
        return 1

    def stop(self):
        return 2


def helper(x):
    return x + 1
"""
    chunks = chunk_content(code, "worker.py", "python")
    names = "\n".join(c.content for c in chunks)
    assert "def run" in names
    assert "def stop" in names
    assert "def helper" in names
    assert "class Worker" in names


def test_empty_content_returns_no_chunks():
    assert chunk_content("", "empty.py", "python") == []
    assert chunk_content("\n\n", "blank.py", "python") == []
