"""Tests for retrieval ranking."""
from core.retrieval.ranking import rank_chunks


def _candidate(path, content, modified=0.0, language="java"):
    return {
        "path": path,
        "content": content,
        "language": language,
        "modified": modified,
        "start_line": 1,
        "end_line": 5,
        "rank": 0.0,
    }


def test_filename_match_scores_higher():
    exact = _candidate("src/UserService.java", "public class UserService {\n}", modified=0)
    other = _candidate("src/OrderService.java", "public class OrderService {\n}", modified=0)
    results = rank_chunks([other, exact], "UserService", limit=2)
    assert results[0]["path"] == "src/UserService.java"


def test_symbol_mention_boost():
    c1 = _candidate("a.py", "def authenticate():\n    pass\n", language="python")
    c2 = _candidate("b.py", "def login():\n    pass\n", language="python")
    results = rank_chunks([c2, c1], "authenticate", limit=1)
    assert results[0]["path"] == "a.py"


def test_recently_edited_gets_boost():
    import time

    recent = _candidate("src/Recent.java", "class Recent {}\n", modified=time.time() - 3600)
    old = _candidate("src/Old.java", "class Old {}\n", modified=time.time() - 30 * 86400)
    results = rank_chunks([old, recent], "recent", limit=1)
    assert results[0]["path"] == "src/Recent.java"


def test_import_match_boost():
    c1 = _candidate("src/ImportUser.java", "import com.example.User;\nclass ImportUser {}", language="java")
    c2 = _candidate("src/PlainUser.java", "class PlainUser {}", language="java")
    results = rank_chunks([c2, c1], "User", limit=1)
    assert results[0]["path"] == "src/ImportUser.java"


def test_limit_respected():
    candidates = [_candidate(f"src/F{i}.java", f"class F{i} {{}}") for i in range(10)]
    results = rank_chunks(candidates, "thing", limit=3)
    assert len(results) == 3
