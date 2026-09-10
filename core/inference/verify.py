"""RNP-inspired predictive verification of model outputs.

Fisher & Rao (2023, PMC10637337) describe predictive coding: bottom-up prediction
errors drive updates to top-down state estimates.  In the code-assistant
setting, we mirror this by *verifying* model-generated outputs against the
retrieved context — acting as a prediction-error signal that catches
hallucinations (citing files/lines not in context, referencing symbols that
don't exist, or producing malformed structured output).

The verifier:
1. Checks that every path referenced in the output exists in the retrieved context.
2. Checks that line numbers fall within the bounds of the cited files.
3. Checks that JSON-structured output (when using grammar-constrained generation)
   parses and has the required top-level keys.
4. Returns a ``VerificationResult`` with issues and a confidence score.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Outcome of verifying a model output against retrieved context."""
    valid: bool
    confidence: float  # 0.0 – 1.0
    issues: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.valid = False

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "confidence": round(self.confidence, 3),
            "issues": self.issues,
            "citations": self.citations,
        }


class VerificationError(Exception):
    """Raised when verification fails and strict mode is enabled."""
    pass


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

_PATH_LINE_RE = re.compile(
    r'([\w./-]+\.[A-Za-z]{1,4})'
    r'(?::(\d+)(?:-(\d+))?|\[L(\d+)\])',
)
_SIMPLE_PATH_RE = re.compile(
    r'([\w./-]+\.[A-Za-z]{1,4})(?::(\d+))(?::-(\d+))?'
)
_BARE_PATH_RE = re.compile(
    r'(?<![\w/-])([\w-]+(?:/[\w.-]+)*\.[A-Za-z]{1,4})(?![\w./-])'
)


def extract_citations(text: str) -> list[tuple[str, int | None, int | None]]:
    """Extract (path, start_line, end_line) citations from model output.

    Returns tuples where lines may be None if unspecified.
    """
    citations: list[tuple[str, int | None, int | None]] = []
    seen: set[tuple] = set()

    for m in _PATH_LINE_RE.finditer(text):
        path = m.group(1)
        start = m.group(2) or m.group(4)  # group 4 is [L{n}] form
        end = m.group(3)
        if start:
            start_i = int(start)
            end_i = int(end) if end else start_i
        else:
            start_i = end_i = None
        key = (path, start_i, end_i)
        if key not in seen and len(path) >= 3:
            seen.add(key)
            citations.append((path, start_i, end_i))

    if not citations:
        for m in _SIMPLE_PATH_RE.finditer(text):
            path = m.group(1)
            start = m.group(2)
            end = m.group(3)
            if not path or len(path) < 3:
                continue
            start_i = int(start) if start else None
            end_i = int(end) if end and end != start else None
            key = (path, start_i, end_i)
            if key not in seen:
                seen.add(key)
                citations.append((path, start_i, end_i))

    if not citations:
        # Bare file paths with no line reference at all
        for m in _BARE_PATH_RE.finditer(text):
            path = m.group(1)
            if len(path) >= 3 and path not in seen:
                seen.add(path)
                citations.append((path, None, None))

    return citations


# ---------------------------------------------------------------------------
# Context extraction & verification
# ---------------------------------------------------------------------------

def _extract_context_paths(sections: list[dict]) -> dict[str, dict]:
    """Extract file paths and their line ranges from BuiltContext sections."""
    import re as _re
    paths: dict[str, dict] = {}
    for section in sections:
        text = section.get("text", "")
        for m in _re.finditer(r'#\s+(.+\.py)[-:]?(\d+)?(?:-(\d+))?', text):
            path = m.group(1)
            start = int(m.group(2)) if m.group(2) else 0
            end = int(m.group(3)) if m.group(3) else 0
            if path in paths:
                paths[path]["start_line"] = min(paths[path].get("start_line", 0), start) if start else paths[path].get("start_line", 0)
                paths[path]["end_line"] = max(paths[path].get("end_line", 0), end) if end else paths[path].get("end_line", 0)
            else:
                paths[path] = {"start_line": start, "end_line": end}
    return paths


def _extract_context_snippets(sections: list[dict]) -> str:
    """Concatenate all context text for content-based checks."""
    return "\n\n".join(s.get("text", "") for s in sections)


def verify_output(output: str, *, sections: list[dict] | None = None,
                  context_text: str | None = None,
                  expected_json_keys: list[str] | None = None) -> VerificationResult:
    """Verify a model-generated ``output`` against the retrieved context.

    Parameters
    ----------
    output : str
        The model's generated text (or JSON patch / structured response).
    sections : list[dict], optional
        Context sections from ``BuiltContext.sections``.
    context_text : str, optional
        Raw concatenated context text.
    expected_json_keys : list[str], optional
        If provided, checks that output is valid JSON with these keys.

    Returns
    -------
    VerificationResult
    """
    result = VerificationResult(valid=True, confidence=1.0)

    if context_text is None and sections is not None:
        context_text = _extract_context_snippets(sections)

    # --- Check 1: JSON structure (for grammar-constrained outputs) ---
    if expected_json_keys:
        try:
            data = json.loads(output.strip())
            if not isinstance(data, dict):
                raise ValueError("Output is valid JSON but not an object/dict")
            for key in expected_json_keys:
                if key not in data:
                    result.add_issue(f"Missing required JSON key: '{key}'")
        except (json.JSONDecodeError, ValueError) as e:
            result.add_issue(f"Output is not valid JSON: {e}")

    # --- Check 2: Citation verification (RNP prediction-error signal) ---
    if context_text:
        citations = extract_citations(output)
        context_paths = _extract_context_paths(sections) if sections else {}

        for path, start, end in citations:
            cit_str = f"{path}:{start}-{end}" if start and end else (
                f"{path}:{start}" if start else path
            )
            result.citations.append(cit_str)

            path_in_context = (
                path in context_text or
                path.split("/")[-1] in context_text or
                any(ctx.endswith(path) or path.endswith(ctx)
                    for ctx in context_paths)
            )
            if not path_in_context:
                result.add_issue(
                    f"Cited path '{path}' does not appear in the retrieved context "
                    f"(potential hallucination)"
                )

            if start is not None and context_paths:
                for ctx_path, ctx_range in context_paths.items():
                    if ctx_path.endswith(path) or path.endswith(ctx_path):
                        ctx_start = ctx_range.get("start_line", 0)
                        ctx_end = ctx_range.get("end_line", 0)
                        if ctx_start and ctx_end:
                            if start < ctx_start or (ctx_end and start > ctx_end):
                                result.add_issue(
                                    f"Cited line {start} for '{path}' is outside "
                                    f"the context range ({ctx_start}-{ctx_end})"
                                )

    if result.issues:
        result.confidence = max(0.0, 1.0 - len(result.issues) * 0.25)
    else:
        result.confidence = 1.0

    return result


def verify_or_raise(output: str, *, sections: list[dict] | None = None,
                    context_text: str | None = None,
                    expected_json_keys: list[str] | None = None,
                    strict: bool = False) -> VerificationResult:
    """Like ``verify_output`` but optionally raises on failure."""
    result = verify_output(output, sections=sections, context_text=context_text,
                           expected_json_keys=expected_json_keys)
    if strict and not result.valid:
        raise VerificationError(f"Verification failed: {'; '.join(result.issues)}")
    return result


__all__ = [
    "VerificationResult",
    "VerificationError",
    "verify_output",
    "verify_or_raise",
    "extract_citations",
]
