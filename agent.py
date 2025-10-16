"""Chat agent orchestration and natural language query handling.

The ``ChatAgent`` class glues together a ``DataHandler`` and one or
more large language models (LLMs) to provide a natural‑language
interface over a tabular dataset.  For each user question the agent
builds a prompt describing the dataset schema and sample values,
delegates to an LLM to generate Python code that answers the question,
executes that code in a controlled environment, and returns both the
result and the intermediate analysis as supporting evidence.

If an LLM API key is not configured, the agent falls back to a very
simple pattern‑matching approach for a limited set of deterministic
queries (e.g., counts, averages).  The fallback exists to maintain
basic functionality when offline or during testing.
"""

from __future__ import annotations

import ast
import builtins
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_handler import DataHandler

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore


def _safe_exec(code: str, local_vars: Dict[str, Any]) -> Any:
    """Execute user‑generated Python code in a restricted namespace.

    To mitigate risks from arbitrary code execution, only a whitelist of
    built‑ins is exposed and the global namespace is empty.  The
    ``local_vars`` dict is provided to the execution so that code can
    reference a pandas DataFrame under the name ``df`` as well as
    common numpy/pandas modules.

    Parameters
    ----------
    code : str
        The Python code to execute.
    local_vars : dict
        Local variables to make available to the code.

    Returns
    -------
    Any
        The result of evaluating the last expression in ``code``, if
        present.  If no expression is present, returns ``None``.
    """
    # Whitelist safe built‑ins
    safe_builtins = {k: getattr(builtins, k) for k in ["abs", "min", "max", "len", "sum", "range", "enumerate"]}
    # Prepare a restricted global namespace
    restricted_globals: Dict[str, Any] = {"__builtins__": safe_builtins, "np": np, "pd": pd}
    # Parse code into an AST and separate the last expression if present
    parsed = ast.parse(code)
    body = parsed.body
    last_value = None
    # If the last statement is an expression, evaluate it explicitly to capture its value
    if body and isinstance(body[-1], ast.Expr):
        expr = ast.Expression(body[-1].value)
        body = body[:-1]
        compiled_expr = compile(expr, filename="<ast>", mode="eval")
        last_value = compiled_expr
    # Execute the body statements
    compiled_body = compile(ast.Module(body=body, type_ignores=[]), filename="<ast>", mode="exec")
    exec(compiled_body, restricted_globals, local_vars)
    # Evaluate the last expression if present
    result = None
    if last_value is not None:
        result = eval(last_value, restricted_globals, local_vars)
    return result


@dataclass
class ChatAgent:
    """Main class providing a natural‑language interface over a dataset.

    Parameters
    ----------
    data_handler : DataHandler
        Loaded data handler instance.
    model : str, optional
        Name of the LLM model to call via OpenAI's API.  Ignored if
        ``openai`` is not installed or no API key is configured.
    temperature : float, optional
        Sampling temperature for LLM generation.
    debug : bool, optional
        If True, prints intermediate prompts and code for inspection.
    """

    data_handler: DataHandler
    # Model name used for LLM calls; default can be overridden via LLM_MODEL env var.
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4-turbo"))
    # Sampling temperature for LLM
    temperature: float = 0.0
    # If True, prints intermediate prompts and generated code
    debug: bool = False
    # LLM provider identifier ("openai" or "anthropic"), defaults to LLM_PROVIDER env var or 'openai'
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    history: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalise provider string and honour environment overrides."""
        # lower-case provider; fall back to openai if unspecified
        if self.provider:
            self.provider = self.provider.lower()
        else:
            self.provider = "openai"

    def _llm_available(self) -> bool:
        """Return True if an LLM provider is configured and credentials are available."""
        if self.provider == "openai":
            return openai is not None and os.getenv("OPENAI_API_KEY") is not None
        if self.provider == "anthropic":
            try:
                import anthropic  # type: ignore
            except Exception:
                return False
            return os.getenv("ANTHROPIC_API_KEY") is not None
        return False

    def _build_prompt(self, question: str) -> List[Dict[str, str]]:
        """Construct a prompt for the LLM describing the dataset and question.

        The prompt instructs the model to return Python code only.  It
        includes a summary of the dataset schema (column names and types)
        and a few example rows to help the model understand value ranges.
        """
        df = self.data_handler.df
        assert df is not None, "Dataset must be loaded before building prompt."
        # Extract first 5 rows for context
        head = df.head(5).to_dict(orient="list")
        head_summary = "\n".join(f"{col}: {values}" for col, values in head.items())
        schema = self.data_handler.infer_schema()
        schema_str = ", ".join(f"{col} ({dtype})" for col, dtype in schema.items())
        # Build a system prompt that instructs the LLM to plan the analysis and capture hypotheses.
        system_message = (
            "You are an expert data scientist. Given a pandas DataFrame `df` and a user question, "
            "first create a Python dict named PLAN with keys { 'steps': list[str], 'columns': list[str], 'filters': list[str] } "
            "describing how you will answer the question. Then write Python code using only numpy, pandas, and sklearn to execute that plan. "
            "After computing the result, set a string variable HYPOTHESIS with a cautious, evidence‑backed interpretation and list any limitations or confounders. "
            "Do not use any external libraries. Do not use backticks. Return only Python code; do not include explanations. "
            "The DataFrame has the following schema: " + schema_str + ". "
            "Here are the first few rows:\n" + head_summary
        )
        user_message = question
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return messages

    def _call_llm(self, prompt: List[Dict[str, str]]) -> str:
        """Call the configured LLM provider to obtain Python code for the user's question."""
        if not self._llm_available():
            raise RuntimeError("No supported LLM provider configured or missing API key.")
        if self.provider == "openai":
            # Use OpenAI ChatCompletion API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
            )
            code = response["choices"][0]["message"]["content"].strip()
            return code
        elif self.provider == "anthropic":
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            # Compose a user-only message for Anthropic
            user_msg = prompt[-1]["content"] if prompt else ""
            completion = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": user_msg}],
            )
            code = getattr(completion, "content", None)
            if not code:
                code = completion["choices"][0]["message"]["content"]  # type: ignore
            return code.strip()
        else:
            raise RuntimeError(f"Unsupported LLM provider: {self.provider}")

    def _fallback_answer(self, question: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """A very simple rule‑based fallback for deterministic queries.

        This method supports queries of the form "count of X", "average of Y",
        and "sum of Z".  It returns a string answer and optionally a
        supporting DataFrame.
        """
        df = self.data_handler.df
        assert df is not None
        q = question.lower()
        words = q.split()
        # Count queries
        if q.startswith("count of") and len(words) >= 3:
            col = q.split("count of", 1)[1].strip()
            if col in df.columns:
                return str(df[col].count()), None
        # Average queries
        if q.startswith("average of") and len(words) >= 3:
            col = q.split("average of", 1)[1].strip()
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                return str(df[col].mean()), None
        # Sum queries
        if q.startswith("sum of") and len(words) >= 3:
            col = q.split("sum of", 1)[1].strip()
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                return str(df[col].sum()), None
        return "I'm sorry, I couldn't understand the question without the LLM.", None

    def ask(self, question: str) -> Tuple[str, Optional[pd.DataFrame], Optional[str], Dict[str, Any]]:
        """Answer a natural‑language question about the dataset.

        The agent first attempts to use an LLM to translate the question
        into Python code.  If no LLM is available, it falls back to a
        minimal deterministic parser for simple aggregate queries.  The
        returned tuple contains a human‑readable answer, an optional
        DataFrame with supporting evidence (e.g., for tables), and the
        Python code executed (for transparency).

        Parameters
        ----------
        question : str
            The user's natural‑language question.

        Returns
        -------
        tuple
            A 4‑tuple consisting of:
            - answer: ``str`` – human‑readable answer.
            - evidence: ``pd.DataFrame`` or ``None`` – supporting table when applicable.
            - code: ``str`` or ``None`` – Python code executed (for transparency).
            - meta: ``dict`` – metadata including the analysis method, referenced columns, filters and an optional hypothesis string.
        """
        # Save the user question to history
        self.history.append(("user", question))
        # If an LLM is available, attempt to generate and execute Python code
        if self._llm_available():
            prompt = self._build_prompt(question)
            code: str = self._call_llm(prompt)
            if self.debug:
                print("Generated code:\n", code)
            # Parse PLAN structure from the generated code to extract columns and filters
            import re
            import ast
            plan_columns: List[str] = []
            plan_filters: List[str] = []
            # Try to find a literal PLAN assignment in the code
            m_plan = re.search(r"PLAN\s*=\s*(\{[\s\S]*?\})", code)
            if m_plan:
                try:
                    plan_obj = ast.literal_eval(m_plan.group(1))
                    if isinstance(plan_obj, dict):
                        cols = plan_obj.get("columns", [])
                        filts = plan_obj.get("filters", [])
                        # Deduplicate while preserving order
                        plan_columns = list(dict.fromkeys(cols)) if isinstance(cols, list) else []
                        plan_filters = list(dict.fromkeys(filts)) if isinstance(filts, list) else []
                except Exception:
                    # ignore parse errors; fallback to regex extraction
                    pass
            # Fallback extraction: search for df[...] and query(...) patterns
            if not plan_columns:
                plan_columns = sorted(set(re.findall(r"df\[['\"]([^'\"]+)['\"]\]", code)))
            if not plan_filters:
                plan_filters = sorted(set(re.findall(r"query\(['\"]([^'\"]+)['\"]\)", code)))
            # Prepare a local namespace and execute the generated code
            local_vars = {"df": self.data_handler.df.copy() if self.data_handler.df is not None else None}
            try:
                result = _safe_exec(code, local_vars)
                # Convert numpy scalars to Python primitives
                if isinstance(result, np.generic):
                    result = result.item()
                # Format the result into an answer, evidence and method
                answer, evidence, method = self._format_result(result)
                # Extract optional hypothesis string from the executed code
                hypothesis = local_vars.get("HYPOTHESIS")
                hypothesis_str: Optional[str] = None
                if isinstance(hypothesis, str):
                    hypothesis_str = hypothesis.strip()
                # Build metadata dictionary
                meta: Dict[str, Any] = {
                    "method": method,
                    "columns": plan_columns,
                    "filters": plan_filters,
                    "hypothesis": hypothesis_str,
                }
                self.history.append(("assistant", answer))
                return answer, evidence, code, meta
            except Exception as e:
                # On execution error, return the error message and no evidence or meta
                error_msg = f"Error executing generated code: {e}"
                self.history.append(("assistant", error_msg))
                meta = {"method": "LLM execution error", "columns": plan_columns, "filters": plan_filters, "hypothesis": None}
                return error_msg, None, code, meta
        # Fallback deterministic path when no LLM is available
        answer, evidence = self._fallback_answer(question)
        # Build a simple metadata entry for deterministic answers
        meta = {"method": "Deterministic fallback", "columns": [], "filters": [], "hypothesis": None}
        self.history.append(("assistant", answer))
        return answer, evidence, None, meta

    def _format_result(self, result: Any) -> Tuple[str, Optional[pd.DataFrame], str]:
        """Format a Python result into a human‑readable answer, optional evidence and method.

        Scalars become strings; pandas Series/Frames are returned as evidence with a concise
        description; other objects are converted via ``str()``.  A method string is
        returned to indicate the type of analysis performed.
        """
        import pandas as pd
        import numpy as np  # local import for clarity
        # DataFrame result: return table and note method
        if isinstance(result, pd.DataFrame):
            rows, cols = result.shape
            answer = f"Returned a table with shape {rows}×{cols}."
            return answer, result, "Tabular analysis (pandas)"
        # Series result: treat as single‑column table
        if isinstance(result, pd.Series):
            answer = f"Returned a table with {len(result)} rows and 1 column."
            return answer, result.to_frame(name=result.name), "Tabular analysis (pandas)"
        # Numeric or scalar results
        if np.isscalar(result) or isinstance(result, (str, int, float)):
            return str(result), None, "Vector/aggregate analysis (pandas/numpy)"
        # Generic iterable (list, tuple, dict) – convert to string but note method
        if isinstance(result, (list, tuple, dict)):
            return str(result), None, "Vector/aggregate analysis (pandas/numpy)"
        # Fallback: convert to string
        return str(result), None, "Direct computation"