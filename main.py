"""Entry point for the Data Agent project.

Run this module as a script to start an interactive command‑line session
with the data agent.  The dataset path can either be supplied via
command‑line argument ``--data`` or environment variable ``DATA_AGENT_DATA_PATH``.
If neither is provided and a ``DATA_AGENT_DATA_URL`` environment
variable exists, the dataset will be downloaded on first run.

Example:

    python main.py --data ~/Downloads/my_dataset.csv

Environment variables:

    OPENAI_API_KEY       Required for LLM integration (if using OpenAI).
    ANTHROPIC_API_KEY    Placeholder for future Anthropic support (not implemented).
    DATA_AGENT_DATA_URL  Optional URL to download dataset if not present locally.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

from data_agent import DataHandler, ChatAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the Data Agent interactive CLI")
    parser.add_argument(
        "--data",
        type=str,
        default=os.getenv("DATA_AGENT_DATA_PATH"),
        help="Path to the CSV or Parquet dataset. If omitted, DATA_AGENT_DATA_PATH env var is used.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("DATA_AGENT_DATA_URL"),
        help="Remote URL to download the dataset if the local path is missing.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable calling the LLM and use rule‑based fallback only.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print generated code and prompts for debugging purposes.",
    )
    return parser.parse_args()


def _determine_dataset_path(cli_path: Optional[str], url: Optional[str]) -> Tuple[str, Optional[str]]:
    """Resolve the dataset path or download from a URL if necessary.

    If a local path is provided and exists, it is returned directly.
    If a path is provided but missing and a URL is given, the file is downloaded
    into a ``data/`` directory relative to the current working directory.
    If no path is provided but a URL is, the file is downloaded similarly.
    If neither is provided, the user is prompted to enter a path.

    Returns a tuple ``(dataset_path, dataset_url)``.  The second value is
    ``None`` when the dataset is resolved locally.
    """
    import urllib.request

    dataset_path = cli_path or os.getenv("DATA_AGENT_DATA_PATH")
    dataset_url = url or os.getenv("DATA_AGENT_DATA_URL")

    # If a valid local path is provided, return it directly
    if dataset_path and os.path.exists(dataset_path):
        return dataset_path, None

    # If a local path is provided but missing and a URL exists, download it
    if dataset_path and not os.path.exists(dataset_path) and dataset_url:
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        local_name = os.path.basename(dataset_path)
        dest = os.path.join(data_dir, local_name)
        if not os.path.exists(dest):
            print(f"Downloading dataset from {dataset_url} to {dest} ...")
            try:
                urllib.request.urlretrieve(dataset_url, dest)
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")
        return dest, None

    # If no local path but URL is provided, download to ./data
    if not dataset_path and dataset_url:
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        local_name = os.path.basename(dataset_url.split("?", 1)[0])
        dest = os.path.join(data_dir, local_name)
        if not os.path.exists(dest):
            print(f"Downloading dataset from {dataset_url} to {dest} ...")
            try:
                urllib.request.urlretrieve(dataset_url, dest)
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")
        return dest, None

    # If no path resolved, prompt interactively
    if not dataset_path:
        user_input = input("Path to local dataset (CSV/Parquet): ").strip()
        if not user_input or not os.path.exists(user_input):
            raise SystemExit("Dataset not found. Provide --data, DATA_AGENT_DATA_PATH or DATA_AGENT_DATA_URL.")
        return user_input, None

    # Should never reach here
    return dataset_path, dataset_url


def main() -> int:
    args = parse_args()
    try:
        dataset_path, url = _determine_dataset_path(args.data, args.url)
    except Exception as e:
        print(f"Error resolving dataset path: {e}")
        return 1
    handler = DataHandler(dataset_path=dataset_path, dataset_url=url)
    try:
        handler.load_dataset()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return 1
    # Fill missing values by default
    handler.handle_missing_values(strategy="fill")
    agent = ChatAgent(data_handler=handler, debug=args.debug)
    if args.no_llm:
        # Disable LLM usage by clearing provider and API keys
        agent.provider = ""
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
    print("\nData Agent ready. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question or question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        # Call the agent and handle both legacy and extended return signatures
        resp = agent.ask(question)
        # Unpack response: handle both 3-tuple (answer,evidence,code) and 4-tuple (answer,evidence,code,meta)
        if isinstance(resp, tuple) and len(resp) == 4:
            answer, evidence, code, meta = resp
        else:
            answer, evidence, code = resp  # type: ignore
            meta = {}
        print("Answer:", answer)
        # Print evidence if present
        if evidence is not None:
            max_rows = 10
            if len(evidence) <= max_rows:
                print(evidence)
            else:
                print(evidence.head(max_rows))
                print(f"... ({len(evidence)} rows total)")
        # Print metadata when available (method, columns, filters, hypothesis)
        if meta:
            method = meta.get("method")
            cols = meta.get("columns")
            filts = meta.get("filters")
            hypothesis = meta.get("hypothesis")
            if method:
                print("Method:", method)
            if cols:
                print("Columns:", ", ".join(cols))
            if filts:
                print("Filters:", "; ".join(filts))
            if hypothesis:
                print("Hypothesis:", hypothesis)
        # Print executed code if debug flag is set
        if args.debug and code:
            print("\nExecuted code:\n", code)
    return 0


if __name__ == "__main__":
    sys.exit(main())