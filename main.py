"""Optional CLI entrypoint for LLM Council backend operations."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import uvicorn

from backend.config import ROUNDS
from backend.council import run_batch_deliberation, run_deliberation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run FastAPI backend")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8001)

    deliberate_parser = subparsers.add_parser("deliberate", help="Run one deliberation")
    deliberate_parser.add_argument("prompt", help="Prompt to deliberate")
    deliberate_parser.add_argument("--rounds", type=int, default=ROUNDS)

    batch_parser = subparsers.add_parser("batch", help="Run a batch deliberation")
    batch_parser.add_argument("prompts", nargs="+", help="Prompts to deliberate")
    batch_parser.add_argument("--rounds", type=int, default=ROUNDS)

    return parser.parse_args()


async def _run_deliberate(args: argparse.Namespace) -> dict[str, Any]:
    return await run_deliberation(prompt=args.prompt, rounds=args.rounds)


async def _run_batch(args: argparse.Namespace) -> dict[str, Any]:
    return await run_batch_deliberation(prompts=args.prompts, rounds=args.rounds)


def main() -> None:
    args = parse_args()

    if args.command == "serve":
        uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=False)
        return

    if args.command == "deliberate":
        result = asyncio.run(_run_deliberate(args))
        print(json.dumps(result, indent=2))
        return

    if args.command == "batch":
        result = asyncio.run(_run_batch(args))
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
