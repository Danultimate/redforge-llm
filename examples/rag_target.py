"""RAG target example.

The target returns a TargetResponse with `sources` metadata so heuristic
scoring can detect attacks that exfiltrate retrieved documents.
"""

from __future__ import annotations

import asyncio

from redforge import Scanner, TargetResponse


async def my_rag_target(prompt: str) -> TargetResponse:
    # docs = await my_retriever.search(prompt, k=3)
    # response = await my_llm.complete(build_context(docs) + prompt)
    # return TargetResponse(
    #     text=response,
    #     metadata={"sources": [d.id for d in docs]},
    # )
    raise NotImplementedError("Plug your retriever + LLM in here.")


async def main() -> None:
    scan = await Scanner(target=my_rag_target).run()
    scan.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
