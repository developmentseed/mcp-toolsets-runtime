"""Turns derived from checkpoints, and a value as it stood at one of them.

The point of the module under test is that nothing new is stored: LangGraph's
checkpoints already hold every past ``tool_state``. So these run *real* turns
against a real checkpointer and read the history back, rather than assembling
snapshot objects by hand — a fake would prove the derivation and not the
premise it rests on.

The publisher here writes a **different value every call**, which the shared
one in the streaming suite does not. Without that, "the value as it stood at
turn 1" and "the value now" are the same bytes and the test would pass whether
or not history worked.
"""

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from mcp_agent.main import BuiltAgent, with_session_state
from mcp_agent_api.history import turns_of
from mcp_agent_api.routes import create_router
from mcp_runtime.declarations import PRODUCES_META_KEY
from tests.mcp_agent.test_streaming import STATE_KEY, StreamingScriptedModel, _tool_call

THREAD = "many-turns"


def _versioning_publisher() -> StructuredTool:
    """Publishes a new value under the same key on every call."""
    calls = {"n": 0}

    async def call() -> tuple[str, dict[str, Any]]:
        calls["n"] += 1
        return f"published {calls['n']}", {
            "structured_content": {
                "message": f"published {calls['n']}",
                "geometry": {"written_on_turn": calls["n"]},
            }
        }

    return StructuredTool(
        name="search",
        description="search",
        args_schema={"type": "object", "properties": {}},
        coroutine=call,
        response_format="content_and_artifact",
        metadata={
            "_meta": {
                PRODUCES_META_KEY: [
                    {
                        "stateKey": STATE_KEY,
                        "field": "geometry",
                    }
                ]
            }
        },
    )


def _built(turns: int = 2) -> BuiltAgent:
    """An agent whose script runs `turns` questions, each publishing once."""
    script: list[Any] = []
    for n in range(turns):
        script += [_tool_call("search", f"c{n}"), AIMessage(content=f"answer {n + 1}")]
    agent = with_session_state(
        StreamingScriptedModel(script=script),
        [_versioning_publisher()],
        InMemorySaver(),
    )
    return BuiltAgent(agent, {}, [_versioning_publisher()], None)


def _client(built: BuiltAgent) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(create_router(lambda: built))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    )


async def _ask(client: httpx.AsyncClient, text: str) -> None:
    response = await client.post(
        "/runs",
        json={
            "threadId": THREAD,
            "messages": [{"id": "u", "role": "user", "content": text}],
        },
    )
    assert response.status_code == 200
    await response.aread()


async def test_a_turn_is_derived_for_each_question() -> None:
    built = _built()
    async with _client(built) as client:
        await _ask(client, "first question")
        await _ask(client, "second question")
        body = (await client.get(f"/threads/{THREAD}/turns")).json()

    assert body["turns"] == 2
    assert body["total"] == 2
    assert [turn["turn"] for turn in body["history"]] == [1, 2]
    assert [turn["question"] for turn in body["history"]] == [
        "first question",
        "second question",
    ]
    assert all(turn["checkpointId"] for turn in body["history"])


async def test_each_turn_carries_the_state_it_ended_with() -> None:
    """Cumulative, and the metadata shape every STATE_SNAPSHOT already uses."""
    built = _built()
    async with _client(built) as client:
        await _ask(client, "first question")
        await _ask(client, "second question")
        history = (await client.get(f"/threads/{THREAD}/turns")).json()["history"]

    for turn in history:
        entry = turn["state"][STATE_KEY]
        assert entry["tool"] == "search"


async def test_a_past_turn_serves_the_value_it_actually_ran_on() -> None:
    """The whole point: turn 1's value survives turn 2 overwriting the key."""
    built = _built()
    async with _client(built) as client:
        await _ask(client, "first question")
        await _ask(client, "second question")

        now = (await client.get(f"/threads/{THREAD}/state/{STATE_KEY}")).json()
        first = (await client.get(f"/threads/{THREAD}/state/{STATE_KEY}?turn=1")).json()
        second = (
            await client.get(f"/threads/{THREAD}/state/{STATE_KEY}?turn=2")
        ).json()

    assert first["value"] == {"written_on_turn": 1}
    assert second["value"] == {"written_on_turn": 2}
    # Without `turn`, the route still answers with the current value.
    assert now["value"] == {"written_on_turn": 2}
    assert now["turn"] is None
    assert first["turn"] == 1


async def test_a_turn_the_thread_never_had_is_404() -> None:
    built = _built()
    async with _client(built) as client:
        await _ask(client, "only question")
        response = await client.get(f"/threads/{THREAD}/state/{STATE_KEY}?turn=9")
    assert response.status_code == 404
    assert "it has had 1" in response.json()["detail"]


async def test_a_key_absent_at_that_turn_says_so() -> None:
    built = _built()
    async with _client(built) as client:
        await _ask(client, "only question")
        response = await client.get(f"/threads/{THREAD}/state/nope/missing?turn=1")
    assert response.status_code == 404
    assert "at turn 1" in response.json()["detail"]


async def test_an_evicted_turn_is_410_not_404() -> None:
    """A pruned checkpointer must not read as "that turn never existed".

    Simulated by pruning the history the way retention does — the distinction
    is drawn from `total`, counted off the thread's surviving messages, so a
    turn the messages remember and the checkpoints do not is `410 Gone`.
    """
    built = _built()
    async with _client(built) as client:
        await _ask(client, "first question")
        await _ask(client, "second question")

        history = await turns_of(built.agent, THREAD)
        assert history.total == 2

        # Only turn 2's checkpoints survive, as an evicted thread's would.
        async def pruned(*_: Any, **__: Any) -> Any:
            for snapshot in []:  # pragma: no cover - empty on purpose
                yield snapshot

        original = type(built.agent).aget_state_history
        try:
            type(built.agent).aget_state_history = pruned  # type: ignore[assignment]
            response = await client.get(f"/threads/{THREAD}/state/{STATE_KEY}?turn=1")
        finally:
            type(built.agent).aget_state_history = original  # type: ignore[assignment]

    assert response.status_code == 410
    assert "no longer retained" in response.json()["detail"]
    assert "keeps 0 of its 2 turns" in response.json()["detail"]


async def test_an_unknown_thread_is_still_404() -> None:
    built = _built()
    async with _client(built) as client:
        assert (await client.get("/threads/nope/turns")).status_code == 404


@pytest.mark.parametrize("turn", [0, -1])
async def test_a_nonsense_turn_number_is_rejected(turn: int) -> None:
    built = _built()
    async with _client(built) as client:
        await _ask(client, "only question")
        response = await client.get(f"/threads/{THREAD}/state/{STATE_KEY}?turn={turn}")
    assert response.status_code == 404
