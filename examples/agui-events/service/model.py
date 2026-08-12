"""The chat model behind the API: a real provider, or a stub when none is set.

`PROVIDER_MODEL` and `PROVIDER_API_KEY` put a real model behind the routes,
which is the only way to see a tool called because a model *decided* to call
it. Without them the stub below answers by keyword, so the example still runs
with no key and no network — worth a lot less, but worth more than nothing.
"""

import asyncio
import importlib.util
import itertools
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from mcp_state import handle_for
from service import REPO
from service.settings import get_settings

AOI_KEY = "dataset-search/geometry"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The suite's streaming stub, so this example and the tests drive the agent with
# one model rather than two that can drift. GenericFakeChatModel drops
# tool_calls on its streaming path, so a stub built on that would show an answer
# and nothing else. Loaded by path: the suite has no __init__.py (pytest imports
# it in importlib mode) and an installed distribution shadows a bare `tests`.
StreamingScriptedModel = _load(
    "streaming_stub", REPO / "tests" / "mcp_agent" / "test_streaming.py"
).StreamingScriptedModel


def call(call_id: str, name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def cited(text: str, *ids: str) -> AIMessage:
    """A final answer carrying citations, the way a provider standardises them."""
    return AIMessage(
        content=[
            {
                "type": "text",
                "text": text,
                "annotations": [{"type": "citation", "id": ref} for ref in ids],
            }
        ]
    )


#: A long answer, because the other four are a sentence each and a sentence
#: streams past faster than a screen can show. Text only: no tool call, so
#: what arrives is tokens and nothing else.
ESSAY = """Session state exists because the useful values in a geospatial \
conversation are too big to say out loud.

An area of interest is a few thousand vertices. Put it in the transcript and \
you pay for it on every subsequent turn, the model can transcribe it wrongly, \
and a long conversation eventually stops fitting. So a tool's large return is \
captured into a namespace beside the messages, and what the model is told is \
that a value now exists under a key.

Getting it back out again happens two ways. When a server tags a parameter \
with a kind, the client matches the kind, fills the value, and removes the \
parameter from the schema the model is offered — the model cannot get the \
geometry wrong because it never learns there was one to get. When nothing is \
tagged, which is every third-party server, the parameter stays in the schema \
with a second accepted form, and the model passes an @state: handle of about \
ten tokens. The client swaps in the payload before the call, and the server \
receives ordinary GeoJSON with no idea any of this happened.

Both paths leave a receipt, which is what you can see in this transcript: the \
parameter, the key it came from, the kind, and which tool published it. That \
matters because a value the model never saw is otherwise unaccountable — \
something decided what this tool ran on, and a user is entitled to know what \
and why.

None of that is visible on the wire as bytes. The stream carries the kind, the \
publishing tool and a size for each key, and the payload stays on the server \
until a client asks for it by name. Which is the whole trick: the geometry is \
in the answer without ever being in the conversation."""


def chat_script(turn: int, question: str) -> list[AIMessage]:
    """The scripted reply to one question, keyed off a word in it.

    Call ids carry the turn number because a thread's ids have to stay unique:
    a client keys its rendered tool calls on them, and a second turn reusing
    ``c1`` would land in the first turn's message.
    """
    asked = question.lower()
    if any(word in asked for word in ("clip", "rainfall", "chirps", "area")):
        return [
            call(f"t{turn}-1", "search_datasets", {"query": "rainfall"}),
            # No `aoi`: the declaration removed it from this tool's schema.
            call(f"t{turn}-2", "clip_raster", {"dataset_id": "chirps-daily"}),
            # The foreign tool's parameter is in the schema, so the model fills
            # it with a handle read off the [state updated: …] breadcrumb.
            call(f"t{turn}-3", "describe_geometry", {"geometry": handle_for(AOI_KEY)}),
            AIMessage(content="Done — clipped chirps-daily to your catchment."),
        ]
    if any(word in asked for word in ("contour", "smooth")):
        return [
            AIMessage(
                content="I can't smooth contours here. smooth_contours wants a "
                "geojson.ContourSet and nothing connected publishes one, so it "
                "was withheld before the conversation started."
            )
        ]
    if any(word in asked for word in ("source", "cite", "which")):
        return [
            call(f"t{turn}-1", "search_datasets", {"query": "rainfall"}),
            cited("Three datasets cover the catchment.", "era5-land", "chirps-daily"),
        ]
    if any(word in asked for word in ("explain", "why", "tell me", "long")):
        return [AIMessage(content=ESSAY)]
    return [
        AIMessage(
            content="Try 'clip chirps to my area' for the full turn, 'what "
            "about contours?' for a withheld tool, 'which datasets, with "
            "sources?' for citations, or 'explain session state' for enough "
            "text to watch it stream."
        )
    ]


class ScriptedChat(StreamingScriptedModel):  # type: ignore[misc, valid-type]
    """The stub, driven by the conversation instead of a fixed list.

    The agent calls the model once per tool round, so how far into a reply we
    are is how many AI messages have arrived since the last human one. Deriving
    that from the messages rather than holding a counter is what lets one
    instance serve a whole thread, and every thread the process is asked for.
    """

    delay: float = 0.0

    async def _astream(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        turns = [message for message in messages if message.type == "human"]
        since = list(
            itertools.takewhile(
                lambda message: message.type != "human", reversed(messages)
            )
        )
        self.script = chat_script(len(turns), turns[-1].text if turns else "")
        self.index = sum(1 for message in since if message.type == "ai")
        async for chunk in super()._astream(messages, *args, **kwargs):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


def build() -> tuple[Any, str]:
    """The model to run with, and a name for it to say so in the logs."""
    settings = get_settings()
    if not (settings.provider_model and settings.provider_api_key):
        return ScriptedChat(script=[], delay=settings.token_delay), "scripted stub"
    return (
        init_chat_model(
            settings.provider_model,
            api_key=settings.provider_api_key.get_secret_value(),
        ),
        settings.provider_model,
    )
