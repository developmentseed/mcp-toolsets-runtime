import sys

import httpx
import pytest

from mcp_runtime.discovery import (
    EcsClients,
    EcsDiscovery,
    KubernetesDiscovery,
    ToolsetService,
    kubernetes_services,
)


def service_item(name: str, toolset: str | None, port: int = 8000) -> dict:
    labels = {"mcp-toolsets/toolset": toolset} if toolset else {}
    return {
        "metadata": {"name": name, "labels": labels},
        "spec": {"ports": [{"port": port}]},
    }


def test_kubernetes_services_parses_and_sorts():
    payload = {
        "items": [
            service_item("mcp-hello", "hello", port=9000),
            service_item("mcp-dataset-search", "dataset-search"),
        ]
    }
    assert kubernetes_services(payload) == [
        ToolsetService("dataset-search", "http://mcp-dataset-search:8000"),
        ToolsetService("hello", "http://mcp-hello:9000"),
    ]


def test_kubernetes_services_skips_unlabelled():
    payload = {"items": [service_item("mcp-index", None)]}
    assert kubernetes_services(payload) == []


def kubernetes_pod(tmp_path, handler):
    """A discovery backend reading a fake service account dir over a fake API."""
    (tmp_path / "namespace").write_text("mcp-toolsets\n")
    (tmp_path / "token").write_text("a-token\n")
    return KubernetesDiscovery(
        service_account_dir=tmp_path,
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )


async def test_kubernetes_discovery_queries_its_own_namespace(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        return httpx.Response(200, json={"items": [service_item("mcp-hello", "hello")]})

    services = await kubernetes_pod(tmp_path, handler).services()

    assert services == [ToolsetService("hello", "http://mcp-hello:8000")]
    assert seen["url"] == (
        "https://kubernetes.default.svc/api/v1/namespaces/mcp-toolsets/services"
        "?labelSelector=mcp-toolsets%2Ftoolset"
    )
    assert seen["authorization"] == "Bearer a-token"


async def test_kubernetes_discovery_raises_on_an_api_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "forbidden"})

    with pytest.raises(httpx.HTTPStatusError):
        await kubernetes_pod(tmp_path, handler).services()


class FakeEcs:
    """The two ECS calls the backend makes, over a canned cluster."""

    def __init__(self, services: list[dict]):
        self._services = services
        self.described: list[list[str]] = []

    def get_paginator(self, name: str):
        assert name == "list_services"
        arns = [service["serviceArn"] for service in self._services]
        return FakePaginator(arns)

    def describe_services(self, cluster: str, services: list[str], include: list[str]):
        assert include == ["TAGS"]
        self.described.append(services)
        by_arn = {service["serviceArn"]: service for service in self._services}
        return {"services": [by_arn[arn] for arn in services]}


class FakePaginator:
    def __init__(self, arns: list[str]):
        self._arns = arns

    def paginate(self, cluster: str):
        # Two pages, so the backend has to accumulate rather than take the first.
        yield {"serviceArns": self._arns[:1]}
        yield {"serviceArns": self._arns[1:]}


class FakeServiceDiscovery:
    """Cloud Map registrations, counting lookups so caching is observable."""

    def __init__(self, services: dict[str, tuple[str, str]]):
        self._services = services
        self.service_lookups = 0
        self.namespace_lookups = 0

    def get_service(self, Id: str):  # noqa: N803 - boto3's own casing
        self.service_lookups += 1
        name, namespace_id = self._services[Id]
        return {"Service": {"Name": name, "NamespaceId": namespace_id}}

    def get_namespace(self, Id: str):  # noqa: N803 - boto3's own casing
        self.namespace_lookups += 1
        return {"Namespace": {"Name": "mcp.internal"}}


def ecs_service(
    name: str,
    toolset: str | None,
    registry: str | None = "srv-1",
    container_port: int | None = None,
    port: int | None = None,
):
    registries = []
    if registry:
        entry: dict = {
            "registryArn": f"arn:aws:servicediscovery:eu-west-1:1:service/{registry}"
        }
        if container_port:
            entry["containerPort"] = container_port
        if port:
            entry["port"] = port
        registries = [entry]
    return {
        "serviceArn": f"arn:aws:ecs:eu-west-1:1:service/mcp/{name}",
        "serviceName": name,
        "serviceRegistries": registries,
        "tags": [{"key": "mcp-toolsets/toolset", "value": toolset}] if toolset else [],
    }


def ecs_discovery(services: list[dict], registrations: dict, **kwargs):
    ecs = FakeEcs(services)
    service_discovery = FakeServiceDiscovery(registrations)
    builds = []

    def clients():
        builds.append(1)
        return EcsClients(ecs, service_discovery)

    backend = EcsDiscovery("mcp-toolsets", clients=clients, **kwargs)
    backend.builds = builds  # type: ignore[attr-defined]
    return backend, ecs, service_discovery


async def test_ecs_discovery_addresses_toolsets_by_their_registration():
    backend, _, service_discovery = ecs_discovery(
        [
            ecs_service("mcp-hello", "hello", registry="srv-hello"),
            ecs_service("mcp-stac", "stac-explorer", registry="srv-stac"),
        ],
        {"srv-hello": ("mcp-hello", "ns-1"), "srv-stac": ("mcp-stac", "ns-1")},
    )

    assert await backend.services() == [
        ToolsetService("hello", "http://mcp-hello.mcp.internal:8000"),
        ToolsetService("stac-explorer", "http://mcp-stac.mcp.internal:8000"),
    ]
    # One namespace, looked up once however many services sit in it.
    assert service_discovery.namespace_lookups == 1


async def test_ecs_discovery_skips_untagged_services():
    backend, _, _ = ecs_discovery(
        [
            ecs_service("mcp-index", None, registry="srv-index"),
            ecs_service("mcp-hello", "hello", registry="srv-hello"),
        ],
        {"srv-index": ("mcp-index", "ns-1"), "srv-hello": ("mcp-hello", "ns-1")},
    )

    assert await backend.services() == [
        ToolsetService("hello", "http://mcp-hello.mcp.internal:8000")
    ]


async def test_ecs_discovery_skips_a_service_with_no_registration():
    """Tagged but unregistered means tagged but unreachable, so it is not listed."""
    backend, _, _ = ecs_discovery(
        [ecs_service("mcp-hello", "hello", registry=None)],
        {},
    )

    assert await backend.services() == []


async def test_ecs_discovery_prefers_the_registered_port():
    by_container, _, _ = ecs_discovery(
        [ecs_service("mcp-hello", "hello", registry="srv-hello", container_port=9000)],
        {"srv-hello": ("mcp-hello", "ns-1")},
    )
    by_host, _, _ = ecs_discovery(
        [ecs_service("mcp-hello", "hello", registry="srv-hello", port=9001)],
        {"srv-hello": ("mcp-hello", "ns-1")},
    )

    assert (await by_container.services())[0].base_url.endswith(":9000")
    assert (await by_host.services())[0].base_url.endswith(":9001")


async def test_ecs_discovery_builds_its_clients_once():
    """Building a boto3 client is the slow part; a startup check and every
    request after it share one pair."""
    backend, _, _ = ecs_discovery(
        [ecs_service("mcp-hello", "hello", registry="srv-hello")],
        {"srv-hello": ("mcp-hello", "ns-1")},
    )

    backend.check()
    await backend.services()
    await backend.services()

    assert backend.builds == [1]  # type: ignore[attr-defined]


async def test_ecs_discovery_looks_a_registration_up_once():
    """What can change between requests is which services exist and how they
    are tagged; a registration's name cannot, so only the first request pays."""
    backend, ecs, service_discovery = ecs_discovery(
        [
            ecs_service("mcp-hello", "hello", registry="srv-hello"),
            ecs_service("mcp-stac", "stac-explorer", registry="srv-stac"),
        ],
        {"srv-hello": ("mcp-hello", "ns-1"), "srv-stac": ("mcp-stac", "ns-1")},
    )

    first = await backend.services()
    second = await backend.services()

    assert first == second
    assert len(ecs.described) == 2  # the sweep itself runs every time
    assert service_discovery.service_lookups == 2  # one per registration, ever
    assert service_discovery.namespace_lookups == 1


async def test_ecs_discovery_falls_back_to_the_configured_port():
    backend, _, _ = ecs_discovery(
        [ecs_service("mcp-hello", "hello", registry="srv-hello")],
        {"srv-hello": ("mcp-hello", "ns-1")},
    )
    ported, _, _ = ecs_discovery(
        [ecs_service("mcp-hello", "hello", registry="srv-hello")],
        {"srv-hello": ("mcp-hello", "ns-1")},
        port=8080,
    )

    assert (await backend.services())[0].base_url.endswith(":8000")
    assert (await ported.services())[0].base_url.endswith(":8080")


async def test_ecs_discovery_batches_describe_calls():
    """DescribeServices takes ten at a time, so eleven services is two calls."""
    services = [
        ecs_service(f"mcp-t{n:02d}", f"t{n:02d}", registry=f"srv-{n:02d}")
        for n in range(11)
    ]
    registrations = {f"srv-{n:02d}": (f"mcp-t{n:02d}", "ns-1") for n in range(11)}
    backend, ecs, _ = ecs_discovery(services, registrations)

    found = await backend.services()

    assert len(found) == 11
    assert [len(call) for call in ecs.described] == [10, 1]


def test_boto3_clients_names_the_extra_when_it_is_missing(monkeypatch):
    from mcp_runtime import discovery

    # None in sys.modules is how the import system spells "not installed".
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(RuntimeError, match=r"mcp-toolsets-runtime\[aws\]"):
        discovery.boto3_clients()
