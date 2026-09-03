"""Find the toolset services an index should describe.

The index reports what is *running*, which is why discovery asks the platform
on every request rather than reading a list handed to it once at deploy time:
a service that never started has to be absent from the directory, not listed
because a deploy said it would exist.

Two backends, selected by ``MCP_INDEX_DISCOVERY``:

- ``kubernetes`` (the default): list Services in this pod's namespace carrying
  the ``mcp-toolsets/toolset`` label, and address each one by its Service name.
  Needs nothing installed and no credentials beyond the pod's own service
  account.
- ``ecs``: list services on an ECS cluster tagged ``mcp-toolsets/toolset``, and
  address each one by the Cloud Map registration ECS made for it. Needs the
  ``[aws]`` extra and a task role that can read both.

Both answer the same question and return the same shape, so everything
downstream — ``describe``, the directory, the connections map — is unaware of
which platform it is on.
"""

import asyncio
import ssl
from collections.abc import Callable, Iterable
from itertools import batched
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import httpx

#: Selector for a toolset service. A Kubernetes label and an ECS tag, spelled
#: the same way so a deployment reads the same on either platform.
TOOLSET_LABEL = "mcp-toolsets/toolset"

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")

#: Port to address a toolset on when its registration does not name one.
#: Matches the port the charts and the Dockerfile use.
DEFAULT_TOOLSET_PORT = 8000

#: ``DescribeServices`` takes at most this many services per call.
ECS_DESCRIBE_BATCH = 10


class ToolsetService(NamedTuple):
    """A deployed toolset and the base URL the index can reach it on."""

    toolset: str
    base_url: str


class Discovery(Protocol):
    """One way of finding the toolsets running beside this index."""

    async def services(self) -> list[ToolsetService]:
        """Return every toolset currently running, sorted by name."""
        ...


def kubernetes_services(service_list: dict[str, Any]) -> list[ToolsetService]:
    """Extract toolset services from a Kubernetes ServiceList payload."""
    services = []
    for item in service_list.get("items", []):
        metadata = item["metadata"]
        toolset = metadata.get("labels", {}).get(TOOLSET_LABEL)
        if not toolset:
            continue
        port = item["spec"]["ports"][0]["port"]
        services.append(ToolsetService(toolset, f"http://{metadata['name']}:{port}"))
    return sorted(services)


def kubernetes_ssl_context(service_account_dir: Path) -> ssl.SSLContext | bool:
    """Trust the cluster CA when running in a pod; default verification otherwise."""
    ca = service_account_dir / "ca.crt"
    return ssl.create_default_context(cafile=str(ca)) if ca.exists() else True


class KubernetesDiscovery:
    """List toolset Services in this pod's namespace via the Kubernetes API.

    The client is built here rather than shared with the index's own, because
    only this call needs the cluster CA: the toolsets are reached over plain
    HTTP on their in-cluster names.
    """

    def __init__(
        self,
        service_account_dir: Path = SERVICE_ACCOUNT_DIR,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._dir = service_account_dir
        self._client_factory = client_factory or self._default_client

    def _default_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(verify=kubernetes_ssl_context(self._dir), timeout=5.0)

    async def services(self) -> list[ToolsetService]:
        namespace = (self._dir / "namespace").read_text().strip()
        token = (self._dir / "token").read_text().strip()
        async with self._client_factory() as client:
            response = await client.get(
                f"https://kubernetes.default.svc/api/v1/namespaces/{namespace}/services",
                params={"labelSelector": TOOLSET_LABEL},
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            return kubernetes_services(response.json())


class EcsClients(NamedTuple):
    """The two AWS clients the ECS backend reads from."""

    ecs: Any
    service_discovery: Any


def boto3_clients() -> EcsClients:
    """Build the AWS clients, naming the extra if it isn't installed."""
    try:
        import boto3
    except ModuleNotFoundError as error:  # pragma: no cover - import guard
        raise RuntimeError(
            "MCP_INDEX_DISCOVERY=ecs needs the AWS client: install "
            "mcp-toolsets-runtime[aws]"
        ) from error
    return EcsClients(boto3.client("ecs"), boto3.client("servicediscovery"))


def _tagged_toolset(service: dict[str, Any]) -> str | None:
    """The toolset a described ECS service is tagged as, if any."""
    for tag in service.get("tags", []):
        if tag.get("key") == TOOLSET_LABEL:
            return str(tag["value"]) or None
    return None


class EcsDiscovery:
    """List services on an ECS cluster, addressed by their Cloud Map names.

    ECS registers each service in Cloud Map under a name in a namespace, which
    together resolve to the running tasks inside the VPC. That registration is
    what gives the index a stable address, the way a Service name does in a
    cluster — so a toolset without one cannot be reached and is skipped.

    boto3 is synchronous, so the whole sweep runs in a worker thread.
    """

    def __init__(
        self,
        cluster: str,
        port: int = DEFAULT_TOOLSET_PORT,
        clients: Callable[[], EcsClients] = boto3_clients,
    ) -> None:
        self._cluster = cluster
        self._port = port
        self._clients = clients
        self._namespaces: dict[str, str] = {}

    def check(self) -> None:
        """Build the clients now, so a bad install or region fails at startup."""
        self._clients()

    async def services(self) -> list[ToolsetService]:
        return await asyncio.to_thread(self._collect)

    def _collect(self) -> list[ToolsetService]:
        clients = self._clients()
        services = []
        for described in self._describe(clients.ecs):
            toolset = _tagged_toolset(described)
            if not toolset:
                continue
            base_url = self._address(clients.service_discovery, described)
            if base_url is None:
                continue
            services.append(ToolsetService(toolset, base_url))
        return sorted(services)

    def _describe(self, ecs: Any) -> Iterable[dict[str, Any]]:
        """Every service on the cluster, with its tags."""
        arns: list[str] = []
        paginator = ecs.get_paginator("list_services")
        for page in paginator.paginate(cluster=self._cluster):
            arns.extend(page.get("serviceArns", []))
        for chunk in batched(arns, ECS_DESCRIBE_BATCH):
            described = ecs.describe_services(
                cluster=self._cluster, services=list(chunk), include=["TAGS"]
            )
            yield from described.get("services", [])

    def _address(self, service_discovery: Any, service: dict[str, Any]) -> str | None:
        """Build a base URL from a service's first Cloud Map registration."""
        registries = service.get("serviceRegistries") or []
        if not registries:
            return None
        registry = registries[0]
        registry_id = str(registry["registryArn"]).rsplit("/", 1)[-1]
        registered = service_discovery.get_service(Id=registry_id)["Service"]
        namespace = self._namespace_name(service_discovery, registered["NamespaceId"])
        port = registry.get("containerPort") or self._port
        return f"http://{registered['Name']}.{namespace}:{port}"

    def _namespace_name(self, service_discovery: Any, namespace_id: str) -> str:
        """Namespace names, looked up once each — usually there is one."""
        if namespace_id not in self._namespaces:
            namespace = service_discovery.get_namespace(Id=namespace_id)["Namespace"]
            self._namespaces[namespace_id] = str(namespace["Name"])
        return self._namespaces[namespace_id]
