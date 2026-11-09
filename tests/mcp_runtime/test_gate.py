import base64
import hashlib
import json
import time
import urllib.parse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import mcp_runtime.gate
from mcp_runtime.discovery import ToolsetService
from mcp_runtime.gate import (
    GateSettings,
    PresetClient,
    Sealer,
    redirect_allowed,
    redirect_matches,
)
from mcp_runtime.index import build_app

PUBLIC = "https://mcp.example.com"
ISSUER = "https://idp.example.com/realms/r"
VERIFIER = "a-client-verifier-that-is-long-enough-for-pkce-0123456789"
CLAUDE = "https://claude.ai/api/mcp/auth_callback"

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
KEY_512 = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwk(key, kid: str, alg: str) -> dict:
    return {
        **json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key())),
        "kid": kid,
        "alg": alg,
        "use": "sig",
    }


# As a Keycloak realm publishes them: two signing keys and an encryption key.
JWKS = [
    jwk(KEY, "k1", "RS256"),
    jwk(KEY_512, "k512", "RS512"),
    {**jwk(KEY, "enc", "RSA-OAEP"), "use": "enc"},
]


def challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def access_token(key=KEY, kid="k1", alg="RS256", **claims) -> str:
    body = {
        "iss": ISSUER,
        "sub": "user-1",
        "azp": "gate",
        "exp": int(time.time()) + 300,
        **claims,
    }
    return jwt.encode(body, key, algorithm=alg, headers={"kid": kid})


class Provider:
    """The identity provider, a toolset and a metadata document, over one transport."""

    def __init__(self):
        self.token_calls = []
        self.toolset_calls = []
        self.document_fetches = 0
        self.down = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if self.down and url.startswith(ISSUER):
            raise httpx.ConnectError("provider down", request=request)
        if url == f"{ISSUER}/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/auth",
                    "token_endpoint": f"{ISSUER}/token",
                    "jwks_uri": f"{ISSUER}/certs",
                },
            )
        if url == f"{ISSUER}/certs":
            return httpx.Response(200, json={"keys": JWKS})
        if url == f"{ISSUER}/token":
            form = dict(urllib.parse.parse_qsl(request.content.decode()))
            self.token_calls.append((request.headers.get("authorization"), form))
            if form.get("code") == "used":
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(
                200,
                json={
                    "access_token": access_token(),
                    "refresh_token": "r1",
                    "expires_in": 300,
                    "token_type": "Bearer",
                    "id_token": "not passed on",
                },
            )
        if url == "http://mcp-cds:8000/mcp":
            self.toolset_calls.append(dict(request.headers))
            # Streamed, as a toolset's streamable HTTP answer is.
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(b'{"ok": true}'),
            )
        if url == "https://chatgpt.com/oauth/client.json":
            self.document_fetches += 1
            return httpx.Response(
                200,
                json={
                    "client_id": url,
                    "redirect_uris": [
                        "https://chatgpt.com/connector_platform_oauth_redirect"
                    ],
                },
            )
        return httpx.Response(404)


class Discovery:
    async def services(self):
        return [ToolsetService("cds", "http://mcp-cds:8000")]


@pytest.fixture
def provider(monkeypatch):
    fake = Provider()
    transport = httpx.MockTransport(fake)
    monkeypatch.setattr(
        mcp_runtime.gate,
        "http_client",
        lambda **kwargs: httpx.AsyncClient(transport=transport),
    )
    return fake


def settings(**overrides) -> GateSettings:
    return GateSettings(
        oidc_discovery_url=f"{ISSUER}/.well-known/openid-configuration",
        client_id="gate",
        client_secret="gate-secret",
        key="k" * 32,
        **overrides,
    )


def app(provider, credentials=None, **overrides) -> TestClient:
    return TestClient(
        build_app(
            PUBLIC, Discovery(), gate=settings(**overrides), credentials=credentials
        ),
        follow_redirects=False,
    )


def sign_in(client: TestClient, client_id: str, redirect: str = CLAUDE) -> str:
    """Authorize, come back from the provider, and return the gate's code."""
    response = client.get(
        "/oauth/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": redirect,
            "response_type": "code",
            "state": "s1",
            "code_challenge": challenge(VERIFIER),
            "code_challenge_method": "S256",
        },
    )
    assert response.status_code == 302, response.text
    to_provider = urllib.parse.urlsplit(response.headers["location"])
    assert to_provider.netloc == "idp.example.com"
    params = dict(urllib.parse.parse_qsl(to_provider.query))
    assert params["client_id"] == "gate"
    assert params["redirect_uri"] == f"{PUBLIC}/oauth/callback"
    assert params["code_challenge_method"] == "S256"

    back = client.get(
        "/oauth/callback", params={"code": "kc-code", "state": params["state"]}
    )
    assert back.status_code == 302
    location = urllib.parse.urlsplit(back.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == redirect
    query = dict(urllib.parse.parse_qsl(location.query))
    assert query["state"] == "s1"
    assert query["iss"] == PUBLIC  # RFC 9207
    return query["code"]


def register(client: TestClient, *uris: str) -> httpx.Response:
    return client.post(
        "/oauth/register", json={"redirect_uris": list(uris), "client_name": "t"}
    )


def test_redirects_allow_loopback_and_the_list_only():
    allowed = [
        "https://claude.ai/api/mcp/auth_callback",
        "https://chatgpt.com/connector/oauth/*",
    ]
    assert redirect_allowed("http://localhost:52799/callback", allowed)
    assert redirect_allowed("http://127.0.0.1/cb", allowed)
    assert redirect_allowed("https://claude.ai/api/mcp/auth_callback", allowed)
    assert redirect_allowed("https://chatgpt.com/connector/oauth/abc", allowed)
    assert not redirect_allowed("https://claude.ai/api/mcp/auth_callback/x", allowed)
    assert not redirect_allowed("https://localhost/cb", allowed)
    assert not redirect_allowed("https://evil.example/cb", allowed)


def test_a_loopback_redirect_may_use_any_port():
    assert redirect_matches(
        "http://localhost:57456/callback", "http://localhost/callback"
    )
    assert redirect_matches("http://127.0.0.1:1/callback", "http://127.0.0.1/callback")
    assert not redirect_matches("http://localhost:1/other", "http://localhost/callback")
    assert not redirect_matches(
        "http://127.0.0.1:1/callback", "http://localhost/callback"
    )
    assert not redirect_matches("https://claude.ai/x:1", "https://claude.ai/x")


def test_sealed_blobs_are_typed_and_tamper_evident():
    sealer = Sealer(b"k" * 32)
    blob = sealer.seal("client", {"r": ["x"]})
    assert sealer.unseal("client", blob) == {"r": ["x"], "t": "client"}
    assert sealer.unseal("code", blob) is None
    raw, _, sig = blob.partition(".")
    forged = base64.urlsafe_b64encode(b'{"r":["y"],"t":"client"}').rstrip(b"=").decode()
    assert sealer.unseal("client", f"{forged}.{sig}") is None
    assert Sealer(b"o" * 32).unseal("client", blob) is None
    assert sealer.unseal("code", sealer.seal("code", {}, ttl=-1)) is None


def test_metadata(provider):
    client = app(provider)
    server = client.get("/.well-known/oauth-authorization-server").json()
    assert server["issuer"] == PUBLIC
    assert server["authorization_response_iss_parameter_supported"] is True
    assert server["client_id_metadata_document_supported"] is True
    resource = client.get("/.well-known/oauth-protected-resource/cds/mcp").json()
    assert resource == {
        "resource": f"{PUBLIC}/cds/mcp",
        "authorization_servers": [PUBLIC],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["openid", "email"],
    }


def test_registration_refuses_an_unlisted_redirect(provider):
    client = app(provider)
    response = register(client, CLAUDE, "https://evil.example/cb")
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_redirect_uri"


def test_a_registered_client_signs_in_and_calls_a_toolset(provider):
    seen = []

    async def credentials(toolset, token, claims):
        seen.append((toolset, claims["sub"]))
        return {"x-cds-token": "pat-for-user-1"}

    client = app(provider, credentials=credentials)
    client_id = register(client, CLAUDE).json()["client_id"]
    code = sign_in(client, client_id)

    tokens = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": CLAUDE,
            "code_verifier": VERIFIER,
        },
    )
    assert tokens.status_code == 200, tokens.text
    body = tokens.json()
    assert set(body) == {"access_token", "refresh_token", "expires_in", "token_type"}

    # The gate redeemed the provider's code as its own client, with its own PKCE.
    auth, form = provider.token_calls[-1]
    assert auth.startswith("Basic ")
    assert form["code"] == "kc-code"
    assert form["redirect_uri"] == f"{PUBLIC}/oauth/callback"
    assert form["code_verifier"]

    called = client.post(
        "/cds/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert called.status_code == 200
    assert called.json() == {"ok": True}
    forwarded = provider.toolset_calls[-1]
    assert forwarded["x-cds-token"] == "pat-for-user-1"
    assert "authorization" not in forwarded  # the user's token stays at the gate
    assert seen == [("cds", "user-1")]


def test_a_code_needs_the_right_verifier_and_client(provider):
    client = app(provider)
    client_id = register(client, CLAUDE).json()["client_id"]
    other = register(client, "http://localhost:1/cb").json()["client_id"]
    code = sign_in(client, client_id)
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": CLAUDE}

    wrong_verifier = client.post(
        "/oauth/token", data={**form, "code_verifier": "x" * 50}
    )
    assert wrong_verifier.json()["error"] == "invalid_grant"
    wrong_client = client.post(
        "/oauth/token", data={**form, "client_id": other, "code_verifier": VERIFIER}
    )
    assert wrong_client.json()["error"] == "invalid_grant"
    # A client ID is not a code.
    swapped = client.post(
        "/oauth/token", data={**form, "code": client_id, "code_verifier": VERIFIER}
    )
    assert swapped.json()["error"] == "invalid_grant"
    assert provider.token_calls == []


def test_authorize_will_not_redirect_anywhere_unregistered(provider):
    client = app(provider)
    client_id = register(client, CLAUDE).json()["client_id"]
    response = client.get(
        "/oauth/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": "http://localhost:9/elsewhere",
            "response_type": "code",
            "code_challenge": challenge(VERIFIER),
            "code_challenge_method": "S256",
        },
    )
    assert response.status_code == 400
    assert "location" not in response.headers


def test_authorize_requires_pkce(provider):
    client = app(provider)
    client_id = register(client, CLAUDE).json()["client_id"]
    response = client.get(
        "/oauth/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": CLAUDE,
            "response_type": "code",
        },
    )
    assert response.status_code == 302
    query = dict(
        urllib.parse.parse_qsl(
            urllib.parse.urlsplit(response.headers["location"]).query
        )
    )
    assert query["error"] == "invalid_request"
    assert query["iss"] == PUBLIC


def test_a_metadata_document_client(provider):
    client = app(provider)
    client_id = "https://chatgpt.com/oauth/client.json"
    code = sign_in(
        client, client_id, "https://chatgpt.com/connector_platform_oauth_redirect"
    )
    tokens = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "code_verifier": VERIFIER,
        },
    )
    assert tokens.status_code == 200


def test_a_metadata_document_on_an_unlisted_host_is_not_fetched(provider, monkeypatch):
    fetched = []
    monkeypatch.setattr(
        mcp_runtime.gate,
        "http_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: fetched.append(r) or httpx.Response(500)
            )
        ),
    )
    client = app(provider)
    response = client.get(
        "/oauth/authorize",
        params={
            "client_id": "https://kubernetes.default.svc/x",
            "response_type": "code",
        },
    )
    assert response.status_code == 400
    assert fetched == []


def test_a_preset_client_must_authenticate(provider):
    preset = PresetClient(
        client_id="gemini",
        client_secret="gemini-secret",
        redirect_uris=["https://vertexaisearch.cloud.google.com/oauth-redirect"],
    )
    client = app(provider, clients=[preset])
    redirect = "https://vertexaisearch.cloud.google.com/oauth-redirect"
    form = {
        "grant_type": "authorization_code",
        "code_verifier": VERIFIER,
        "redirect_uri": redirect,
    }

    code = sign_in(client, "gemini", redirect)
    assert client.post("/oauth/token", data={**form, "code": code}).status_code == 401
    wrong = client.post(
        "/oauth/token",
        data={**form, "code": code, "client_id": "gemini", "client_secret": "no"},
    )
    assert wrong.status_code == 401
    right = client.post(
        "/oauth/token", data={**form, "code": code}, auth=("gemini", "gemini-secret")
    )
    assert right.status_code == 200


def test_the_toolsets_refuse_without_a_valid_token(provider):
    client = app(provider)
    missing = client.post("/cds/mcp", json={})
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == (
        f'Bearer resource_metadata="{PUBLIC}/.well-known/oauth-protected-resource/cds/mcp"'
    )
    for token in (
        access_token(azp="someone-else"),
        access_token(iss="https://other.example.com"),
        access_token(exp=int(time.time()) - 10),
        "not-a-jwt",
    ):
        refused = client.post(
            "/cds/mcp", json={}, headers={"Authorization": f"Bearer {token}"}
        )
        assert refused.status_code == 401, token
        assert 'error="invalid_token"' in refused.headers["www-authenticate"]
    assert provider.toolset_calls == []


def test_refresh_goes_to_the_provider_for_a_named_client(provider):
    client = app(provider)
    client_id = register(client, CLAUDE).json()["client_id"]
    form = {"grant_type": "refresh_token", "refresh_token": "r1"}
    anonymous = client.post("/oauth/token", data=form)
    assert anonymous.status_code == 401
    assert provider.token_calls == []
    response = client.post("/oauth/token", data={**form, "client_id": client_id})
    assert response.status_code == 200
    assert provider.token_calls[-1][1] == form


def test_a_preset_client_refreshes_with_its_secret(provider):
    preset = PresetClient(
        client_id="gemini", client_secret="s" * 8, redirect_uris=["http://localhost/cb"]
    )
    client = app(provider, clients=[preset])
    form = {"grant_type": "refresh_token", "refresh_token": "r1", "client_id": "gemini"}
    assert client.post("/oauth/token", data=form).status_code == 401
    # A non-ASCII secret is wrong, not a crash.
    assert (
        client.post("/oauth/token", data={**form, "client_secret": "é"}).status_code
        == 401
    )
    assert (
        client.post("/oauth/token", data=form, auth=("gemini", "s" * 8)).status_code
        == 200
    )


def test_tokens_signed_with_any_published_signing_key_verify(provider):
    client = app(provider)
    rs512 = access_token(key=KEY_512, kid="k512", alg="RS512")
    response = client.post(
        "/cds/mcp", json={}, headers={"Authorization": f"Bearer {rs512}"}
    )
    assert response.status_code == 200
    # The encryption key is in the JWKS too, but is no key to verify with.
    enc = access_token(kid="enc")
    refused = client.post(
        "/cds/mcp", json={}, headers={"Authorization": f"Bearer {enc}"}
    )
    assert refused.status_code == 401


def gate_of(client: TestClient):
    route = next(
        r
        for r in client.app.router.routes
        if getattr(r, "path", "") == "/{toolset}/mcp"
    )
    return route.endpoint.__self__


def test_a_provider_blip_keeps_the_keys_already_held(provider):
    client = app(provider)
    headers = {"Authorization": f"Bearer {access_token()}"}
    assert client.post("/cds/mcp", json={}, headers=headers).status_code == 200
    provider.down = True
    gate_of(client).provider._read_at = 0.0  # the hourly re-read is due, and fails
    assert client.post("/cds/mcp", json={}, headers=headers).status_code == 200


def test_a_provider_down_from_the_start_is_a_503(provider):
    provider.down = True
    client = app(provider)
    headers = {"Authorization": f"Bearer {access_token()}"}
    assert client.post("/cds/mcp", json={}, headers=headers).status_code == 503
    client_id = register(client, CLAUDE).json()["client_id"]
    authorize = client.get(
        "/oauth/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": CLAUDE,
            "response_type": "code",
            "code_challenge": challenge(VERIFIER),
            "code_challenge_method": "S256",
        },
    )
    assert authorize.status_code == 302
    location = urllib.parse.urlsplit(authorize.headers["location"])
    assert (
        dict(urllib.parse.parse_qsl(location.query))["error"]
        == "temporarily_unavailable"
    )


def test_a_metadata_document_is_fetched_once_per_while(provider):
    client = app(provider)
    client_id = "https://chatgpt.com/oauth/client.json"
    redirect = "https://chatgpt.com/connector_platform_oauth_redirect"
    sign_in(client, client_id, redirect)
    sign_in(client, client_id, redirect)
    assert provider.document_fetches == 1


def test_unknown_toolsets_do_not_relist_the_platform(provider):
    listed = 0

    class Counting:
        async def services(self):
            nonlocal listed
            listed += 1
            return [ToolsetService("cds", "http://mcp-cds:8000")]

    client = TestClient(build_app(PUBLIC, Counting(), gate=settings()))
    headers = {"Authorization": f"Bearer {access_token()}"}
    for name in ("a", "b", "c", "cds"):
        client.post(f"/{name}/mcp", json={}, headers=headers)
    assert listed == 1


def test_the_index_lists_internal_addresses_on_request(provider, monkeypatch):
    async def healthy(client, service, public_url):
        from mcp_runtime.index import ToolsetEntry

        url = f"{public_url}/cds/mcp" if public_url else f"{service.base_url}/mcp"
        return ToolsetEntry(name="cds", url=url, status="ok", tools=[])

    monkeypatch.setattr("mcp_runtime.index.describe", healthy)
    client = app(provider)
    assert client.get("/").json()["connections"]["cds"]["url"] == f"{PUBLIC}/cds/mcp"
    internal = client.get("/", params={"network": "internal"}).json()
    assert internal["connections"]["cds"]["url"] == "http://mcp-cds:8000/mcp"


def test_no_gate_without_its_settings(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("MCP_GATE_"):
            monkeypatch.delenv(name)
    assert not mcp_runtime.gate.configured()
    plain = TestClient(build_app(PUBLIC, Discovery()))
    assert plain.get("/.well-known/oauth-authorization-server").status_code == 404
