"""Sign MCP clients in through an OpenID Connect provider, and gate the toolsets.

The toolsets serve MCP with no authentication of their own, which is right
inside a cluster, where an agent reaches them directly. On a public host, an
MCP client — Claude, ChatGPT, Gemini — needs the MCP authorization flow:
protected-resource metadata, an OAuth authorization server, and a way to
identify itself. Most identity providers will not let arbitrary clients
register themselves, and an organisation's realm rarely can. This module is
that authorization server, in front of one confidential client of the
provider's:

- **Clients** identify themselves by registering (``/register``, RFC 7591), by
  a client ID metadata document (an https ``client_id``), or as a client a
  deployment configured in advance (``MCP_GATE_CLIENTS``, for Gemini
  Enterprise, which can do neither).
- **Sign-in** is handed on to the provider with the gate's own client, so the
  client secret never leaves this process. PKCE is used on both legs.
- **Tokens** are the provider's own. ``/<toolset>/mcp`` verifies the bearer
  (signature, issuer, expiry, ``azp``) and forwards the call to the toolset.
- **Credentials**: a deployment can pass a function that turns the verified
  token into headers for a toolset — swapping it for a per-user API key, say.

The gate keeps no state. Client IDs, the round trip through the provider and
the codes it issues are HMAC-sealed with ``MCP_GATE_KEY``, so any replica can
serve any step and a restart breaks nothing. The provider's code is single-use,
so a replayed code fails there.

Redirects are the one thing to get right: a client may only be sent back to a
URI ``MCP_GATE_REDIRECT_URIS`` allows (loopback always). Anything wider lets
anyone register a client, send a signed-in user a link, and receive that user's
token.

Mounted by the index (``mcp_runtime.index``) when ``MCP_GATE_*`` is set, so the
``/.well-known/`` documents — which clients look for at the host's root — and
every toolset are on one host. Driven by environment variables:

- ``MCP_GATE_OIDC_DISCOVERY_URL`` (required): the provider's
  ``.well-known/openid-configuration``.
- ``MCP_GATE_CLIENT_ID``, ``MCP_GATE_CLIENT_SECRET`` (required): the gate's
  client at the provider. Its valid redirect URIs must include
  ``<public url>/oauth/callback``.
- ``MCP_GATE_KEY`` (required, 32+ characters): seals client IDs and codes.
  Changing it signs every client out and makes them register again.
- ``MCP_GATE_ACCEPTED_AZP`` (JSON list, default ``[client id]``): the
  authorized parties whose tokens the toolsets accept.
- ``MCP_GATE_SCOPE`` (default ``openid email``).
- ``MCP_GATE_REDIRECT_URIS`` (JSON list, default :data:`DEFAULT_REDIRECT_URIS`):
  where clients may be sent back to. An entry ending in ``*`` is a prefix.
- ``MCP_GATE_CLIENTS`` (JSON list of ``{"client_id", "client_secret",
  "redirect_uris"}``): clients configured in advance.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import urllib.parse
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import jwt
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route
from typing_extensions import Self

from mcp_runtime.discovery import Discovery, ToolsetService

__all__ = [
    "DEFAULT_REDIRECT_URIS",
    "Credentials",
    "GateSettings",
    "PresetClient",
    "gate_routes",
]

logger = logging.getLogger(__name__)

#: The vendors' published callbacks. Loopback is allowed besides these.
DEFAULT_REDIRECT_URIS = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    # ChatGPT uses the stable one when the server returns `iss` (RFC 9207),
    # which this does; the per-connector form is for servers that do not.
    "https://chatgpt.com/connector_platform_oauth_redirect",
    "https://chatgpt.com/connector/oauth/*",
]

#: How long a sign-in may take at the provider, and a code may wait to be used.
SIGN_IN_TTL = 600
CODE_TTL = 120

#: Discovery and JWKS are re-read at most this often, and on an unknown key id.
KEYS_TTL = 3600.0
KEYS_RETRY = 60.0
DOCUMENT_TTL = 300.0
TOOLSETS_TTL = 30.0
HTTP_TIMEOUT = 10.0

#: A client ID metadata document is small; refuse anything that is not.
CIMD_MAX_BYTES = 16_384

#: Hop-by-hop headers, and the caller's own token, which a toolset never sees.
_REQUEST_DROP = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "authorization",
}
_RESPONSE_DROP = {"content-length", "connection", "transfer-encoding"}

#: ``(toolset, access token, verified claims) -> headers to add``.
Credentials = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, str]]]


class PresetClient(BaseModel):
    """A client configured in advance, for one that cannot register itself."""

    client_id: str
    client_secret: SecretStr
    redirect_uris: list[str]


class GateSettings(BaseSettings):
    """Gate configuration, validated from ``MCP_GATE_*``."""

    model_config = SettingsConfigDict(env_prefix="MCP_GATE_")

    oidc_discovery_url: str
    client_id: str
    client_secret: SecretStr
    key: SecretStr = Field(min_length=32)
    accepted_azp: list[str] = []
    scope: str = "openid email"
    redirect_uris: list[str] = Field(
        default_factory=lambda: list(DEFAULT_REDIRECT_URIS)
    )
    clients: list[PresetClient] = []

    @model_validator(mode="after")
    def _default_azp(self) -> Self:
        if not self.accepted_azp:
            self.accepted_azp = [self.client_id]
        return self


def configured() -> bool:
    """Whether the environment asks for a gate at all."""
    return any(name.startswith("MCP_GATE_") for name in os.environ)


def http_client(**kwargs: Any) -> httpx.AsyncClient:
    """Every outbound call's client: the provider, metadata documents, toolsets."""
    return httpx.AsyncClient(**{"timeout": HTTP_TIMEOUT, **kwargs})


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def s256(verifier: str) -> str:
    """The PKCE S256 challenge for a verifier."""
    return _b64(hashlib.sha256(verifier.encode()).digest())


class Sealer:
    """HMAC-sealed, typed, expiring blobs: the gate's only state.

    Sealed, not encrypted: nothing in one is secret on its own. The type stops a
    blob minted for one purpose (a client ID) being presented as another (a
    code).
    """

    def __init__(self, key: bytes) -> None:
        self._key = key

    def _sign(self, raw: bytes) -> bytes:
        return hmac.digest(self._key, raw, "sha256")

    def seal(self, kind: str, data: dict[str, Any], ttl: int | None = None) -> str:
        body = {**data, "t": kind}
        if ttl is not None:
            body["exp"] = int(time.time()) + ttl
        raw = _b64(json.dumps(body, separators=(",", ":")).encode()).encode()
        return f"{raw.decode()}.{_b64(self._sign(raw))}"

    def unseal(self, kind: str, blob: str) -> dict[str, Any] | None:
        raw, _, sig = blob.partition(".")
        try:
            if not hmac.compare_digest(_unb64(sig), self._sign(raw.encode())):
                return None
            body = json.loads(_unb64(raw))
        except ValueError:
            return None
        if not isinstance(body, dict) or body.get("t") != kind:
            return None
        if "exp" in body and body["exp"] < time.time():
            return None
        return body

    def derive(self, label: str) -> str:
        """A secret derived from the key, e.g. the gate's PKCE verifier."""
        return _b64(self._sign(label.encode()))


def redirect_allowed(uri: str, allowed: list[str]) -> bool:
    """Loopback (RFC 8252, any port), or an entry of ``allowed``."""
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    return any(
        uri.startswith(entry[:-1]) if entry.endswith("*") else uri == entry
        for entry in allowed
    )


def redirect_matches(requested: str, registered: str) -> bool:
    """Exact, except that a loopback redirect's port may vary (RFC 8252, 7.3).

    A native client registers ``http://localhost/callback`` once and listens
    on whichever port is free when it signs in.
    """
    if requested == registered:
        return True
    a, b = urllib.parse.urlsplit(requested), urllib.parse.urlsplit(registered)
    loopback = {"localhost", "127.0.0.1", "::1"}
    return (
        a.scheme == b.scheme == "http"
        and a.hostname == b.hostname
        and a.hostname in loopback
        and (a.path, a.query) == (b.path, b.query)
    )


class OAuthError(Exception):
    def __init__(self, error: str, description: str, status: int = 400) -> None:
        super().__init__(description)
        self.error, self.description, self.status = error, description, status

    def response(self) -> JSONResponse:
        return JSONResponse(
            {"error": self.error, "error_description": self.description},
            status_code=self.status,
            headers={"Cache-Control": "no-store"},
        )


class ProviderUnavailable(Exception):
    """The provider could not be read, and nothing cached stands in."""


class Provider:
    """The OpenID Connect provider: its endpoints, its keys, and its token call."""

    def __init__(self, settings: GateSettings, http: httpx.AsyncClient) -> None:
        self.settings = settings
        self.http = http
        self._config: dict[str, Any] | None = None
        self._keys: dict[str, jwt.PyJWK] = {}
        self._read_at = 0.0

    async def config(self) -> dict[str, Any]:
        if self._config is None or time.monotonic() - self._read_at > KEYS_TTL:
            return await self._refresh()
        return self._config

    async def _refresh(self) -> dict[str, Any]:
        """Re-read discovery and the keys; keep what was there if that fails.

        A provider blip must not sign everyone out: the keys already held
        still verify tokens. Only with nothing held is the failure raised.
        """
        try:
            response = await self.http.get(self.settings.oidc_discovery_url)
            response.raise_for_status()
            config = response.json()
            jwks = (await self.http.get(config["jwks_uri"])).raise_for_status().json()
            # PyJWKSet skips keys PyJWT cannot verify with — a realm's
            # encryption key, say — where PyJWK on each would raise.
            keys = jwt.PyJWKSet(jwks["keys"]).keys
        except (httpx.HTTPError, KeyError, ValueError, jwt.PyJWKSetError) as exc:
            logger.warning("could not refresh the provider's configuration: %s", exc)
            if self._config is None:
                raise ProviderUnavailable(str(exc)) from exc
            self._read_at = time.monotonic()  # retry no sooner than a fresh read
            return self._config
        self._keys = {key.key_id: key for key in keys if key.key_id}
        self._config = config
        self._read_at = time.monotonic()
        return config

    async def verify(self, token: str) -> dict[str, Any]:
        """The token's claims, or :class:`jwt.PyJWTError`."""
        config = await self.config()
        kid = str(jwt.get_unverified_header(token).get("kid"))
        # Rotation: a new key id means new keys. At most once a minute, so a
        # stream of made-up key ids cannot turn into a stream of fetches.
        if kid not in self._keys and time.monotonic() - self._read_at > KEYS_RETRY:
            await self._refresh()
        key = self._keys.get(kid)
        if key is None:
            raise jwt.InvalidTokenError(f"unknown signing key {kid!r}")
        claims: dict[str, Any] = jwt.decode(
            token,
            key.key,
            # The key's own algorithm (RS256, RS512, ES256…): a realm chooses,
            # and the JWK says. Only asymmetric ones are ever published.
            algorithms=[key.algorithm_name],
            issuer=config["issuer"],
            # Realms commonly mint no audience for a client; `azp` names the
            # client the token was issued to instead.
            options={"verify_aud": False, "require": ["exp", "iss", "sub"]},
        )
        if claims.get("azp") not in self.settings.accepted_azp:
            raise jwt.InvalidTokenError(
                f"authorized party {claims.get('azp')!r} is not accepted"
            )
        return claims

    async def token(self, form: dict[str, str]) -> dict[str, Any]:
        """Call the provider's token endpoint as the gate's client."""
        config = await self.config()
        try:
            response = await self.http.post(
                config["token_endpoint"],
                data=form,
                auth=(
                    self.settings.client_id,
                    self.settings.client_secret.get_secret_value(),
                ),
            )
        except httpx.HTTPError as exc:
            raise OAuthError(
                "temporarily_unavailable", "the provider could not be reached", 503
            ) from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code != 200:
            raise OAuthError(
                body.get("error", "invalid_grant"),
                body.get(
                    "error_description", f"the provider answered {response.status_code}"
                ),
            )
        keep = ("access_token", "token_type", "expires_in", "refresh_token", "scope")
        return {name: body[name] for name in keep if name in body}


class Gate:
    def __init__(
        self,
        settings: GateSettings,
        public_url: str,
        discovery: Discovery,
        credentials: Credentials | None = None,
    ) -> None:
        self.settings = settings
        self.issuer = public_url.rstrip("/")
        self.discovery = discovery
        self.credentials = credentials
        # One pool for every outbound call: keep-alive to the toolsets, and no
        # new TLS handshake with the provider per token call.
        self.http = http_client()
        self.provider = Provider(settings, self.http)
        self.sealer = Sealer(settings.key.get_secret_value().encode())
        self.presets = {client.client_id: client for client in settings.clients}
        self._toolsets: dict[str, ToolsetService] = {}
        self._toolsets_at = 0.0
        self._documents: dict[str, tuple[list[str], float]] = {}

    # -- Clients -----------------------------------------------------------

    async def client(self, client_id: str) -> list[str]:
        """The redirect URIs of a client, however its ID was issued."""
        if client_id in self.presets:
            return self.presets[client_id].redirect_uris
        if client_id.startswith("https://"):
            return await self.metadata_document(client_id)
        registered = self.sealer.unseal("client", client_id)
        if registered is None:
            raise OAuthError("invalid_client", "unknown client_id", 401)
        uris: list[str] = registered["r"]
        return uris

    async def metadata_document(self, client_id: str) -> list[str]:
        """A client ID metadata document's redirect URIs: the ID is its URL.

        Fetched only from a host one of the allowed redirect URIs is on, so a
        client ID cannot make the gate request an arbitrary URL, and held for
        a while, so a sign-in is not a fetch.
        """
        cached = self._documents.get(client_id)
        if cached and time.monotonic() - cached[1] < DOCUMENT_TTL:
            return cached[0]
        hosts = {
            urllib.parse.urlsplit(entry.rstrip("*")).hostname
            for entry in self.settings.redirect_uris
        }
        if urllib.parse.urlsplit(client_id).hostname not in hosts:
            raise OAuthError(
                "invalid_client", "client metadata host is not allowed", 401
            )
        try:
            response = await self.http.get(client_id)
        except httpx.HTTPError as exc:
            raise OAuthError(
                "invalid_client", "could not fetch the client metadata", 401
            ) from exc
        if response.status_code != 200 or len(response.content) > CIMD_MAX_BYTES:
            raise OAuthError(
                "invalid_client", "client metadata document unavailable", 401
            )
        try:
            document = response.json()
        except ValueError as exc:
            raise OAuthError(
                "invalid_client", "client metadata is not JSON", 401
            ) from exc
        if not isinstance(document, dict) or document.get("client_id") != client_id:
            raise OAuthError(
                "invalid_client", "client metadata names another client_id", 401
            )
        uris = [
            uri for uri in document.get("redirect_uris", []) if isinstance(uri, str)
        ]
        refused = [
            uri
            for uri in uris
            if not redirect_allowed(uri, self.settings.redirect_uris)
        ]
        if not uris or refused:
            raise OAuthError(
                "invalid_client", f"redirect URIs not allowed: {refused}", 401
            )
        self._documents[client_id] = (uris, time.monotonic())
        return uris

    # -- Metadata ----------------------------------------------------------

    async def authorization_server(self, request: Request) -> Response:
        return JSONResponse(
            {
                "issuer": self.issuer,
                "authorization_endpoint": f"{self.issuer}/oauth/authorize",
                "token_endpoint": f"{self.issuer}/oauth/token",
                "registration_endpoint": f"{self.issuer}/oauth/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": [
                    "none",
                    "client_secret_basic",
                    "client_secret_post",
                ],
                "scopes_supported": self.settings.scope.split(),
                "authorization_response_iss_parameter_supported": True,
                "client_id_metadata_document_supported": True,
            }
        )

    async def protected_resource(self, request: Request) -> Response:
        toolset = request.path_params["toolset"]
        return JSONResponse(
            {
                "resource": f"{self.issuer}/{toolset}/mcp",
                "authorization_servers": [self.issuer],
                "bearer_methods_supported": ["header"],
                "scopes_supported": self.settings.scope.split(),
            }
        )

    # -- The flow ----------------------------------------------------------

    async def register(self, request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return OAuthError("invalid_client_metadata", "body is not JSON").response()
        uris = body.get("redirect_uris") if isinstance(body, dict) else None
        if (
            not isinstance(uris, list)
            or not uris
            or not all(isinstance(u, str) for u in uris)
        ):
            return OAuthError(
                "invalid_redirect_uri", "redirect_uris is required"
            ).response()
        refused = [
            uri
            for uri in uris
            if not redirect_allowed(uri, self.settings.redirect_uris)
        ]
        if refused:
            logger.warning("registration refused for redirect URIs %s", refused)
            return OAuthError(
                "invalid_redirect_uri", f"not allowed: {refused}"
            ).response()
        return JSONResponse(
            {
                "client_id": self.sealer.seal("client", {"r": uris}),
                "client_id_issued_at": int(time.time()),
                "client_name": body.get("client_name"),
                "redirect_uris": uris,
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
            status_code=201,
        )

    async def authorize(self, request: Request) -> Response:
        query = request.query_params
        # Until the redirect URI is known to be the client's, errors are shown
        # here rather than sent anywhere.
        try:
            uris = await self.client(query.get("client_id", ""))
        except OAuthError as exc:
            return PlainTextResponse(f"{exc.error}: {exc.description}", status_code=400)
        redirect = query.get("redirect_uri") or (uris[0] if len(uris) == 1 else "")
        if not any(redirect_matches(redirect, uri) for uri in uris):
            return PlainTextResponse(
                "invalid_request: redirect_uri was not registered", status_code=400
            )

        state = query.get("state")
        if query.get("response_type") != "code":
            return self._back(redirect, state, error="unsupported_response_type")
        if query.get("code_challenge_method") != "S256" or not query.get(
            "code_challenge"
        ):
            return self._back(
                redirect,
                state,
                error="invalid_request",
                error_description="PKCE S256 is required",
            )

        nonce = secrets.token_urlsafe(16)
        sign_in = self.sealer.seal(
            "sign-in",
            {
                "c": query["client_id"],
                "r": redirect,
                "s": state,
                "cc": query["code_challenge"],
                "n": nonce,
            },
            ttl=SIGN_IN_TTL,
        )
        try:
            config = await self.provider.config()
        except ProviderUnavailable:
            return self._back(
                redirect,
                state,
                error="temporarily_unavailable",
                error_description="the identity provider could not be reached",
            )
        params = {
            "client_id": self.settings.client_id,
            "redirect_uri": f"{self.issuer}/oauth/callback",
            "response_type": "code",
            "scope": self.settings.scope,
            "state": sign_in,
            # Derived, not carried: the provider-side verifier never rides the
            # browser round trip.
            "code_challenge": s256(self.sealer.derive(f"pkce:{nonce}")),
            "code_challenge_method": "S256",
        }
        return RedirectResponse(
            f"{config['authorization_endpoint']}?{urllib.parse.urlencode(params)}",
            status_code=302,
        )

    async def callback(self, request: Request) -> Response:
        query = request.query_params
        sign_in = self.sealer.unseal("sign-in", query.get("state", ""))
        if sign_in is None:
            return PlainTextResponse(
                "This sign-in expired or is not valid. Start again from your client.",
                status_code=400,
            )
        if "error" in query:
            return self._back(
                sign_in["r"],
                sign_in["s"],
                error=query["error"],
                error_description=query.get("error_description", ""),
            )
        code = self.sealer.seal(
            "code",
            {
                "k": query.get("code", ""),
                "c": sign_in["c"],
                "r": sign_in["r"],
                "cc": sign_in["cc"],
                "n": sign_in["n"],
            },
            ttl=CODE_TTL,
        )
        return self._back(sign_in["r"], sign_in["s"], code=code)

    def _back(
        self, redirect: str, state: str | None, **params: str
    ) -> RedirectResponse:
        """Send the browser back to the client, naming this issuer (RFC 9207)."""
        params["iss"] = self.issuer
        if state is not None:
            params["state"] = state
        joiner = "&" if urllib.parse.urlsplit(redirect).query else "?"
        return RedirectResponse(
            f"{redirect}{joiner}{urllib.parse.urlencode(params)}", status_code=302
        )

    async def token(self, request: Request) -> Response:
        form = {key: str(value) for key, value in (await request.form()).items()}
        try:
            client_id = await self._authenticate(request, form)
            grant = form.get("grant_type")
            if grant == "authorization_code":
                tokens = await self._code(form, client_id)
            elif grant == "refresh_token":
                # A refresh is bound to its client as a code is (RFC 6749,
                # section 6): a configured client's secret was checked above,
                # and no client may refresh anonymously.
                if client_id is None:
                    raise OAuthError("invalid_client", "client_id is required", 401)
                tokens = await self.provider.token(
                    {
                        "grant_type": "refresh_token",
                        "refresh_token": form.get("refresh_token", ""),
                    }
                )
            else:
                raise OAuthError(
                    "unsupported_grant_type", f"{grant!r} is not supported"
                )
        except OAuthError as exc:
            return exc.response()
        return JSONResponse(
            tokens, headers={"Cache-Control": "no-store", "Pragma": "no-cache"}
        )

    async def _authenticate(self, request: Request, form: dict[str, str]) -> str | None:
        """The client ID, with a configured client's secret checked."""
        client_id, secret = form.get("client_id"), form.get("client_secret")
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                user, _, password = base64.b64decode(header[6:]).decode().partition(":")
            except ValueError as exc:
                raise OAuthError(
                    "invalid_client", "malformed basic credentials", 401
                ) from exc
            client_id = urllib.parse.unquote(user)
            secret = urllib.parse.unquote(password)
        if client_id in self.presets:
            expected = self.presets[client_id].client_secret.get_secret_value()
            if not secret or not hmac.compare_digest(
                secret.encode(), expected.encode()
            ):
                raise OAuthError("invalid_client", "client authentication failed", 401)
        return client_id

    async def _code(
        self, form: dict[str, str], client_id: str | None
    ) -> dict[str, Any]:
        code = self.sealer.unseal("code", form.get("code", ""))
        if code is None:
            raise OAuthError("invalid_grant", "unknown or expired code")
        if client_id is not None and client_id != code["c"]:
            raise OAuthError("invalid_grant", "the code was issued to another client")
        if code["c"] in self.presets and client_id is None:
            raise OAuthError("invalid_client", "client authentication required", 401)
        if form.get("redirect_uri", code["r"]) != code["r"]:
            raise OAuthError("invalid_grant", "redirect_uri does not match")
        challenge = s256(form.get("code_verifier", "")).encode()
        if not hmac.compare_digest(challenge, str(code["cc"]).encode()):
            raise OAuthError("invalid_grant", "PKCE verification failed")
        return await self.provider.token(
            {
                "grant_type": "authorization_code",
                "code": code["k"],
                "redirect_uri": f"{self.issuer}/oauth/callback",
                "code_verifier": self.sealer.derive(f"pkce:{code['n']}"),
            }
        )

    # -- The toolsets ------------------------------------------------------

    async def toolset(self, name: str) -> ToolsetService | None:
        # Misses are cached too: a loop over made-up names must not become a
        # loop over the platform's API.
        if time.monotonic() - self._toolsets_at > TOOLSETS_TTL:
            self._toolsets = {s.toolset: s for s in await self.discovery.services()}
            self._toolsets_at = time.monotonic()
        return self._toolsets.get(name)

    async def mcp(self, request: Request) -> Response:
        name = request.path_params["toolset"]
        challenge = {
            "WWW-Authenticate": (
                f'Bearer resource_metadata="{self.issuer}/.well-known/oauth-protected-resource/{name}/mcp"'
            )
        }
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return JSONResponse(
                {"error": "unauthorized"}, status_code=401, headers=challenge
            )
        token = header[7:].strip()
        try:
            claims = await self.provider.verify(token)
        except jwt.PyJWTError as exc:
            logger.info("refused a token for %s: %s", name, exc)
            challenge["WWW-Authenticate"] += ', error="invalid_token"'
            return JSONResponse(
                {"error": "invalid_token"}, status_code=401, headers=challenge
            )
        except ProviderUnavailable:
            return JSONResponse({"error": "provider unavailable"}, status_code=503)

        service = await self.toolset(name)
        if service is None:
            return JSONResponse(
                {"error": f"no toolset named {name!r}"}, status_code=404
            )

        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _REQUEST_DROP
        }
        if self.credentials is not None:
            try:
                headers.update(await self.credentials(name, token, claims))
            except Exception:
                logger.exception("credentials for %s failed", name)
                return JSONResponse(
                    {"error": "credentials unavailable"}, status_code=503
                )

        try:
            upstream = await self.http.send(
                self.http.build_request(
                    request.method,
                    f"{service.base_url}/mcp",
                    params=request.query_params,
                    headers=headers,
                    content=await request.body(),
                    # A streamed answer may stay open as long as the call runs.
                    timeout=httpx.Timeout(HTTP_TIMEOUT, read=None),
                ),
                stream=True,
            )
        except httpx.HTTPError:
            logger.exception("toolset %s unreachable", name)
            return JSONResponse(
                {"error": f"toolset {name!r} unreachable"}, status_code=502
            )

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            headers={
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in _RESPONSE_DROP
            },
        )


def gate_routes(
    settings: GateSettings,
    public_url: str,
    discovery: Discovery,
    credentials: Credentials | None = None,
) -> list[Route]:
    """The gate's routes, to mount at the root of the public host."""
    gate = Gate(settings, public_url, discovery, credentials)
    return [
        Route("/.well-known/oauth-authorization-server", gate.authorization_server),
        # Some clients look for OpenID discovery first; the same document serves.
        Route("/.well-known/openid-configuration", gate.authorization_server),
        Route(
            "/.well-known/oauth-protected-resource/{toolset}/mcp",
            gate.protected_resource,
        ),
        Route("/oauth/register", gate.register, methods=["POST"]),
        Route("/oauth/authorize", gate.authorize),
        Route("/oauth/callback", gate.callback),
        Route("/oauth/token", gate.token, methods=["POST"]),
        Route("/{toolset}/mcp", gate.mcp, methods=["GET", "POST", "DELETE"]),
    ]
