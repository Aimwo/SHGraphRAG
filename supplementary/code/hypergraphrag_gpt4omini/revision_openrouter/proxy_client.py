"""Public aiohttp transport with bounded total time and proxy tunnel cleanup."""
from __future__ import annotations

import asyncio

import aiohttp
import httpx


class ProxyTextClient:
    def __init__(self, proxy, workers, total_timeout_s=240):
        self.proxy = proxy
        self.session = aiohttp.ClientSession(
            trust_env=False,
            timeout=aiohttp.ClientTimeout(
                total=total_timeout_s, connect=20, sock_connect=20, sock_read=180),
            connector=aiohttp.TCPConnector(
                limit=max(4, workers + 1), limit_per_host=max(4, workers + 1),
                keepalive_timeout=15, ttl_dns_cache=300),
        )

    async def post(self, url, *, json, headers):
        try:
            async with self.session.post(
                url, json=json, headers=headers,
                proxy=self.proxy, allow_redirects=False,
            ) as response:
                content = await response.read()
                # Retain the existing response adapter, identity checks and usage
                # logging. No change to the complete Chat Completions payload.
                return httpx.Response(
                    response.status,
                    headers={name: value for name, value in response.headers.items()
                             if name.lower() not in (
                                 "content-encoding", "content-length", "transfer-encoding")},
                    content=content,
                    request=httpx.Request("POST", url))
        except aiohttp.ClientConnectorError as exc:
            raise httpx.ConnectError(type(exc).__name__) from exc
        except asyncio.TimeoutError as exc:
            raise httpx.ReadTimeout("bounded_text_request_timeout") from exc
        except aiohttp.ClientError as exc:
            raise httpx.TransportError(type(exc).__name__) from exc

    async def aclose(self):
        await self.session.close()
