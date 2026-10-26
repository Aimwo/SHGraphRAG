"""Regression for TLS failures occupying every slot of the proxy pool."""
import asyncio
import gzip
import unittest

import httpcore
import httpx

from .proxy_client import ProxyTextClient


class FailingTlsStream:
    async def read(self, max_bytes, timeout=None):
        return b"HTTP/1.1 200 Connection established\r\n\r\n"

    async def write(self, buffer, timeout=None):
        pass

    async def aclose(self):
        pass

    def get_extra_info(self, info):
        return None

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        raise httpcore.ConnectError("controlled_TLS_failure")


class FailingBackend:
    async def connect_tcp(self, **kwargs):
        return FailingTlsStream()

    async def sleep(self, seconds):
        pass


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_compressed_response_is_decoded_exactly_once(self):
        payload = gzip.compress(b'{"result":"OK"}')
        async def proxy(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n"
                    + b"Content-Length: " + str(len(payload)).encode()
                    + b"\r\nConnection: close\r\n\r\n" + payload)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(proxy, "127.0.0.1", 0)
        client = ProxyTextClient(
            f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", workers=1)
        try:
            result = await client.post("http://test.invalid/response", json={}, headers={})
            self.assertEqual(result.json(), {"result": "OK"})
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()

    async def test_original_httpcore_tunnel_leak_is_reproduced(self):
        pool = httpcore.AsyncHTTPProxy(
            proxy_url="http://127.0.0.1:6789", max_connections=2,
            network_backend=FailingBackend())
        try:
            for _ in range(2):
                with self.assertRaises(httpcore.ConnectError):
                    await pool.handle_async_request(httpcore.Request(
                        "POST", "https://openrouter.ai/api/v1/chat/completions",
                        extensions={"timeout": {"pool": 0.05, "connect": 0.05}}))
            with self.assertRaises(httpcore.PoolTimeout):
                await pool.handle_async_request(httpcore.Request(
                    "POST", "https://openrouter.ai/api/v1/chat/completions",
                    extensions={"timeout": {"pool": 0.05, "connect": 0.05}}))
        finally:
            await pool.aclose()

    async def test_replacement_releases_failed_tls_and_completes_next_requests(self):
        accepted = 0

        async def proxy(reader, writer):
            nonlocal accepted
            accepted += 1
            try:
                request = await reader.readuntil(b"\r\n\r\n")
                if request.startswith(b"CONNECT "):
                    # Deliberately break the TLS handshake immediately after
                    # CONNECT succeeds, matching the failure that stalled the run.
                    writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                else:
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                        b"Connection: close\r\n\r\nOK")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(proxy, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = ProxyTextClient(f"http://127.0.0.1:{port}", workers=1)
        try:
            # More failures than the connector's entire capacity must not
            # prevent subsequent requests from reaching this local proxy.
            for _ in range(7):
                with self.assertRaises(httpx.ConnectError):
                    await asyncio.wait_for(client.post(
                        "https://failure.invalid/test", json={},
                        headers={}), timeout=3)
            response = await asyncio.wait_for(client.post(
                "http://success.invalid/test", json={}, headers={}), timeout=3)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "OK")
            self.assertEqual(accepted, 8)
        finally:
            await client.aclose()
            server.close()
            await server.wait_closed()


if __name__ == "__main__":
    unittest.main()
