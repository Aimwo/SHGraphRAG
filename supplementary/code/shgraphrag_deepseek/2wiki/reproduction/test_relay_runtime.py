"""Provider identity, accounting, cancellation and credential boundary checks."""

import json
import hashlib
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from model.deepseek_relay import DeepSeekRelayChat, normalized_usage, response_text, active_refusal_chunk_hashes
from reproduction.relay_config import config_identity, load_relay_config
from reproduction.runtime import RequestSlots, write_json


class RelayBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {
            "status": "ready", "provider": "deepseek-relay",
            "protocol": "openai-chat-completions",
            "base_url": "https://offline.invalid/v1", "model": "offline-model",
            "allowed_returned_models": ["offline-model"],
            "credential_id": "offline-credential", "generation_parameters": {},
            "max_in_flight": 64, "request_timeout_seconds": 600,
        }
        write_json(self.root / "effective_config.json", {
            "relay_config_sha256": config_identity(self.config),
        })

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, responder):
        model = DeepSeekRelayChat(model_name="offline-model")
        model._client = httpx.Client(transport=httpx.MockTransport(responder))
        slots = RequestSlots(self.root / "slots", capacity=64)
        with (
            patch("model.deepseek_relay.load_relay_config", return_value=self.config),
            patch("model.deepseek_relay.run_root", return_value=self.root),
            patch("model.deepseek_relay.check_cancelled"),
            patch("model.deepseek_relay.RequestSlots", return_value=slots),
            patch("model.deepseek_relay.read_authorized_key", return_value="offline-secret"),
        ):
            return model.invoke([SystemMessage(content="Original system"), HumanMessage(content="Original user")])

    def test_missing_target_never_loads_credentials(self):
        pending = {**load_relay_config(require_ready=False), "status": "pending",
                   "base_url": None, "model": None}
        pending_path = self.root / "pending.json"
        write_json(pending_path, pending)
        with (
            patch("reproduction.relay_config.CONFIG_FILE", pending_path),
            patch("model.deepseek_relay.read_authorized_key") as read_key,
        ):
            with self.assertRaisesRegex(RuntimeError, "Base URL"):
                load_relay_config()
            read_key.assert_not_called()

    def test_actual_messages_usage_and_response_are_saved(self):
        seen = []

        def respond(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "offline-response", "model": "offline-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                          "prompt_cache_hit_tokens": 6,
                          "completion_tokens_details": {"reasoning_tokens": 2}},
            })

        answer = self.invoke(respond)
        self.assertEqual(answer.content, "answer")
        self.assertEqual(seen[0]["messages"], [
            {"role": "system", "content": "Original system"},
            {"role": "user", "content": "Original user"},
        ])
        self.assertNotIn("temperature", seen[0])
        self.assertNotIn("max_tokens", seen[0])
        record = json.loads(next((self.root / "requests").glob("*.record.json")).read_text())
        self.assertEqual(record["usage"]["total_tokens"], 14)
        self.assertEqual(record["usage"]["input_tokens_details"]["cached_tokens"], 6)
        self.assertEqual(record["credential_id"], "offline-credential")
        self.assertEqual(record["status"], "completed")
        self.assertTrue(list((self.root / "requests").glob("*.response.txt")))

    def test_error_body_redacts_credential_and_stops_auth_failures(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            self.invoke(lambda request: httpx.Response(401, json={
                "error": {"message": "echo offline-secret"},
            }))
        self.assertTrue((self.root / "FATAL_PROVIDER_ERROR.json").exists())
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix in (".txt", ".json"):
                self.assertNotIn("offline-secret", path.read_text())

    def test_returned_model_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "allowlist"):
            self.invoke(lambda request: httpx.Response(200, json={
                "model": "different-provider-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
            }))
        self.assertTrue((self.root / "FATAL_PROVIDER_ERROR.json").exists())

    def test_upstream_outage_stops_following_requests_before_key_loading(self):
        for status in (429, 502, 503, 504):
            marker = self.root / "FATAL_PROVIDER_ERROR.json"
            marker.unlink(missing_ok=True)
            sent = []

            def respond(request):
                sent.append(request)
                return httpx.Response(status, json={"error": {"code": "upstream_error"}})

            model = DeepSeekRelayChat(model_name="offline-model")
            model._client = httpx.Client(transport=httpx.MockTransport(respond))
            slots = RequestSlots(self.root / "slots", capacity=64)
            with (
                patch("model.deepseek_relay.load_relay_config", return_value=self.config),
                patch("model.deepseek_relay.run_root", return_value=self.root),
                patch("reproduction.runtime.run_root", return_value=self.root),
                patch("model.deepseek_relay.RequestSlots", return_value=slots),
                patch("model.deepseek_relay.read_authorized_key", return_value="offline-secret") as read_key,
            ):
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                    model.invoke([HumanMessage(content="First real work item")])
                with self.assertRaisesRegex(RuntimeError, "FATAL_PROVIDER_ERROR"):
                    model.invoke([HumanMessage(content="Queued work must remain unsent")])
                self.assertEqual(read_key.call_count, 3)
            self.assertEqual(len(sent), 3)
            self.assertTrue(json.loads(marker.read_text())["retryable_after_cooldown"])

    def test_changed_contract_is_rejected_before_key_loading(self):
        write_json(self.root / "effective_config.json", {"relay_config_sha256": "different"})
        with self.assertRaisesRegex(ValueError, "mix"):
            self.invoke(lambda request: self.fail("HTTP must not be called"))

    def test_reviewed_gateway_500_stops_queued_requests_before_key_loading(self):
        sent = []

        def respond(request):
            sent.append(request)
            return httpx.Response(500, json={"error": {
                "type": "new_api_error", "code": "do_request_failed",
                "message": "upstream error: do request failed",
            }})

        model = DeepSeekRelayChat(model_name="offline-model")
        model._client = httpx.Client(transport=httpx.MockTransport(respond))
        slots = RequestSlots(self.root / "slots", capacity=64)
        with (
            patch("model.deepseek_relay.load_relay_config", return_value=self.config),
            patch("model.deepseek_relay.run_root", return_value=self.root),
            patch("reproduction.runtime.run_root", return_value=self.root),
            patch("model.deepseek_relay.RequestSlots", return_value=slots),
            patch("model.deepseek_relay.read_authorized_key", return_value="offline-secret") as read_key,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                model.invoke([HumanMessage(content="Pending real passage")])
            with self.assertRaisesRegex(RuntimeError, "FATAL_PROVIDER_ERROR"):
                model.invoke([HumanMessage(content="Queued passage stays unsent")])
        self.assertEqual(len(sent), 3)
        self.assertEqual(read_key.call_count, 3)
        marker = json.loads((self.root / "FATAL_PROVIDER_ERROR.json").read_text())
        self.assertTrue(marker["retryable_after_cooldown"])
        self.assertEqual(marker["transport_error_kind"], "gateway_do_request_failed")

    def test_unreviewed_http500_is_not_automatically_retryable(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            self.invoke(lambda request: httpx.Response(500, json={
                "error": {"type": "other_error", "code": "unknown"},
            }))
        marker = self.root / "FATAL_PROVIDER_ERROR.json"
        if marker.exists():
            self.assertFalse(json.loads(marker.read_text())["retryable_after_cooldown"])

    def test_recoverable_failure_retries_identical_payload_and_preserves_each_attempt(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                sent = []
                before = set((self.root / "requests").glob("*.record.json"))

                def respond(request):
                    sent.append(json.loads(request.content))
                    if len(sent) == 1:
                        return httpx.Response(status, json={"error": {
                            "type": "new_api_error", "code": "do_request_failed",
                        }})
                    return httpx.Response(200, json={
                        "model": "offline-model",
                        "choices": [{"finish_reason": "stop",
                                     "message": {"content": "answer after retry"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                                  "total_tokens": 5},
                    })

                result = self.invoke(respond)
                self.assertEqual(result.content, "answer after retry")
                self.assertEqual(len(sent), 2)
                self.assertEqual(sent[0], sent[1])
                records = [
                    json.loads(path.read_text())
                    for path in set((self.root / "requests").glob("*.record.json")) - before
                ]
                records.sort(key=lambda row: row["logical_attempt"])
                self.assertEqual([row["status"] for row in records], ["failed", "completed"])
                self.assertEqual(len({row["logical_call_id"] for row in records}), 1)
                self.assertEqual(len({row["request_sha256"] for row in records}), 1)
                self.assertEqual(records[1]["retry_of_call_id"], records[0]["id"])
                self.assertIsNone(records[0]["usage"])
                self.assertEqual(records[1]["usage"]["total_tokens"], 5)
                self.assertFalse((self.root / "FATAL_PROVIDER_ERROR.json").exists())

    def test_stop_between_retry_attempts_prevents_another_request(self):
        sent = []

        def respond(request):
            sent.append(request)
            (self.root / "STOP_REQUESTED").write_text("operator stop")
            return httpx.Response(502, json={"error": {"code": "rate_limit_exceeded"}})

        model = DeepSeekRelayChat(model_name="offline-model")
        model._client = httpx.Client(transport=httpx.MockTransport(respond))
        with (
            patch("model.deepseek_relay.load_relay_config", return_value=self.config),
            patch("model.deepseek_relay.run_root", return_value=self.root),
            patch("reproduction.runtime.run_root", return_value=self.root),
            patch("model.deepseek_relay.RequestSlots",
                  return_value=RequestSlots(self.root / "slots", capacity=64)),
            patch("model.deepseek_relay.read_authorized_key", return_value="offline-secret") as key,
        ):
            with self.assertRaisesRegex(RuntimeError, "STOP_REQUESTED"):
                model.invoke([HumanMessage(content="Genuine pending item")])
        self.assertEqual(len(sent), 1)
        self.assertEqual(key.call_count, 1)

    def test_dns_failure_stops_queued_work_before_another_key_load(self):
        sent = []
        def respond(request):
            sent.append(request)
            raise httpx.ConnectError("[Errno 11004] getaddrinfo failed", request=request)
        model = DeepSeekRelayChat(model_name="offline-model")
        model._client = httpx.Client(transport=httpx.MockTransport(respond))
        slots = RequestSlots(self.root / "slots", capacity=64)
        with (
            patch("model.deepseek_relay.load_relay_config", return_value=self.config),
            patch("model.deepseek_relay.run_root", return_value=self.root),
            patch("reproduction.runtime.run_root", return_value=self.root),
            patch("model.deepseek_relay.RequestSlots", return_value=slots),
            patch("model.deepseek_relay.read_authorized_key", return_value="offline-secret") as read_key,
        ):
            with self.assertRaisesRegex(httpx.ConnectError, "getaddrinfo failed"):
                model.invoke([HumanMessage(content="Genuine pending item")])
            with self.assertRaisesRegex(RuntimeError, "FATAL_PROVIDER_ERROR"):
                model.invoke([HumanMessage(content="Queued item")])
        self.assertEqual(len(sent), 1)
        self.assertEqual(read_key.call_count, 1)
        marker = json.loads((self.root / "FATAL_PROVIDER_ERROR.json").read_text())
        self.assertEqual(marker["transport_error_kind"], "dns_resolution")
        self.assertIsNone(marker["http_status"])
        self.assertTrue(marker["retryable_after_cooldown"])
        record = json.loads(next((self.root / "requests").glob("*.record.json")).read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["transport_error_kind"], "dns_resolution")
        self.assertIsNone(record["usage"])

    def test_certificate_connect_error_is_not_classified_as_dns(self):
        def respond(request):
            raise httpx.ConnectError("CERTIFICATE_VERIFY_FAILED", request=request)
        with self.assertRaisesRegex(httpx.ConnectError, "CERTIFICATE_VERIFY_FAILED"):
            self.invoke(respond)
        self.assertFalse((self.root / "FATAL_PROVIDER_ERROR.json").exists())

    def test_explicit_refusal_is_preserved_once_and_distinguished_from_answer(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={
                "model": "offline-model",
                "choices": [{"finish_reason": "content_filter",
                             "message": {"content": "Provider declined this request."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            })

        result = self.invoke(respond)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.content, "Provider declined this request.")
        self.assertEqual(result.response_metadata["provider_outcome"], "refused")
        record = json.loads(next((self.root / "requests").glob("*.record.json")).read_text())
        self.assertEqual(record["status"], "refused")
        self.assertEqual(record["response_status"], "refused")
        self.assertEqual(record["usage"]["total_tokens"], 14)
        ledger = (self.root / "provider_refusals.jsonl").read_text().splitlines()
        self.assertEqual(len(ledger), 1)
        self.assertEqual(json.loads(ledger[0])["call_id"], record["id"])
        self.assertFalse((self.root / "FATAL_PROVIDER_ERROR.json").exists())

    def test_truncation_and_missing_content_are_not_accepted(self):
        for choice in (
            {"finish_reason": "length", "message": {"content": "truncated"}},
            {"finish_reason": "stop", "message": {"content": None}},
        ):
            with self.assertRaises(RuntimeError):
                response_text({"choices": [choice]})

    def test_successful_retry_resolves_active_refusal_without_erasing_history(self):
        cache = self.root / "cache/graph/example.pkl"
        cache.parent.mkdir(parents=True)
        text = "Provider declined this request."
        ledger = self.root / "provider_refusals.jsonl"
        ledger.write_text(json.dumps({
            "stage": "graph_extraction", "chunk_hash": "example",
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }) + "\n")
        with cache.open("wb") as stream:
            pickle.dump(text, stream)
        self.assertEqual(active_refusal_chunk_hashes(self.root), {"example"})
        with cache.open("wb") as stream:
            pickle.dump("Actual successful extraction after the user changed routing.", stream)
        self.assertEqual(active_refusal_chunk_hashes(self.root), set())
        self.assertEqual(len(ledger.read_text().splitlines()), 1)

    def test_missing_usage_and_details_are_explicit(self):
        self.assertIsNone(normalized_usage(None))
        usage = normalized_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.assertEqual(usage["input_tokens_details"], {})
        self.assertEqual(usage["output_tokens_details"], {})

    def test_global_capacity_cannot_exceed_authorized_100(self):
        with self.assertRaises(ValueError):
            RequestSlots(self.root / "slots", capacity=101)
        slots = RequestSlots(self.root / "slots", capacity=100)
        self.assertEqual(slots.capacity, 100)


if __name__ == "__main__":
    unittest.main()
