"""Versioned, user-approved visible-summary policy for the DeepSeek backend."""
from __future__ import annotations

from collections import Counter
import uuid

from .common import ITEM, append_jsonl, file_hash, read_json, read_jsonl, signature
from .transport import Transport
from hypergraphrag.utils import (
    LLMUnavailableError, LLMTruncationError,
    encode_string_by_tiktoken, decode_tokens_by_tiktoken,
)


POLICY = {
    "name": "user_approved_bounded_visible_summary_20261025",
    "visible_target_tokens": 500,
    "visible_tokenizer": "gpt-4o-mini/o200k_base",
    "physical_generation_budget_tokens": 8192,
    "length_constraint": (
        "\n\nLength constraint: The final summary must be no more than 500 tokens. "
        "Return only that concise final summary."
    ),
    "nonempty_provider_length_output": "use, label provider truncation",
    "visible_text_over_target": "use prefix of at most 500 native tokenizer tokens, label local truncation",
    "empty_or_filtered_output": "unavailable; retain original extracted description through the existing explicit fallback",
    "legacy_summary": "reuse exact original completed or nonempty length-terminated output",
    "other_stages": "unchanged",
    "authorization": "User explicitly approved a reasonable, complete and fair HyperGraphRAG run on 2026-10-25, with a soft summary output limit and truncated nonempty summaries accepted as results; 100 percent stage success is not required.",
}


def bounded_text(raw, limit=500):
    tokens = encode_string_by_tiktoken(raw, model_name="gpt-4o-mini")
    if len(tokens) <= limit:
        return raw, len(tokens), len(tokens)
    used_tokens = tokens[:limit]
    used = decode_tokens_by_tiktoken(used_tokens, model_name="gpt-4o-mini")
    # A token boundary can split a Unicode scalar. Keep a valid text prefix.
    while used_tokens and (
            "\ufffd" in used and "\ufffd" not in raw
            or len(encode_string_by_tiktoken(used, model_name="gpt-4o-mini")) > limit):
        used_tokens = used_tokens[:-1]
        used = decode_tokens_by_tiktoken(used_tokens, model_name="gpt-4o-mini")
    if not used.strip() or not raw.startswith(used):
        raise ValueError("Summary clipping did not produce a nonempty original text prefix")
    return used, len(tokens), len(encode_string_by_tiktoken(used, model_name="gpt-4o-mini"))


class SummaryTransport(Transport):
    async def complete(self, prompt, system_prompt=None, history_messages=None,
                       max_tokens=None, **kwargs):
        if max_tokens != 500 or "comprehensive summary" not in prompt:
            return await super().complete(
                prompt, system_prompt=system_prompt, history_messages=history_messages,
                max_tokens=max_tokens, **kwargs,
            )
        if system_prompt or history_messages or set(kwargs) - {"hashing_kv", "keyword_extraction"}:
            raise ValueError("The audited native summary call shape changed")
        native_body = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, **self.config["generation_parameters"], "max_tokens": 500,
        }
        legacy_key = signature({"scope": self.scope_hash, "kind": "text", "body": native_body})
        legacy_path = self.cache / "responses" / (legacy_key + ".json")
        selected_key, raw, provider_truncated = None, None, False
        variant = "legacy_exact_summary"
        if legacy_path.exists():
            try:
                raw = await super().complete(prompt, max_tokens=500)
                selected_key = legacy_key
            except LLMTruncationError as exc:
                if exc.raw_text.strip():
                    raw, selected_key, provider_truncated = exc.raw_text, legacy_key, True
            except LLMUnavailableError:
                # A filtered/incomplete output is never accepted as a summary.
                # A budget change is not a mechanism for retrying refusals.
                self._summary_record(prompt, legacy_key, variant, status="unavailable")
                raise
        if raw is None:
            variant = "bounded_visible_summary"
            revised = prompt + POLICY["length_constraint"]
            body = {**native_body,
                    "messages": [{"role": "user", "content": revised}],
                    "max_tokens": POLICY["physical_generation_budget_tokens"]}
            selected_key = signature({"scope": self.scope_hash, "kind": "text", "body": body})
            try:
                raw = await super().complete(
                    revised, max_tokens=POLICY["physical_generation_budget_tokens"]
                )
            except LLMTruncationError as exc:
                if not exc.raw_text.strip():
                    self._summary_record(prompt, selected_key, variant, status="unavailable")
                    raise
                raw, provider_truncated = exc.raw_text, True
            except LLMUnavailableError:
                self._summary_record(prompt, selected_key, variant, status="unavailable")
                raise
        used, raw_tokens, used_tokens = bounded_text(raw)
        self._summary_record(
            prompt, selected_key, variant, status="used",
            raw_visible_text=raw, used_text=used,
            raw_visible_tokens=raw_tokens, used_visible_tokens=used_tokens,
            provider_truncated=provider_truncated,
            locally_truncated=raw_tokens > POLICY["visible_target_tokens"],
        )
        return used

    def _summary_record(self, prompt, cache_key, variant, **fields):
        path = self.cache / "responses" / (cache_key + ".json")
        saved = read_json(path)
        append_jsonl(self.log_root / "summary_outputs.jsonl", {
            **ITEM.get(), "summary_call_id": uuid.uuid4().hex,
            "policy_name": POLICY["name"], "native_prompt": prompt,
            "native_prompt_sha256": signature(prompt), "source_variant": variant,
            "source_cache_key": cache_key,
            "source_cache_file_sha256": file_hash(path),
            "source_request_sha256": saved["request_sha256"],
            "source_status": saved["status"], **fields,
        })


def audit_summary_outputs(run, graph, index_directory):
    records = read_jsonl(run / "summary_outputs.jsonl")
    latest = [row for row in records if row.get("index_directory") == index_directory]
    descriptions = {str(data.get("description", "")) for _, data in graph.nodes(data=True)}
    for row in records:
        path = run / "cache/responses" / (row["source_cache_key"] + ".json")
        if file_hash(path) != row["source_cache_file_sha256"]:
            raise ValueError("A used summary response cache changed")
        saved = read_json(path)
        if (saved["request_sha256"] != row["source_request_sha256"]
                or saved["status"] != row["source_status"]
                or signature(row["native_prompt"]) != row["native_prompt_sha256"]):
            raise ValueError("Summary selection identity changed")
        if row["status"] == "used":
            raw = saved["text"] if saved["status"] == "completed" else saved["raw_text"]
            expected, raw_tokens, used_tokens = bounded_text(raw)
            if (raw != row["raw_visible_text"] or expected != row["used_text"]
                    or raw_tokens != row["raw_visible_tokens"]
                    or used_tokens != row["used_visible_tokens"]
                    or row["provider_truncated"] != (saved["status"] == "output_truncated")
                    or row["locally_truncated"] != (raw_tokens > POLICY["visible_target_tokens"])):
                raise ValueError("A saved summary differs from its actual selected output")
            if row.get("index_directory") == index_directory and row["used_text"] not in descriptions:
                raise ValueError("A selected final-build summary is absent from the graph")
    return {
        "policy": POLICY, "all_build_summary_outcomes": dict(Counter(r["status"] for r in records)),
        "final_build_summary_outcomes": dict(Counter(r["status"] for r in latest)),
        "final_build_used_summaries": sum(r["status"] == "used" for r in latest),
        "final_build_untruncated_used": sum(
            r["status"] == "used" and not r["provider_truncated"]
            and not r["locally_truncated"] for r in latest),
        "final_build_truncated_used": sum(
            r["status"] == "used" and (r["provider_truncated"] or r["locally_truncated"])
            for r in latest),
        "final_build_provider_truncated_used": sum(
            r["status"] == "used" and r["provider_truncated"] for r in latest),
        "final_build_locally_truncated_used": sum(
            r["status"] == "used" and r["locally_truncated"] for r in latest),
        "final_build_unavailable_summaries": sum(r["status"] == "unavailable" for r in latest),
        "final_build_summary_source_variants": dict(Counter(
            r["source_variant"] for r in latest if r["status"] == "used")),
        "summary_outputs_file_sha256": (
            file_hash(run / "summary_outputs.jsonl")
            if (run / "summary_outputs.jsonl").exists() else None),
    }
