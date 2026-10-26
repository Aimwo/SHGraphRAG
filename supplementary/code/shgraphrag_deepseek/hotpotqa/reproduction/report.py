"""Generate an offline, evidence-based progress/final report from run artifacts."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import time

import numpy as np
import psutil

from reproduction.campaign import DEFAULT_RUN, read_records, usage_summary
from reproduction.runtime import completed_interval_peak, utc_now, write_json
from reproduction.freeze_benchmark import DESTINATION
from reproduction.database import NEO4J_HOME


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may not have finished its final line yet.
            continue
    return records


def distribution(values):
    values = np.array([value for value in values if isinstance(value, (int, float))])
    if not len(values):
        return None
    return {
        "count": len(values), "sum": float(values.sum()), "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def storage_evidence():
    groups = Counter()
    data_root = NEO4J_HOME / "data"
    for path in data_root.rglob("*"):
        try:
            if not path.is_file():
                continue
            relative = path.relative_to(data_root).as_posix()
            if relative.startswith("databases/neo4j/schema/index/"):
                category = "neo4j_schema_index_file_bytes"
            elif relative.startswith("databases/neo4j/"):
                category = "neo4j_store_file_bytes_excluding_schema_indexes"
            elif relative.startswith("transactions/"):
                category = "transaction_file_bytes"
            elif relative.startswith("databases/system/"):
                category = "system_database_file_bytes"
            else:
                category = "other_data_file_bytes"
            groups[category] += path.stat().st_size
        except FileNotFoundError:
            continue
    return {
        **dict(groups), "total_data_file_bytes": sum(groups.values()),
        "note": (
            "Logical file lengths from stat, including Neo4j preallocation; "
            "not an estimate of graph-only bytes or physical allocated disk blocks"
        ),
    }


def process_evidence(state):
    try:
        process = psutil.Process(state["pid"])
        same = abs(process.create_time() - state["process_create_time"]) < 0.1
        return {"alive_and_identity_matches": same, "pid": state["pid"]}
    except (KeyError, psutil.Error):
        return {"alive_and_identity_matches": False, "pid": state.get("pid")}


def make_report(root):
    config = read_json(root / "effective_config.json", {})
    state = read_json(root / "state.json", {})
    records = read_records(root)
    physical = [row for row in records if row.get("api_request_started")]
    finished = [row for row in physical if row["status"] in ("completed", "failed", "refused")]
    usage = usage_summary(root)
    # In-flight requests have not returned usage yet; report them separately.
    usage["requests_missing_usage"] = sum(row.get("usage") is None for row in finished)
    usage["missing_usage_call_ids"] = [
        row["id"] for row in finished if row.get("usage") is None
    ]
    usage["pending_usage_requests"] = sum(
        row["status"] == "in_flight" for row in physical
    )
    usage["requested_models"] = sorted({row["requested_model"] for row in physical})
    usage["returned_models"] = sorted({
        row["returned_model"] for row in physical if row.get("returned_model")
    })
    by_stage = {}
    for stage in sorted({row["stage"] for row in physical}):
        subset = [row for row in finished if row["stage"] == stage]
        by_stage[stage] = {
            "response_seconds": distribution([row.get("response_seconds") for row in subset]),
            "queue_seconds": distribution([row.get("queue_seconds") for row in subset]),
            "total_elapsed_seconds": distribution([row.get("elapsed_seconds") for row in subset]),
            "http_statuses": dict(Counter(str(row.get("http_status")) for row in subset)),
        }
    stages = read_jsonl(root / "stages.jsonl")
    stage_wall = defaultdict(float)
    for row in stages:
        stage_wall[row["stage"]] += row["elapsed_seconds"]
    embeddings = read_jsonl(root / "embedding_calls.jsonl")
    embedding_by_stage = defaultdict(Counter)
    for row in embeddings:
        counters = embedding_by_stage[row["stage"]]
        counters["calls"] += 1
        counters["items"] += row["items"]
        counters["elapsed_seconds_sum"] += row["elapsed_seconds"]
        counters[row["status"]] += 1
        counters["retry_attempts"] += row.get("attempt", 1) > 1
    embedding_latency_by_stage = {}
    for stage in sorted(embedding_by_stage):
        rows = [row for row in embeddings if row["stage"] == stage]
        embedding_latency_by_stage[stage] = {
            "response_seconds": distribution([row.get("elapsed_seconds") for row in rows]),
            "queue_seconds": distribution([row.get("queue_seconds") for row in rows]),
            "observed_completed_interval_peak_concurrency": completed_interval_peak(rows),
            "error_types": dict(Counter(
                row["error_type"] for row in rows if row.get("error_type")
            )),
        }
    qa = read_json(root / "qa/predictions_with_supporting_facts.json", [])
    metrics = read_json(root / "metrics.json")
    metrics_by_hops = defaultdict(list)
    for row in (metrics or {}).get("per_question", []):
        match = row.get("question_type", "unknown")
        metrics_by_hops[match].append(row)
    construction = read_json(root / "construction_summary.json")
    summary = read_json(root / "final_summary.json")
    resources = read_jsonl(root / "resources.jsonl")
    peak = {
        field: max((row[field] for row in resources), default=None)
        for field in ("controller_rss_bytes", "neo4j_rss_bytes")
    }
    started = datetime.fromisoformat(state["started_at"]).timestamp() if state.get("started_at") else None
    finished_at = state.get("finished_at")
    end = datetime.fromisoformat(finished_at).timestamp() if finished_at else time.time()
    report = {
        "generated_at": utc_now(), "full_reproduction_completed": bool(
            state.get("full_reproduction_completed") and metrics and summary
        ),
        "state": state, "process": process_evidence(state),
        "effective_config": config,
        "active_code_parameters": read_json(
            Path(__file__).parent / "provenance/active_code_parameters.json"
        ),
        "sample_manifest": read_json(DESTINATION / "manifest.json"),
        "hardware": read_json(Path(__file__).parent / "audit/hardware.json"),
        "nvidia_query": (
            (Path(__file__).parent / "audit/nvidia-gpu.txt").read_text(encoding="utf-8").strip()
            if (Path(__file__).parent / "audit/nvidia-gpu.txt").exists() else None
        ),
        "current_invocation_wall_seconds": end - started if started else None,
        "stage_events": stages, "recorded_stage_wall_seconds_all_invocations": dict(stage_wall),
        "extraction": read_json(root / "extraction_progress.json"),
        "corpus_scope_audit": read_json(root / "corpus_scope_audit.json"),
        "tuple_parse_audit": read_json(root / "tuple_parse_audit.json"),
        "cost_architecture_audit": read_json(root / "cost_architecture_audit.json"),
        "execution_log_audit": read_json(root / "execution_log_audit.json"),
        "completion_verification": read_json(root / "completion_verification.json"),
        "provider_route_epoch": read_json(root / "provider_route_epoch.json"),
        "provider_route_history": read_jsonl(root / "provider_route_history.jsonl"),
        "provider_refusals": {
            "physical_content_filter_attempts": usage["content_filter_physical_attempts"],
            "distinct_extraction_chunk_hashes": usage["content_filter_extraction_chunk_hashes"],
            "ledger": read_jsonl(root / "provider_refusals.jsonl"),
            "active_extraction_chunk_hashes": usage["active_content_filter_extraction_chunk_hashes"],
            "policy": "Exact refusal text preserved; original parser and all candidate/question IDs retained",
        },
        "model_usage": usage, "request_latency_by_stage": by_stage,
        "embedding_calls": len(embeddings),
        "embedding_by_stage": {key: dict(value) for key, value in embedding_by_stage.items()},
        "embedding_observed_completed_interval_peak_concurrency": completed_interval_peak(embeddings),
        "embedding_request_latency_by_stage": embedding_latency_by_stage,
        "embedding_retry_attempts": sum(row.get("attempt", 1) > 1 for row in embeddings),
        "embedding_tokens": None,
        "embedding_tokens_note": "Original Ollama wrapper does not expose token usage",
        "qa": {
            "expected": 1000, "recorded": len(qa),
            "completed": sum(row["status"] == "completed" for row in qa),
            "failed": sum(row["status"] != "completed" for row in qa),
            "empty_answers": sum(
                isinstance(row["answer"], str) and not row["answer"].strip() for row in qa
            ),
            "invalid_answer_types": sum(not isinstance(row["answer"], str) for row in qa),
            "question_attempts": dict(Counter(str(row["attempts"]) for row in qa)),
            "latency_seconds": distribution([row["elapsed_seconds"] for row in qa]),
            "throughput_questions_per_second": (
                len(qa) / stage_wall["qa_inference"] if stage_wall.get("qa_inference") else None
            ),
        },
        "metrics": metrics,
        "metrics_by_question_type": {
            key: {
                "questions": len(rows),
                "answer_em": sum(row["answer_em"] for row in rows) / len(rows),
                "answer_f1": sum(row["answer_f1"] for row in rows) / len(rows),
            }
            for key, rows in metrics_by_hops.items()
        },
        "construction": construction, "final_summary": summary,
        "resource_peaks": peak,
        "storage": storage_evidence(),
        "resource_note": "Controller and owned Neo4j RSS sampled every2s; existing Ollama/GPU not included",
        "source_limitations": [
            "Official GitHub graph code plus user-approved supplied local QA/evaluators",
            "Original two-hop/hyperedge construction remains TODO",
            "Offline path-question repository absent, records0 and bytes0",
            "Supplied global-top10-then-question-filter retrieval preserved",
            "No verified relay tariff; no invented monetary spend",
            "A sampling seed does not make external model outputs bitwise deterministic",
        ],
    }
    return report


def markdown(report):
    usage = report["model_usage"]
    config = report["effective_config"]
    state = report["state"]
    tokens = usage["reported_token_totals"]
    status = "全流程完成" if report["full_reproduction_completed"] else "运行中／尚未完成"
    lines = [
        "# hotpotqa seed42 1000题 SHGraphRAG 独立中转复现记录", "",
        f"生成时间：{report['generated_at']}。状态：**{status}**。",
        f"当前阶段：`{state.get('stage')}`；控制进程状态：`{state.get('status')}`。",
        "", "## 数据、来源与配置", "",
        "- Frozen development sample: 1000 unique IDs, random.Random(42), all associated native candidates.",
        f"- Frozen candidates: {report['sample_manifest']['candidate_occurrences']:,} occurrences; {report['sample_manifest']['unique_candidate_passages']:,} exact unique passages.",
        "- 上游处理器默认选前500题，逐题遍历全部候选 context；本轮改为seed42抽样1000题。",
        "- 建图包含候选干扰段落，没有只选 gold 支持段落；没有构建完整开发集或完整维基语料。",
        f"- 官方建图提交：`{config.get('upstream_commit')}`。",
        f"- 实际运行提交：`{config.get('working_commit')}`。",
        f"- 实际运行源码内容清单 SHA256：`{config.get('source_bundle_sha256')}`（ZIP 文件本身的哈希另记）。",
        f"- 样本 ID 文件 SHA256：`{config.get('sample_ids_sha256')}`。",
        "- QA/评估来自用户提供的本地仓库，必要修复与逐文件来源均已记录。",
        f"- 独立中转：`{config.get('base_url')}`；模型：`{config.get('model')}`；凭据标识：`{config.get('credential_id')}`；文本请求并发上限64。",
        "- 原提示词、桥实体剪枝、逐题实体作用域、BGE-M3 和原配置常量保留。",
        "- 原始配置未指定温度及输出上限，实际请求省略这些参数。",
        "", "## 实际模型调用与 token", "",
        f"- 已发物理请求：{usage['physical_requests']}；完成：{usage['completed_requests']}；失败：{usage['failed_requests']}。",
        f"- 尚在途、尚无 usage 的请求：{usage['pending_usage_requests']}；已结束但无 usage：{usage['requests_missing_usage']}。",
        "- 已结束但无 usage 的调用不能按0 token处理；下方已报告总数可能低于实际消耗。",
        f"- 已报告输入 token：{tokens.get('input_tokens', 0):,}。",
        f"- 已报告输出 token：{tokens.get('output_tokens', 0):,}。",
        f"- 已报告总 token：{tokens.get('total_tokens', 0):,}。",
        f"- 输入中的缓存 token：{tokens.get('cached_input_tokens', 0):,}；输出中的推理 token：{tokens.get('reasoning_output_tokens', 0):,}。",
        f"- 各 usage 字段实际报告的请求数：`{usage['reported_usage_field_call_counts']}`。字段报告数为0时，累计0不能解释为实际用量为0。",
        "- 缓存 token 是输入 token 的子集，推理 token 是输出 token 的子集，不能重复相加。",
        f"- 已结束请求区间观测到的并发峰值：{usage['observed_completed_interval_peak_concurrency']}。",
        f"- 实际返回模型集合：`{usage['returned_models']}`。",
        "- 未提供可验证的本轮计价标准，金额记为未知；不把 token 数冒充费用。",
        f"- \u4f9b\u5e94\u5546\u5185\u5bb9\u62d2\u7edd\uff1a{usage['content_filter_physical_attempts']} \u6b21\u7269\u7406\u8bf7\u6c42\uff1b\u6d89\u53ca {len(usage['content_filter_extraction_chunk_hashes'])} \u4e2a\u552f\u4e00\u6bb5\u843d\u3002\u62d2\u7edd\u539f\u6587\u548c\u9898\u76ee\u4fdd\u7559\uff0c\u4e0d\u5c06\u62d2\u7edd\u8ba1\u4e3a\u6b63\u5e38\u5b8c\u6210\u7684\u6a21\u578b\u7b54\u6848\u3002",
        "", "| 阶段 | 已发请求 | 已完成 | 失败 | 已报告输入 token | 已报告输出 token |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, values in usage["by_stage"].items():
        lines.append(
            f"| {stage} | {values.get('physical_requests', 0)} | {values.get('completed', 0)} | "
            f"{values.get('failed', 0)} | {values.get('input_tokens', 0):,} | {values.get('output_tokens', 0):,} |"
        )
    lines.extend(["", "## 时间与效率", "",
                  "以下为已记录结束事件的墙钟时间，含失败尝试；当前未结束阶段不会伪造最终耗时。",
                  "", "| 阶段 | 累计墙钟秒 |", "| --- | ---: |"])
    for stage, seconds in report["recorded_stage_wall_seconds_all_invocations"].items():
        lines.append(f"| {stage} | {seconds:.3f} |")
    if report["current_invocation_wall_seconds"] is not None:
        lines.append(
            f"\n本次控制进程墙钟时间：{report['current_invocation_wall_seconds']:.3f}s。"
            "运行中时这是截至报告生成时的已用时间。"
        )
    lines.extend([
        "", "逐个已结束物理模型请求的响应时间（含失败尝试；不等于逐题端到端时间）：", "",
        "| 阶段 | 请求数 | 均值秒 | P50秒 | P95秒 | 最大秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for stage, values in report["request_latency_by_stage"].items():
        timing = values["response_seconds"]
        if timing:
            lines.append(
                f"| {stage} | {timing['count']} | {timing['mean']:.3f} | "
                f"{timing['p50']:.3f} | {timing['p95']:.3f} | {timing['max']:.3f} |"
            )
    qa = report["qa"]
    lines.extend([
        "", f"QA 已保存 {qa['recorded']}/1000 题，完成 {qa['completed']}，失败 {qa['failed']}，空答案 {qa['empty_answers']}。",
        f"本地嵌入调用 {report['embedding_calls']} 次；原嵌入封装不暴露 usage，嵌入 token 记为未知。",
        f"本地嵌入并发上限：{config.get('engineering_scheduling', {}).get('local_embedding_max_in_flight', '尚未启动')}；"
        f"已结束请求区间的实际并发峰值：{report['embedding_observed_completed_interval_peak_concurrency']}；"
        f"重试尝试数：{report['embedding_retry_attempts']}。逐阶段排队、响应时间和错误类型另存 JSON 报告。",
        f"QA 非字符串答案 {qa['invalid_answer_types']}；逐题尝试次数分布：`{qa['question_attempts']}`。",
        "并发调用的逐请求耗时相加不等于墙钟时间。资源峰值是每2秒抽样的控制进程与独立 Neo4j RSS，不含共享 Ollama/GPU。",
    ])
    if qa["latency_seconds"]:
        latency = qa["latency_seconds"]
        lines.append(
            f"逐题处理耗时（从工作线程开始，含内部重试与请求等待，不含批作业线程排队）：均值 {latency['mean']:.3f}s，"
            f"P50 {latency['p50']:.3f}s，P95 {latency['p95']:.3f}s。"
        )
    if qa["throughput_questions_per_second"] is not None:
        lines.append(
            f"QA批作业吞吐量：{qa['throughput_questions_per_second']:.6f}题/秒。"
        )
    if report["metrics"]:
        metrics = report["metrics"]
        lines.extend([
            "", "## 测评", "",
            f"- 固定分母：{metrics['denominator']}。",
            f"- Answer EM：{metrics['answer_em'] * 100:.4f}%。",
            f"- Answer F1：{metrics['answer_f1'] * 100:.4f}%。",
            "- 使用原归一化和别名取最大值；缺失预测计零、额外 ID 被拒绝，逐题分数在 metrics.json。",
        ])
        lines.extend([
            "", "| Native question type | Questions | Answer EM % | Answer F1 % |",
            "| --- | ---: | ---: | ---: |",
        ])
        for hops, values in sorted(report["metrics_by_question_type"].items()):
            lines.append(
                f"| {hops} | {values['questions']} | "
                f"{values['answer_em'] * 100:.4f} | {values['answer_f1'] * 100:.4f} |"
            )
    if report["construction"]:
        graph = report["construction"]["graph"]
        lines.extend([
            "", "## 图、向量与存储", "",
            f"- 文档 {graph['documents']}；文段 {graph['chunks']}；实体 {graph['entities']}。",
            f"- 全部关系 {graph['all_relationships']}；实体间关系 {graph['entity_relationships']}。",
            f"- Neo4j data 目录字节：{graph['database_disk_bytes']:,}（含原生存储、索引及事务文件）。",
            "- 实体与文段向量均检查为1024维；entity/chunk向量索引检查为 ONLINE。",
        ])
    lines.extend([
        "", "存储按逻辑文件长度统计，包含预分配；不能解释为图本身的净占用或磁盘实际分配块数。",
        "", "| 文件类别 | 字节 |", "| --- | ---: |",
    ])
    for category, value in report["storage"].items():
        if isinstance(value, int):
            lines.append(f"| {category} | {value:,} |")
    scope = report["corpus_scope_audit"]
    if scope:
        sample = scope["executed_seed42_sample_contexts"]
        lines.extend([
            "", "## 候选语料范围核验", "",
            f"- 本轮候选记录：支持段落 {sample['candidate_occurrences_with_native_supporting_title']:,} 次，"
            f"干扰段落 {sample['candidate_occurrences_without_native_supporting_title']:,} 次；两者均进入建图输入。",
            "- 上述支持标签只用于离线核验，没有用于筛选或缩减建图输入。",
            "- 固定ID、原生候选内容与顺序、候选导出一致性均已检查。",
            "- 换样本复用图时必须检查候选覆盖与逐题实体作用域；相同段落的抽取缓存可以复用。",
        ])
    tuple_audit = report["tuple_parse_audit"]
    if tuple_audit:
        lines.extend([
            "", "## 原 tuple 解析核验", "",
            f"- 本次核验覆盖 {tuple_audit['cached_unique_passages']:,} 个缓存段落，"
            f"生成时间 `{tuple_audit['generated_at']}`。",
            f"- 原正则未解析出实体的段落 {len(tuple_audit['zero_entity_chunk_hashes']):,} 个；"
            f"其中含实体标记 {tuple_audit['zero_entities_with_entity_marker']:,} 个。",
            f"- 无法转换为数值的关系权重 {tuple_audit['unparseable_relation_weights']:,} 个。",
            "- 保留原解析行为，核验未触发格式修复或额外模型调用。零实体不自动等同于格式错误。",
        ])
    log_audit = report["execution_log_audit"]
    if log_audit:
        counts = log_audit["explicit_diagnostic_line_counts"]
        lines.extend([
            "", "## 显式运行诊断", "",
            f"- 日志核验时间：`{log_audit['generated_at']}`。",
            f"- 数据库查询失败日志 {counts['neo4j_query_failed']} 条；"
            f"文段检索失败日志 {counts['chunk_retrieval_failed']} 条；"
            f"逐题尝试失败日志 {counts['question_attempt_failed']} 条。",
            f"- 分解解析回退日志 {counts['decomposition_parse_fallback']} 条；"
            f"图 tuple 解析错误日志 {counts['tuple_parse_error']} 条。",
            f"- QA答案状态分布：`{log_audit['qa_answer_type_counts']}`。",
            "- 此处只统计指定诊断日志的出现次数，不能证明检索非空或答案语义正确。",
        ])
    verification = report["completion_verification"]
    if verification:
        lines.extend([
            "", f"离线完成核验：`passed={verification['passed']}`，"
            f"共 {len(verification['checks'])} 项检查。",
            "源码 ZIP 文件 SHA256："
            f"`{verification['execution_source_archive_file_sha256']}`。",
        ])
    lines.extend([
        "", "## 可比性和限制", "",
        "- 官方两跳/超边离线构建仍为 TODO；所选源码不存在离线路径问题库，记录数和字节数为0。",
        "- 本轮是所选公开代码与提供 QA 的可追溯复现，不能据此声称补齐了论文缺失机制。",
        "- 保留原 QA 的全局 top10 后按题过滤；未为提高分数改变检索算法。",
        "- Python3.12 的必要依赖兼容调整和实际环境冻结均保留。",
        "- seed42 固定样本；外部模型本身不保证逐位确定。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 50:
        parser.error("wait-seconds must be between0 and50")
    time.sleep(args.wait_seconds)
    report = make_report(args.run_dir)
    write_json(args.run_dir / "report.json", report)
    (args.run_dir / "REPORT_ZH.md").write_text(markdown(report), encoding="utf-8")
    if args.compact:
        print(json.dumps({
            "status": report["state"]["status"], "stage": report["state"]["stage"],
            "process_alive": report["process"]["alive_and_identity_matches"],
            "extracted": (report["extraction"] or {}).get("completed_unique_chunks"),
            "unique_chunks": (report["extraction"] or {}).get("unique_chunks"),
            "requests_completed": report["model_usage"]["completed_requests"],
            "requests_failed": report["model_usage"]["failed_requests"],
            "requests_refused": report["model_usage"]["refused_requests"],
            "content_filter_physical_attempts": report["model_usage"]["content_filter_physical_attempts"],
            "refused_unique_extraction_chunks": len(report["model_usage"]["active_content_filter_extraction_chunk_hashes"]),
            "historically_refused_unique_extraction_chunks": len(report["model_usage"]["content_filter_extraction_chunk_hashes"]),
            "in_flight": report["model_usage"]["pending_usage_requests"],
            "tokens": report["model_usage"]["reported_token_totals"],
            "qa_completed": report["qa"]["completed"], "qa_failed": report["qa"]["failed"],
            "full_reproduction_completed": report["full_reproduction_completed"],
        }, ensure_ascii=False))
        return
    print(json.dumps({
        "state": report["state"]["status"], "stage": report["state"]["stage"],
        "process": report["process"], "extraction": report["extraction"],
        "physical_requests": report["model_usage"]["physical_requests"],
        "completed_requests": report["model_usage"]["completed_requests"],
        "failed_requests": report["model_usage"]["failed_requests"],
        "in_flight": report["model_usage"]["pending_usage_requests"],
        "tokens": report["model_usage"]["reported_token_totals"],
        "qa": report["qa"], "metrics": (
            {key: report["metrics"][key] for key in ("denominator", "answer_em", "answer_f1")}
            if report["metrics"] else None
        ),
        "report_path": str(args.run_dir / "REPORT_ZH.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
