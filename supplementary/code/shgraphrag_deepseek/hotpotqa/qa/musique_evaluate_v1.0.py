import argparse
import json
from collections import Counter

"""
An abstract class representing a metric which can be accumulated.
"""
from typing import Any, Dict

"""
Answer metric -- mostly taken directly from squad_tools of allennlp.
"""
import re
import string
import collections
from typing import Tuple, List


class Metric:
    """
    An abstract class representing a metric which can be accumulated.
    """

    def __call__(self, predictions: Any, gold_labels: Any):
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AnswerMetric(Metric):
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(
            self,
            predicted_answer: str,
            ground_truth_answers: List[str],
    ):
        exact_scores = metric_max_over_ground_truths(
            compute_exact, predicted_answer, ground_truth_answers
        )
        f1_scores = metric_max_over_ground_truths(
            compute_f1, predicted_answer, ground_truth_answers
        )

        self._total_em += int(exact_scores)
        self._total_f1 += f1_scores
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def evaluate(filepath_with_predictions: str, filepath_with_ground_truths: str, qid=None) -> Dict:
    """Score every frozen ID; missing answers contribute zero, never shrink gold."""
    with open(filepath_with_predictions, encoding="utf-8") as f:
        prediction_instances = json.load(f)
    ground_truth_instances = read_jsonl(filepath_with_ground_truths)
    gold_by_id = {row["id"]: row for row in ground_truth_instances}
    if len(gold_by_id) != len(ground_truth_instances):
        raise ValueError("Duplicate gold IDs; this entry point evaluates MuSiQue-Ans only")
    selected_ids = list(qid) if qid is not None else list(gold_by_id)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Expected a nonempty list of unique selected question IDs")
    unknown_gold = set(selected_ids) - set(gold_by_id)
    if unknown_gold:
        raise ValueError(f"Selected IDs absent from gold: {sorted(unknown_gold)[:5]}")
    answers = prediction_instances["answer"]
    extra_ids = set(answers) - set(selected_ids)
    if extra_ids:
        raise ValueError(f"Predictions outside frozen sample: {sorted(extra_ids)[:5]}")
    answer_metric = AnswerMetric()
    per_question = []
    for question_id in selected_ids:
        gold = gold_by_id[question_id]
        if not gold["answerable"]:
            raise ValueError("MuSiQue-Full sufficiency scoring is outside this Ans-only run")
        predicted_answer = answers.get(question_id, "")
        if not isinstance(predicted_answer, str):
            raise ValueError(f"Non-string answer: {question_id}")
        gold_answers = [gold["answer"]] + gold.get("answer_aliases", [])
        answer_metric(predicted_answer, gold_answers)
        per_question.append({
            "id": question_id,
            "answer_em": metric_max_over_ground_truths(compute_exact, predicted_answer, gold_answers),
            "answer_f1": metric_max_over_ground_truths(compute_f1, predicted_answer, gold_answers),
            "prediction_present": question_id in answers,
        })
    exact, f1 = answer_metric.get_metric()
    missing = [question_id for question_id in selected_ids if question_id not in answers]
    return {
        "answer_em": exact, "answer_f1": f1,
        "answer_em_percent": 100 * exact, "answer_f1_percent": 100 * f1,
        "denominator": len(selected_ids),
        "prediction_count": len(answers), "prediction_coverage": len(answers) / len(selected_ids),
        "missing_prediction_ids": missing,
        "per_question": per_question,
        "metric_definition": "Supplied MuSiQue Ans normalization, token F1 and max over native aliases",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MuSiQue predictions.")
    parser.add_argument(
        "--filepath_with_predictions",
        default="./outputs/predictions.json",
        type=str,
        help="jsonl filepath to predicted instances.",
    )
    parser.add_argument(
        "--filepath_with_ground_truths",
        default="./inputs/musique_ans_v1.0_dev.jsonl",
        type=str,
        help="jsonl filepath to data instances.",
    )
    parser.add_argument(
        "--output_filepath",
        default=None,
        type=str,
        help="(optional) filepath to save output metrics."
    )
    parser.add_argument("--ids_path", type=str, required=True, help="Frozen sample IDs")
    args = parser.parse_args()

    with open(args.ids_path, encoding="utf-8") as stream:
        selected_ids = json.load(stream)
    metrics = evaluate(args.filepath_with_predictions, args.filepath_with_ground_truths, selected_ids)

    if args.output_filepath:
        print(f"Writing metrics output in: {args.output_filepath}")
        with open(args.output_filepath, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)
    else:
        print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
