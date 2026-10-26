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
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def evaluate(filepath_with_predictions: str, filepath_with_ground_truths: str, qid) -> Dict:
    with open(filepath_with_predictions) as f:
        prediction_instances = json.load(f)

    ground_truth_instances = read_jsonl(filepath_with_ground_truths)[:len(prediction_instances["answer"].keys())]

    do_sufficiency_eval = False
    answer_metric = AnswerMetric()

    for ground_truth_instance in ground_truth_instances:

        question_id = ground_truth_instance["id"]
        # if question_id not in qid:
        #     continue
        if ground_truth_instance["id"] not in prediction_instances["answer"]:
            print('missing answer {}'.format(question_id))

        predicted_answer = prediction_instances["answer"][question_id]
        ground_truth_answers = [
                                   ground_truth_instance["answer"]
                               ] + ground_truth_instance["answer_aliases"]

        ground_truth_sufficiency = ground_truth_instance["answerable"]

        if ground_truth_sufficiency:
            answer_metric(predicted_answer, ground_truth_answers)

        # If there's any instance with ground truth of unanswerable, we'll assume
        # it's full version of the dataset and not only the answerable version.
        if not ground_truth_sufficiency:
            do_sufficiency_eval = True

    metrics = {}
    metrics["answer_em"] = round(answer_metric.get_metric()[0], 3)
    metrics["answer_f1"] = round(answer_metric.get_metric()[1], 3)

    if do_sufficiency_eval:
        assert set(Counter([e['id'] for e in prediction_instances]).values()) == {2}, \
            "For sufficiency evaluation, there should two instances for each question."

    return metrics


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
    args = parser.parse_args()
    qid = ['2hop__574419_270105', '2hop__197470_271394', '2hop__697790_864352', '2hop__554167_451128', '2hop__252311_366220', '2hop__460946_294723', '2hop__499577_617190', '2hop__299142_347613', '2hop__166824_185403', '2hop__528263_865831', '2hop__549146_223121', '2hop__731228_609866', '2hop__744055_408817', '2hop__726717_610238', '2hop__685280_510757', '2hop__701895_752697', '2hop__195917_575657', '2hop__402530_774333', '2hop__481349_302087', '2hop__698577_851686', '2hop__235471_565925', '2hop__647996_551681', '2hop__207476_625987', '2hop__832795_350510', '2hop__335418_279684', '2hop__455158_247871', '2hop__700093_455653', '2hop__423972_540249', '2hop__304852_758385', '2hop__189094_612080', '2hop__661847_288421', '2hop__496961_554601', '2hop__175175_347233', '2hop__274110_413723', '2hop__628752_538661', '2hop__860241_766393', '2hop__231890_595414', '2hop__636597_480615', '2hop__29367_29376', '2hop__709357_136043', '2hop__373759_136043', '2hop__29368_29376', '2hop__471509_136043', '2hop__61318_362941', '2hop__20268_42004', '2hop__497191_129669', '2hop__115515_779396', '2hop__201704_135481', '2hop__485221_544917', '2hop__286093_361551', '2hop__124592_402641', '2hop__256553_617289', '2hop__124412_402641', '2hop__780652_93362', '2hop__499003_853511', '2hop__94507_654855', '2hop__62823_654855', '2hop__800166_156034', '2hop__771727_93362', '2hop__50910_177869', '2hop__82270_177869', '2hop__258940_135844', '2hop__747894_135844', '2hop__632234_122868', '2hop__640262_122868', '2hop__849534_122868', '2hop__439614_349102', '2hop__703493_162341', '2hop__158010_58067', '2hop__153573_109006', '2hop__153573_44085', '2hop__153573_45901', '2hop__7046_44085', '2hop__551235_310309', '2hop__488811_30078', '2hop__77775_162341', '2hop__567471_123283', '2hop__8347_8311', '2hop__488811_30110', '2hop__488811_30113', '2hop__208602_11402', '2hop__176712_8311', '2hop__71755_463572', '2hop__67956_463572', '2hop__28287_89399', '2hop__587048_621800', '2hop__85713_72187', '2hop__79875_72187', '2hop__734057_88628', '2hop__95878_88628', '2hop__726892_131968', '2hop__693157_131890', '2hop__25396_212301', '2hop__623501_297043', '2hop__30741_85990', '2hop__216535_370666', '2hop__846388_19320', '2hop__599396_19320', '2hop__685045_19320', '2hop__631024_18974', '2hop__10122_18974', '2hop__810411_159673', '2hop__96062_159673', '2hop__77832_159673', '2hop__527400_159673', '2hop__107690_124896', '2hop__507722_124896', '2hop__107548_124896', '2hop__13592_85544', '2hop__809785_606637', '2hop__656411_345851', '2hop__539184_119915', '2hop__233551_746883', '2hop__538735_3296', '2hop__538735_3299', '2hop__252521_80650', '2hop__593688_292889', '2hop__143915_68489', '2hop__598467_68489', '2hop__230022_68489', '2hop__142842_68489', '2hop__75207_46837', '2hop__44522_58556', '2hop__753715_51329', '2hop__37097_53663', '2hop__678516_179720', '2hop__591435_51329', '2hop__385392_756039', '2hop__170914_152093', '2hop__705035_152093', '2hop__39954_80303', '2hop__203437_152093', '2hop__142198_152093', '2hop__409166_152023', '2hop__452580_70131', '2hop__124807_135019', '2hop__605207_135019', '2hop__271045_68633', '2hop__645367_68633', '2hop__853227_68633', '2hop__126306_396277', '2hop__126188_396277', '2hop__613779_55984', '2hop__69765_43945', '2hop__275891_55984', '2hop__559076_55984', '2hop__37097_86208', '2hop__21991_43945', '2hop__170341_43945', '2hop__455392_40315', '2hop__312223_144857', '2hop__289517_139339', '2hop__507528_139339', '2hop__332453_139339', '2hop__706038_14251', '2hop__857585_14251', '2hop__821368_14251', '2hop__451574_120065', '2hop__123166_47134', '2hop__159146_120065', '2hop__326404_120065', '2hop__123545_47134', '2hop__347652_47134', '2hop__846844_47134', '2hop__238499_431914', '2hop__155781_155922', '2hop__852657_155922', '2hop__358270_108549', '2hop__67660_81007', '2hop__67660_158861', '2hop__91667_67223', '2hop__91667_81007', '2hop__245835_762148', '2hop__67660_67223', '2hop__151705_141308', '2hop__152052_141308', '2hop__501624_181960', '2hop__152700_141308', '2hop__151750_141308', '2hop__152027_141308', '2hop__157498_127908', '2hop__179149_396277', '2hop__448615_127908', '2hop__394720_120259', '2hop__418941_429437', '2hop__865828_377862', '2hop__586343_83289', '2hop__91626_96912', '2hop__690412_526810']

    metrics = evaluate(args.filepath_with_predictions, args.filepath_with_ground_truths, qid)

    if args.output_filepath:
        print(f"Writing metrics output in: {args.output_filepath}")
        with open(args.output_filepath, "w") as file:
            json.dump(metrics, file, indent=4)
    else:
        print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
