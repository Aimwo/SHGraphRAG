"""
    2Wiki-Multihop QA evaluation script
    Adapted from HotpotQA evaluation at https://github.com/hotpotqa/hotpot
"""
import sys
import json
import re
import string
from collections import Counter
import pickle


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def normalize_sp(sps):
    new_sps = []
    for sp in sps:
        sp = list(sp)
        sp[0] = sp[0].lower()
        new_sps.append(sp)
    return new_sps


def update_sp(metrics, prediction, gold):
    cur_sp_pred = normalize_sp(set(map(tuple, prediction)))
    gold_sp_pred = normalize_sp(set(map(tuple, gold)))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def normalize_evi(evidences):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    for idx_1 in range(len(evidences)):
        for idx_2 in range(len(evidences[idx_1])):
            evidences[idx_1][idx_2] = white_space_fix((remove_punc(lower(evidences[idx_1][idx_2]))))
    return evidences


def update_evi(metrics, prediction, gold):
    prediction_normalize = normalize_evi(prediction)
    gold_normalize = normalize_evi(gold)
    #
    cur_evi_pred = set(map(tuple, prediction_normalize))
    gold_evi_pred = set(map(tuple, gold_normalize))
    #
    #
    tp, fp, fn = 0, 0, 0
    for e in cur_evi_pred:
        if e in gold_evi_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_evi_pred:
        if e not in cur_evi_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['evi_em'] += em
    metrics['evi_f1'] += f1
    metrics['evi_prec'] += prec
    metrics['evi_recall'] += recall
    return em, prec, recall


def eval(prediction_file, gold_file, qid):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)[:len(prediction["answer"].keys())]

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'evi_em': 0, 'evi_f1': 0, 'evi_prec': 0, 'evi_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for dp in gold:
        cur_id = dp['_id']
        # if cur_id not in qid:
        #     continue
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])
        # #
        # if cur_id not in prediction['evidence']:
        #     print('missing evidence {}'.format(cur_id))
        #     can_eval_joint = False
        # else:
        #     evi_em, evi_prec, evi_recall = update_evi(
        #         metrics, prediction['evidence'][cur_id], dp['evidences'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            #
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] = round(metrics[k] / N * 100, 2)

    print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    prediction_file = "./inputs/predictions.json"
    gold_file = "./inputs/2wiki_dev.json"
    qid = ['1869092c0bb011ebab90acde48001122', 'c531e7560baf11ebab90acde48001122', '2207e9060bda11eba7f7acde48001122', '8981e1ce0bb011ebab90acde48001122', 'a9cc5f240bdb11eba7f7acde48001122', '019bed300bde11eba7f7acde48001122', '0cd3bdea0bde11eba7f7acde48001122', 'c19b72520bdb11eba7f7acde48001122', '8813f87c0bdd11eba7f7acde48001122', 'b3110a5e0bda11eba7f7acde48001122', '3705552a0bb011ebab90acde48001122', 'db19bf4c086e11ebbd62ac1f6bf848b6', '96fb4aee0baf11ebab90acde48001122', '27e835ee0bb011ebab90acde48001122', '931a85420bdd11eba7f7acde48001122', '0bbded440bda11eba7f7acde48001122', '3cbd107e0bde11eba7f7acde48001122', '6638110e0bdb11eba7f7acde48001122', '258867240bdd11eba7f7acde48001122', 'a8cbc4c40bdc11eba7f7acde48001122', '3717c0600bde11eba7f7acde48001122', '6638a0f0084e11ebbd56ac1f6bf848b6', '424d56700bde11eba7f7acde48001122', 'bc633e2c0baf11ebab90acde48001122', '2b4bd5020baf11ebab90acde48001122', '3647118c0bdd11eba7f7acde48001122', 'ea1fc06c0bda11eba7f7acde48001122', '00c75ffc0bde11eba7f7acde48001122', '71d8e20e0bdb11eba7f7acde48001122', '48c8f7f80bdd11eba7f7acde48001122', 'b5e9ef2c0bdd11eba7f7acde48001122', 'a5bed06e0bdc11eba7f7acde48001122', 'e8d8f7200bdd11eba7f7acde48001122', 'b5b63b160bdb11eba7f7acde48001122', '6021e2d80bde11eba7f7acde48001122', '97ce0ec00baf11ebab90acde48001122', 'b4e780a5088f11ebbd72ac1f6bf848b6', 'a08505d20bdc11eba7f7acde48001122', '5839dbac0bde11eba7f7acde48001122', '81ca57e60bdc11eba7f7acde48001122', '5b637e060bb011ebab90acde48001122', '1a2188840bb011ebab90acde48001122', 'f6e2b9280bdd11eba7f7acde48001122', 'e6688a7e0baf11ebab90acde48001122', '49ec1ade0bdd11eba7f7acde48001122', 'e4e02a240bda11eba7f7acde48001122', '47806bc308c411ebbd8eac1f6bf848b6', '437f3b9a0bdd11eba7f7acde48001122', '652d7dac0bdc11eba7f7acde48001122', 'b1b78c480bdd11eba7f7acde48001122', '027970120bdc11eba7f7acde48001122', '94873b220bd911eba7f7acde48001122', '6e5e9de109af11ebbdb0ac1f6bf848b6', '79675c6e0bd911eba7f7acde48001122', 'c40d35580bda11eba7f7acde48001122', '421b26d80bb011ebab90acde48001122', '5e60d12e0bda11eba7f7acde48001122', 'd133120e0bdd11eba7f7acde48001122', '87154f1e0bda11eba7f7acde48001122', 'a6d861000bdb11eba7f7acde48001122', '5d8037920bb011ebab90acde48001122', '231d667c0bda11eba7f7acde48001122', '8ba63ce8098f11ebbdb0ac1f6bf848b6', '9f62be2008ac11ebbd83ac1f6bf848b6', 'befbd60c0bdd11eba7f7acde48001122', 'b035afca0bd911eba7f7acde48001122', '1734c9b60baf11ebab90acde48001122', 'be0a16240baf11ebab90acde48001122', '6e9a0d6a08eb11ebbda7ac1f6bf848b6', 'ea9832140bd911eba7f7acde48001122', 'e1f0c38c0bda11eba7f7acde48001122', '0ac40d780bdb11eba7f7acde48001122', 'b3d918160bdd11eba7f7acde48001122', 'b8ef144a0bdd11eba7f7acde48001122', '8c82acb20bda11eba7f7acde48001122', '6b5393240bdc11eba7f7acde48001122', '3d3bc0720bde11eba7f7acde48001122', 'b411910a0bdd11eba7f7acde48001122', '09945977084d11ebbd56ac1f6bf848b6', '3068a40c0baf11ebab90acde48001122', '215f53a00bde11eba7f7acde48001122', '6900298c0bde11eba7f7acde48001122', '9c24e2900bdd11eba7f7acde48001122', '08de5d000baf11ebab90acde48001122', 'b8d121ba099d11ebbdb0ac1f6bf848b6', '6e3e70200bde11eba7f7acde48001122', '967b39ca0bdd11eba7f7acde48001122', '3b9299900bdd11eba7f7acde48001122', '859991680bda11eba7f7acde48001122', 'a0ae26f60baf11ebab90acde48001122', 'eb7ece320bdd11eba7f7acde48001122', '33b51db508af11ebbd83ac1f6bf848b6', '747175620bdb11eba7f7acde48001122', '5affe63d08f311ebbdaaac1f6bf848b6', '8e07f1f00bda11eba7f7acde48001122', 'd6be0cee0bdb11eba7f7acde48001122', 'a5b095c00bdd11eba7f7acde48001122', '3a3c2efe0bdc11eba7f7acde48001122', 'c8c9313a0baf11ebab90acde48001122', '2d907ba60bdc11eba7f7acde48001122', 'f7b1002b099a11ebbdb0ac1f6bf848b6', '6c7f03240bdb11eba7f7acde48001122', 'dea2ea800bd911eba7f7acde48001122', 'dbcd3f5e0bd911eba7f7acde48001122', '0c0830a00baf11ebab90acde48001122', '9e506db00bdc11eba7f7acde48001122', 'bac99ab40bd911eba7f7acde48001122', 'ab94845c0baf11ebab90acde48001122', '6b3de52e0bdc11eba7f7acde48001122', '0c585dbc0bde11eba7f7acde48001122', 'b65877e20bda11eba7f7acde48001122', 'bc69662a0bdd11eba7f7acde48001122', '70ec660c0bdd11eba7f7acde48001122', '850c2ecc0bda11eba7f7acde48001122', '7e73e6560bdd11eba7f7acde48001122', 'c81d185a0baf11ebab90acde48001122', '1c0dd3b00bdc11eba7f7acde48001122', '5db7e2d80bdc11eba7f7acde48001122', '5e418da00bda11eba7f7acde48001122', 'a022d8580bdc11eba7f7acde48001122', '475b89f80bb011ebab90acde48001122', '5d71855608e711ebbda6ac1f6bf848b6', '86ae8b920baf11ebab90acde48001122', '85dd56440baf11ebab90acde48001122']

    eval(prediction_file, gold_file, qid)