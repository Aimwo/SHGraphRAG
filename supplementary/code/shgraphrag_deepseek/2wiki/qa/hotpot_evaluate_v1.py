import sys
import json
import re
import string
from collections import Counter
import pickle

def normalize_answer(s):

    def remove_brackets(text):
        return re.sub(r'\(.*?\)', ' ', text)

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
    res = normalize_answer(prediction) == normalize_answer(ground_truth)
    if res:
        return res
    else:
        return res

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
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

def eval(prediction_file, gold_file, qid):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)[:len(prediction["answer"].keys())]

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        if cur_id not in qid:
            continue
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

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(prediction)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)

if __name__ == '__main__':
    prediction_file = "./inputs/predictions.json"
    gold_file = "./inputs/hotpot_dev_distractor_v1.json"
    qid = ['5ab6d09255429954757d337d', '5adddccd5542997dc7907069', '5ae0361155429925eb1afc2c', '5abd94525542992ac4f382d2', '5ab3e45655429976abd1bcd4', '5a828c8355429966c78a6a50', '5ae4a3265542995ad6573de5', '5ae0d4c9554299603e418468', '5a80721b554299485f5985ef', '5a8ef2a75542995a26add583', '5a8a3e745542996c9b8d5e70', '5ae22b8d554299234fd0440f', '5adf37a95542995ec70e8f97', '5a8e3ea95542995a26add48d', '5a8133725542995ce29dcbdb', '5a877e5d5542993e715abf7d', '5ab2d3df554299194fa9352c', '5a8e068b5542995085b37384', '5a7d54165542995f4f402256', '5ae73acb5542991e8301cc07', '5a7a0e1e5542990783324e1a', '5adc53f75542996e6852530a', '5a77152355429966f1a36c2e', '5ae1f4cb554299234fd0436d', '5a77cb335542997042120b3a', '5a80840f554299485f59863b', '5a88658955429938390d3f47', '5ae5aba0554299546bf82f17', '5abdf12255429976d4830a2f', '5aba5d2e55429901930fa799', '5abfb3425542990832d3a1c0', '5a80b3a9554299485f5986cc', '5ac32b565542995ef918c154', '5a85eed75542996432c5713b', '5a7dcc775542990b8f503ab5', '5a818387554299260e20a291', '5ae7eaf65542994a481bbe1d', '5a75f0ea5542994ccc91866c', '5a89f5815542993b751ca9c5', '5a7b12a355429927d897bf2a', '5abd01205542992ac4f3819b', '5ae0006755429925eb1afbd3', '5ac1c6805542994ab5c67e11', '5a80071f5542992bc0c4a684', '5a7c960655429935c91b5244', '5ab4e3065542990594ba9cb4', '5a7d61775542991319bc93b9', '5ab5ed505542997d4ad1f255', '5ab6a2be554299710c8d1f08', '5a87d8215542994775f60800', '5a713a5a5542994082a3e6a9', '5ae52c0f5542992663a4f12e', '5ab7f0015542992aa3b8c88b', '5a7ddc5a5542995f4f4022da', '5ae0a59a55429945ae9593e2', '5a88dcf9554299206df2b383', '5a8eea4a5542990e94052bb7', '5ac0d83a554299294b219038', '5ae0bae555429906c02dab08', '5a790d8f554299029c4b5eec', '5aba65a055429939ce03dcd2', '5add673e5542992ae4cec54d', '5ae74fc45542997b22f6a6b4', '5ae819ac55429952e35eaa16', '5ac3e0f7554299194317388b', '5ac02e31554299294b218f3f', '5a79ea435542996c55b2dc9e', '5abd2eab5542992ac4f38210', '5ab7e4dc5542995dae37e9df', '5ab41d855542996a3a969f4b', '5adf58f15542993a75d264d2', '5a791a97554299148911f9f2', '5a738fe855429908901be2fb', '5a87ab9b5542996e4f3088c2', '5ab8688b55429934fafe6daa', '5ade51b45542997c77adedf4', '5a8c65a1554299585d9e3680', '5ae4f2c75542993aec5ec0fb', '5ae6828f5542996d980e7bb9', '5ab31864554299233954ff06', '5abe239355429976d4830a7d', '5ac1795f5542994ab5c67d5f', '5ae4d3c95542990ba0bbb165', '5ab7f97a5542991d322237ef', '5a86769c5542994775f60776', '5a864ff7554299211dda2af2', '5a7caae75542996dd594b971', '5ab7da6f55429928e1fe3928', '5abc3a65554299700f9d782d', '5ae3ff575542995dadf242aa', '5adccdd55542992c1e3a2408', '5ab9debb55429955dce3ed97', '5a78db8055429970f5fffdb2', '5ae048a255429924de1b708e', '5adfbca255429942ec259b2c', '5ae4c39f5542995dadf243fb', '5a8cb288554299585d9e3726', '5a76e88555429972597f13f4', '5a7cb9b95542990527d55515', '5a79c6a55542994bb9457091', '5a85ab905542994c784ddb35', '5a886886554299206df2b258', '5a7f5c9f55429969796c1a0f', '5a8b770f5542995d1e6f13a7', '5ade15f35542997545bbbe47', '5a72a0be5542992359bc3143', '5a77cb5e55429967ab1052a7', '5ae5072e55429960a22e0246', '5ab5f854554299637185c64e', '5a79f6ca554299029c4b5f90', '5ade24e35542992fa25da6e6', '5ab5b6c1554299637185c5e5', '5add67555542992ae4cec54e', '5a77a65b5542992a6e59df57', '5a8da81e554299068b959d7e', '5ac497025542997ea680ca8d', '5ac01bca5542992a796dec8d', '5a7b7bed5542997c3ec971bc', '5ab67772554299110f219a12', '5a7b65ab5542997c3ec97144', '5ac3d9c9554299204fd21ea3', '5ae74890554299572ea547af', '5a7b76735542997c3ec9719f', '5a8799c55542996e4f308877', '5a8f69b55542997ba9cb324f', '5ac038db554299294b218f57', '5a85dfcc5542994775f60673', '5a7c9f325542990527d554e6', '5ab5057b5542991779162d56', '5a85c3a15542992a431d1b97', '5a81ebee554299676cceb16d', '5a895d305542993b751ca931', '5a8d02cc554299441c6b9fb6', '5adc094a55429947ff1738c0', '5a88f9555542995153361214', '5ab531cc5542990594ba9d23', '5a7a014a5542996c55b2dccf', '5ac45b58554299076e296dad', '5a8038b55542996402f6a485', '5ab83d595542990e739ec886', '5ab45a0b5542991779162c7e', '5ade42b55542992fa25da717', '5ae4c2b95542995dadf243ef', '5a8eb4d65542990e94052b82', '5adddd075542992200553b71', '5ab78edc5542995dae37e95c', '5ab3306a55429969a97a80fd', '5adf8fe55542993344016d0f', '5a7252db5542990c210a4104', '5a7634f155429976ec32bd6b', '5a8b78775542997f31a41d3d', '5ab6aaff55429953192ad35b', '5ae1f596554299234fd04372', '5adc8977554299438c868de2', '5ae4890a5542995ad6573d60', '5add311b5542994ed6169cc5', '5ae60426554299546bf83019', '5adda7585542997dc790700c', '5a81ff62554299676cceb1c4', '5adc1ae5554299438c868d5c', '5a77beae5542992a6e59dfae', '5a7757215542993569682d58', '5ade908b55429975fa854f13', '5a84873e5542997175ce1eec', '5ac537975542996feb3fea3c']

    eval(prediction_file, gold_file, qid)


# gemini 2  加入examples的 rag + 模糊匹配{'em': 0.84, 'f1': 0.7794363560416191, 'prec': 0.7739039555216027, 'recall': 0.8398333333333334, 'sp_em': 0.22, 'sp_f1': 0.6654047619047614, 'sp_prec': 0.8916666666666666, 'sp_recall': 0.5635000000000001, 'joint_em': 0.2, 'joint_f1': 0.5436021369934411, 'joint_prec': 0.719142050759698, 'joint_recall': 0.48772222222222217}

# gemini 2  加入examples的 混合检索    {'em': 0.65, 'f1': 0.7964767600820231, 'prec': 0.7895980392156862, 'recall': 0.8448333333333334, 'sp_em': 0.21, 'sp_f1': 0.6232857142857139, 'sp_prec': 0.8383333333333334, 'sp_recall': 0.5331666666666668, 'joint_em': 0.14, 'joint_f1': 0.5232473496239479, 'joint_prec': 0.6895980392156864, 'joint_recall': 0.4694722222222222}

# gemini 2   原始  {'em': 0.63, 'f1': 0.7738490544543174, 'prec': 0.7815071301247771, 'recall': 0.8023333333333333, 'sp_em': 0.32, 'sp_f1': 0.6778095238095235, 'sp_prec': 0.8416666666666666, 'sp_recall': 0.6065, 'joint_em': 0.22, 'joint_f1': 0.5573030129149052, 'joint_prec': 0.6965980392156862, 'joint_recall': 0.5085555555555555}
# gemini 2   加入对答案的简洁性  {'em': 0.61, 'f1': 0.7617463238515869, 'prec': 0.7654714307431699, 'recall': 0.7918333333333334, 'sp_em': 0.33, 'sp_f1': 0.6752380952380952, 'sp_prec': 0.8416666666666666, 'sp_recall': 0.6023333333333334, 'joint_em': 0.23, 'joint_f1': 0.5718083576945038, 'joint_prec': 0.7002150204867597, 'joint_recall': 0.5263888888888889}
# gemini 2  加入examples，只有rag    {'em': 0.63, 'f1': 0.7781487501450165, 'prec': 0.7687435608935819, 'recall': 0.8333333333333335, 'sp_em': 0.21, 'sp_f1': 0.617380952380952, 'sp_prec': 0.84, 'sp_recall': 0.524, 'joint_em': 0.15, 'joint_f1': 0.5140527794882014, 'joint_prec': 0.68090756302521, 'joint_recall': 0.45580555555555546}
# gemini 2  加入examples，只有稀疏图    {'em': 0.44, 'f1': 0.5216596532552414, 'prec': 0.5164451970443351, 'recall': 0.5492936507936508, 'sp_em': 0.05, 'sp_f1': 0.3183333333333334, 'sp_prec': 0.46833333333333327, 'sp_recall': 0.25416666666666665, 'joint_em': 0.05, 'joint_f1': 0.23558044733044736, 'joint_prec': 0.32627853037766824, 'joint_recall': 0.20124074074074078}

# llama 3.1 8b {'em': 0.42, 'f1': 0.5538809523809523, 'prec': 0.5869166666666666, 'recall': 0.5487619047619048, 'sp_em': 0.09, 'sp_f1': 0.39133333333333326, 'sp_prec': 0.53, 'sp_recall': 0.3314999999999999, 'joint_em': 0.07, 'joint_f1': 0.26451875901875904, 'joint_prec': 0.36458333333333326, 'joint_recall': 0.22330158730158728}
# llama 3.1 8b {'em': 0.52, 'f1': 0.6338253413253411, 'prec': 0.6389719887955182, 'recall': 0.6506666666666666, 'sp_em': 0.1, 'sp_f1': 0.40685714285714264, 'sp_prec': 0.53, 'sp_recall': 0.3516666666666666, 'joint_em': 0.07, 'joint_f1': 0.2861224700134059, 'joint_prec': 0.3724271708683473, 'joint_recall': 0.2573611111111111}
