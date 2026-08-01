import json
import os
import random as rnd
from tqdm import tqdm
from copy import deepcopy

rnd.seed(42)


def categorize_based_on_relation(questions):
    relations_dict = dict()
    for q_id in questions:
        relation = questions[q_id]['rel']
        if relation not in relations_dict:
            relations_dict[relation] = []
        relations_dict[relation].append(q_id)
    return relations_dict


def extract_questions():
    unanswered_questions = dict()
    for model in MODELS:
        unanswered_questions[model] = dict()
        correctness = dict()
        for i in range(1, 21):
            with open(f'../../dataset/runs/{model}_{Q_TYPE}_{i}.json', 'r') as f:
                run_file = json.load(f)
            for q_id in run_file:
                if q_id not in correctness:
                    correctness[q_id] = []
                correctness[q_id].append(run_file[q_id]['is_correct'])
        for q_id in correctness:
            if correctness[q_id].count(True) > 2 or correctness[q_id].count(True) == 0:
                continue
            if 0 < dataset[model][q_id]['relation_support'] <= SUPPORT_THRESHOLD:
                unanswered_questions[model][q_id] = dataset[model][q_id]
    return unanswered_questions


def is_so(lexical_so, lexical_sro, models_responses):
    unsupport = all([model_resp == 'no' for model_resp in models_responses.values()])
    return lexical_so > 0 and lexical_sro == 0 and unsupport


def is_sro(lexical_so, lexical_sro, models_responses):
    unsupport = all([model_resp == 'no' for model_resp in models_responses.values()])
    return lexical_so > 0 and lexical_sro > 0 and unsupport


def is_support(lexical_so, lexical_sro, models_responses):
    support = all([model_resp == 'yes' for model_resp in models_responses.values()])
    return lexical_so > 0 and lexical_sro == 0 and support


def irrelevant_passage(current_qid, _model, rel):
    random_qid = rnd.choice(relations_dict[_model][rel])
    while random_qid == current_qid:
        random_qid = rnd.choice(relations_dict[_model][rel])
    with open(f'../../dataset/passages/{_model}/{random_qid}.json', 'r') as f:
        rnd_passage_json = json.load(f)
    for _passage in rnd_passage_json:
        if rnd_passage_json[_passage]['relevance']:
            return _passage


def extract_passages(chosen_questions):
    question_passage_dict = dict()
    for model in MODELS:
        question_passage_dict[model] = dict()
        for q_id in tqdm(chosen_questions[model], desc=model):
            with open(f'../../dataset/passages/{model}/{q_id}.json', 'r') as f:
                passages_json = json.load(f)
            passages = {'lexical_so': None, 'lexical_sro': None, 'support': None, 'irrelevant': None}
            for passage in passages_json:
                lexical_so = passages_json[passage]['frequency']
                lexical_sro = passages_json[passage]['lexical_frequency']
                models_responses = passages_json[passage]['models']
                if passages['lexical_so'] is None and is_so(lexical_so, lexical_sro, models_responses):
                    passages['lexical_so'] = passage
                if passages['lexical_sro'] is None and is_sro(lexical_so, lexical_sro, models_responses):
                    passages['lexical_sro'] = passage
                if passages['support'] is None and is_support(lexical_so, lexical_sro, models_responses):
                    passages['support'] = passage
                if passages['irrelevant'] is None:
                    passages['irrelevant'] = irrelevant_passage(q_id, model, dataset[model][q_id]['rel'])
            if None not in passages.values():
                question_passage_dict[model][q_id] = passages
    return question_passage_dict


def ensure_required_data() -> bool:
    missing = []
    dataset_path = '../../dataset/exposureQA.json'
    passages_dir = '../../dataset/passages'
    runs_dir = '../../dataset/runs'

    if not os.path.isfile(dataset_path):
        missing.append(dataset_path)
    if not os.path.isdir(passages_dir):
        missing.append(passages_dir)
    if not os.path.isdir(runs_dir):
        missing.append(runs_dir)

    if missing:
        print('ERROR: Missing required ExposureQA files/directories:')
        for path in missing:
            print(f'- {path}')
        print()
        print('What to do:')
        print('1) Go to the dataset directory of this project.')
        print('2) Run `download.sh`.')
        print('3) Select `ExposureQA-Full` mode to download dataset, passages, and runs.')
        return False

    return True


if __name__ == "__main__":
    Q_TYPE = 'simple'
    SUPPORT_THRESHOLD = 60
    MODELS = ['amber', 'redpajama', 'olmo']

    if ensure_required_data():
        with open('../../dataset/exposureQA.json') as f:
            dataset = json.load(f)
        relations_dict = dict()
        for model in MODELS:
            relations_dict[model] = categorize_based_on_relation(dataset[model])

        chosen_questions = extract_questions()
        chosen_passages = extract_passages(chosen_questions)

        for model in MODELS:
            for q_id in chosen_passages[model]:
                new_item = dataset[model][q_id]
                new_item['passages'] = deepcopy(chosen_passages[model][q_id])
                chosen_passages[model][q_id] = new_item

        with open('./samples.json', 'w') as f:
            json.dump(chosen_passages, f, indent=4)
