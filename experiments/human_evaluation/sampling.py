import json
from tqdm import tqdm
from copy import deepcopy
import random as rnd

rnd.seed(42)


def sro(rel, passage):
    passage = passage.lower()
    rel = rel.lower()
    return rel in passage


def so(models_results):
    return list(models_results.values()).count('yes') == 0


def support(models_results):
    return list(models_results.values()).count('yes') == 3 or list(models_results.values()).count('yes') == 2


def categorize(model):
    categories = dict()
    questions_keys = deepcopy(list(dataset[model].keys()))
    rnd.shuffle(questions_keys)
    for qid in questions_keys:
        rel = dataset[model][qid]['rel']
        if rel not in categories:
            categories[rel] = dict()
        categories[rel][qid] = dataset[model][qid]
    return categories


def validate_questions(questions, model, rel, reserved):
    categorize_sample = NUM_OF_CATEGORIZE_SAMPLES // 2
    for qid in questions:
        stored_passages = {'sro': {}, 'so': {}, 'support': {}}
        if qid in reserved:
            continue
        with open(f'../../supporting/labelled_passages/{model}/{qid}.json') as f:
            passages = json.load(f)

        # Ascending Order
        for passage in passages:
            if sum([len(val) for val in stored_passages.values()]) >= categorize_sample * 3:
                break

            if sro(rel, passage):
                if len(stored_passages['sro']) < categorize_sample:
                    stored_passages['sro'][passage] = {'models': passages[passage]['models'],
                                                       'support_score': passages[passage]['frequency']}
            if so(passages[passage]['models']):
                if len(stored_passages['so']) < categorize_sample:
                    stored_passages['so'][passage] = {'models': passages[passage]['models'],
                                                      'support_score': passages[passage]['frequency']}
            if support(passages[passage]['models']):
                if len(stored_passages['support']) < categorize_sample:
                    stored_passages['support'][passage] = {'models': passages[passage]['models'],
                                                           'support_score': passages[passage]['frequency']}

        # Descending Order
        passages = dict(sorted(passages.items(), key=lambda x: x[1]['frequency'], reverse=True))
        for passage in passages:
            if sum([len(val) for val in stored_passages.values()]) >= NUM_OF_CATEGORIZE_SAMPLES * 3:
                return qid, stored_passages
            if sro(rel, passage):
                if len(stored_passages['sro']) < NUM_OF_CATEGORIZE_SAMPLES:
                    stored_passages['sro'][passage] = {'models': passages[passage]['models'],
                                                       'support_score': passages[passage]['frequency']}
            if so(passages[passage]['models']):
                if len(stored_passages['so']) < NUM_OF_CATEGORIZE_SAMPLES:
                    stored_passages['so'][passage] = {'models': passages[passage]['models'],
                                                      'support_score': passages[passage]['frequency']}
            if support(passages[passage]['models']):
                if len(stored_passages['support']) < NUM_OF_CATEGORIZE_SAMPLES:
                    stored_passages['support'][passage] = {'models': passages[passage]['models'],
                                                           'support_score': passages[passage]['frequency']}
    return None, None


if __name__ == "__main__":
    NUM_OF_CATEGORIZE_SAMPLES = 4
    NUM_OF_SAMPLES = 1000

    with open('../../final_files/dataset_final.json', 'r') as f:
        dataset = json.load(f)
    samples = dict()
    reserved = []
    for i, model in enumerate(dataset):
        categories = categorize(model)
        qids = dict()
        for rel in tqdm(categories, desc=model):
            num_of_repeat = NUM_OF_SAMPLES // (len(categories) * len(dataset))
            if rel in ['director', 'producer', 'capital', 'occupation']:
                num_of_repeat += 2
            for _ in range(num_of_repeat):
                questions = categories[rel]
                qid, passages = validate_questions(questions, model, rel, reserved)
                if qid is None and passages is None:
                    break
                qids[qid] = passages
                qids[qid]['sub'] = dataset[model][qid]['sub']
                qids[qid]['rel'] = dataset[model][qid]['rel']
                qids[qid]['obj'] = dataset[model][qid]['obj']
                reserved.append(qid)
        samples[model] = qids
    with open('./samples.json', 'w') as f:
        json.dump(samples, f, indent=4)
