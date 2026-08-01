import json
import os
from tqdm import tqdm


class Example_Extractor:
    def __init__(self, model, q_type, conf_method):
        self.supports = dict()
        dataset_model = 'olmo' if model == 'olmo32' else model
        for qid in support_dataset[dataset_model]:
            rel_support = support_dataset[dataset_model][qid]['relation_support']
            if rel_support not in self.supports:
                self.supports[rel_support] = []
            self.supports[rel_support].append(qid)

        self.confidences = dict()
        self.accuracy = dict()
        for qid in acc_conf_dataset[model][q_type]:
            conf = acc_conf_dataset[model][q_type][qid][conf_method]
            acc = acc_conf_dataset[model][q_type][qid]['accuracy']
            if conf not in self.confidences:
                self.confidences[conf] = []
            if acc not in self.accuracy:
                self.accuracy[acc] = []
            self.confidences[conf].append(qid)
            self.accuracy[acc].append(qid)

        self.support_cats = self._categorize(self.supports, ['low', 'medium', 'high'])
        self.accuracy_cats = self._categorize(self.accuracy, ['low', 'high'])
        self.confidences_cats = self._categorize(self.confidences, ['low', 'medium', 'high'])

    def _categorize(self, _dict, _categories):
        num_of_cats = len(_categories)
        sorted_keys = sorted(list(_dict.keys()))
        cat_length = int(len(sorted_keys) / num_of_cats)
        categories = dict()
        for i in range(num_of_cats):
            categories[_categories[i]] = sorted_keys[i * cat_length:(i + 1) * cat_length]
        return categories

    def choose(self, support_category, correctness, confidence_category):
        supports = self.support_cats[support_category]
        confidences = self.confidences_cats[confidence_category]

        if support_category == 'high':
            supports.reverse()
        if confidence_category == 'high':
            confidences.reverse()
        accuracy = int(correctness)

        valid_qids = dict()
        for support in tqdm(supports):
            for confidence in confidences:
                support_qids = set(self.supports[support])
                confidence_qids = set(self.confidences[confidence])
                accuracy_qids = set(self.accuracy[accuracy])
                shared_qids = support_qids.intersection(confidence_qids).intersection(accuracy_qids)
                if len(shared_qids) > 0:
                    valid_qids[f'{support}-{confidence}-{accuracy}'] = list(shared_qids)

        return valid_qids


def ensure_required_data() -> bool:
    missing = []
    dataset_path = '../../dataset/exposureQA.json'
    runs_dir = '../../dataset/runs'
    calibration_results_path = '../calibration/results.json'

    if not os.path.isfile(dataset_path):
        missing.append(dataset_path)
    if not os.path.isdir(runs_dir):
        missing.append(runs_dir)
    if not os.path.isfile(calibration_results_path):
        missing.append(calibration_results_path)

    if missing:
        print('ERROR: Missing required files/directories:')
        for path in missing:
            print(f'- {path}')
        print()
        print('What to do:')
        if dataset_path in missing or runs_dir in missing:
            print('1) Go to the dataset directory of this project.')
            print('2) Run `download.sh`.')
            print('3) Select `ExposureQA-Full` mode to download both the dataset and runs.')
        if calibration_results_path in missing:
            print('4) Run `compute.py` in the calibration experiment to generate `../calibration/results.json`.')
        return False

    return True


if __name__ == "__main__":
    if ensure_required_data():
        with open('../../dataset/exposureQA.json', 'r') as f:
            support_dataset = json.load(f)
        with open('../calibration/results.json', 'r') as f:
            acc_conf_dataset = json.load(f)

        model = 'amber'
        q_type = 'complex'
        conf_method = 'self_consistency'

        extractor = Example_Extractor(model, q_type, conf_method)
        valid_qids = extractor.choose(support_category='high', correctness=True, confidence_category='low')
        for config in valid_qids:
            for qid in valid_qids[config]:
                dataset_model = 'olmo' if model == 'olmo32' else model
                question = support_dataset[dataset_model][qid]['questions'][q_type]
                with open(f'../../dataset/runs/{model}_{q_type}_1.json', 'r') as f:
                    run_json = json.load(f)
                generated_answer = run_json[qid]['answer']
                print(f'[{config}]')
                print('Question:', question)
                print('Answer:', generated_answer)
                input()
