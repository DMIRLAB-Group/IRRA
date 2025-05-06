import json
from .utils import extract_outer_braces


types_lists = {
            'ai': ['algorithm', 'conference', 'country', 'field', 'location', 'metrics', 'misc', 'organisation', 'person', 'product', 'programlang', 'researcher', 'task', 'university'],
            'literature': ['award', 'book', 'country', 'event', 'literarygenre', 'location', 'magazine', 'misc', 'organisation', 'person', 'poem', 'writer'],
            'music': ['album', 'award', 'band', 'country', 'event', 'location', 'misc', 'musicalartist', 'musicalinstrument', 'musicgenre', 'organisation', 'person', 'song'],
            'politics': ['country', 'election', 'event', 'location', 'misc', 'organisation', 'person', 'politicalparty', 'politician'],
            'science': ['academicjournal', 'astronomicalobject', 'award', 'chemicalcompound', 'chemicalelement', 'country', 'discipline', 'enzyme', 'event', 'location', 'misc', 'organisation', 'person', 'protein', 'scientist', 'theory', 'university']
        }


class RCMetric:
    def __init__(self,
                 source: str,
                 result_jsonl: str,
                 base_json: str,
                 tc_json: str) -> None:

        self._pd = []
        self._gt = []

        with open(base_json, 'r') as f:
            raw = json.load(f)

        with open(tc_json, 'r') as f:
            entities_json = json.load(f)

        count = 0
        for datum in raw:
            id = datum['id']
            output = json.loads(datum['output'])
            if datum['source'] != source and source != 'all':
                continue
            for entity_type, entities in output.items():
                for entity in entities:
                    self._gt.append((
                        id,
                        entity.strip(),
                        entity_type.strip().lower()
                    ))

        with open(result_jsonl, 'r') as f:
            line = f.readline()
            while line:
                line = json.loads(line.strip())
                id = line['label']
                if entities_json[count]['source'] != source and source != 'all':
                    count += 1
                    line = f.readline()
                    continue
                try:
                    output = json.loads(extract_outer_braces(line['predict'])[0])
                    entity = entities_json[count]['entity']
                    entity_type = output['entity_type'].replace(' ', '')
                    if entity_type not in types_lists[entities_json[count]['source']] and entity_type not in entities_json[count]['types']:
                        raise Exception(f'entity type Error: {entity_type}')
                except Exception as e:
                    entity = entities_json[count]['entity']
                    for entity_type in entities_json[count]['types']:
                        self._pd.append((
                            id,
                            entity.strip(),
                            entity_type.strip().lower()
                            ))
                    line = f.readline()
                    count += 1
                    continue
                self._pd.append((
                    id,
                    entity.strip(),
                    entity_type.strip().lower()))
                line = f.readline()
                count += 1

    def calc_metric(self):
        '''
        :param y_gt: [(tuple), ...]
        :param y_pd: [(tuple), ...]
        :return:
        '''
        y_pd = list(self._pd)
        
        y_gt = list(self._gt)
        num_proposed = len(y_pd)
        num_gold = len(y_gt)

        y_gt_set = set(y_gt)
        num_correct = 0
        for item in y_pd:
            if item in y_gt_set:
                num_correct += 1
            else:
                print(item)

        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 1.0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 1.0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        return precision, recall, f1