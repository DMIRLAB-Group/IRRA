import json
from os import path
from tqdm import tqdm
from uuid import uuid5, NAMESPACE_X500


class Builder:
    def __init__(self) -> None:
        self._raw_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'test', 'crossner', 'raw')
        self._base_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'test', 'crossner', 'base')
        self._sources = ['ai', 'literature', 'music', 'politics', 'science']
        self._schemas = {
            'ai': ['algorithm', 'conference', 'country', 'field', 'location', 'metrics', 'misc', 'organisation', 'person', 'product', 'programlang', 'researcher', 'task', 'university'],
            'literature': ['award', 'book', 'country', 'event', 'literarygenre', 'location', 'magazine', 'misc', 'organisation', 'person', 'poem', 'writer'],
            'music': ['album', 'award', 'band', 'country', 'event', 'location', 'misc', 'musicalartist', 'musicalinstrument', 'musicgenre', 'organisation', 'person', 'song'],
            'politics': ['country', 'election', 'event', 'location', 'misc', 'organisation', 'person', 'politicalparty', 'politician'],
            'science': ['academicjournal', 'astronomicalobject', 'award', 'chemicalcompound', 'chemicalelement', 'country', 'discipline', 'enzyme', 'event', 'location', 'misc', 'organisation', 'person', 'protein', 'scientist', 'theory', 'university']
        }

    def build(self, num_schema: int = 3) -> int:
        all_save = []
        for source in self._sources:
            cur_save = []
            schema = self._schemas[source]
            k, m = divmod(len(schema), num_schema)
            with open(path.join(self._raw_dir, f'{source}.json')) as s:
                save = json.load(s)
                for item in tqdm(save, total=len(save)):
                    instruction = 'Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.'
                    sentence = item.pop('sentence')
                    id = str(uuid5(NAMESPACE_X500, sentence))
                    schemas = (schema[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_schema))
                    for cur_schema in schemas:
                        output = {}
                        for entity_type in cur_schema:
                            output[entity_type] = []
                        for entity in item['entities']:
                            if output.get(entity['type'], None) is not None:
                                output[entity['type']].append(entity['name'])
                            elif entity['type'] not in schema:
                                print(entity['type'])
                        output = json.dumps(output, ensure_ascii=False)
                        cur_save.append({
                            'id': id,
                            'source': source,
                            'instruction': json.dumps({
                                'instruction': instruction,
                                'schema': cur_schema,
                                'input': sentence
                            }, ensure_ascii=False),
                            'output': output
                        })
            all_save.extend(cur_save)
            with open(path.join(self._base_dir, f'{source}.json'), 'w') as t:
                json.dump(cur_save, t, ensure_ascii=False, indent=2)

        with open(path.join(self._base_dir, 'cross.json'), 'w') as t:
                json.dump(all_save, t, ensure_ascii=False, indent=2)
        return len(all_save)