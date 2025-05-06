import json
from os import path
from tqdm import tqdm
from typing import Dict
from uuid import uuid5, NAMESPACE_X500

class Builder:
    def __init__(self, model_name: str) -> Dict[str, int]:
        self._model_name = model_name
        self._augmentation_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'train', 'iepile-augmentation')
        self._ec_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'train', 'iepile-ec', self._model_name)

    def build(self, with_dev: bool = False):
        lens = {
            'train': 0,
            'dev': 0
        }
        if with_dev:
            divisions = ['train','dev']
        else:
            divisions = ['train']
        for division in divisions:
            pd = []
            with open(path.join(self._ec_dir, f'{division}.base.jsonl'), 'r') as f:
                line = f.readline()
                while line:
                    pd.append(json.loads(line))
                    line = f.readline()

            gt = []
            with open(path.join(self._augmentation_dir, f'{division}.ner.json'), 'r') as f:
                gt = json.load(f)

            
            save = []
            for pd_item, gt_item in tqdm(zip(pd, gt), total=len(pd)):
                assert pd_item['label'] == gt_item['output']
                instruction = json.loads(gt_item['instruction'])
                if '你' in instruction['instruction']:
                    instruction['instruction'] = '请根据input和schema修正抽取出来的实体。如果你认为可能缺少了某些实体或者对某些内容不理解，请均将其当作实体列出。'
                else:
                    instruction['instruction'] = 'Modify the extracted entities based on input and schema. If you think there may be some entities missing or that you don\'t understand something, please list them as entities.'
                try:
                    instruction['entities'] = json.loads(pd_item['predict'])
                except Exception as _:
                    instruction['entities'] = {}
                output = json.loads(pd_item['label'])
                source = gt_item['source']

                save.append({
                    'id': str(uuid5(NAMESPACE_X500, instruction['input'])),
                    'source': f'{source}_ec',
                    'instruction': json.dumps(instruction, ensure_ascii=False),
                    'output': json.dumps(output, ensure_ascii=False)
                })
            lens[division] = len(save)
            with open(path.join(self._ec_dir, f'{division}.ec.json'), 'w') as f:
                json.dump(save, f, ensure_ascii=False, indent=1)
        
        return lens