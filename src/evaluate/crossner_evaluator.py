from tqdm import tqdm
from os import path, listdir
from tabulate import tabulate

from .BaseMetric import BaseMetric
from .RCMetric import RCMetric


class Evaluator:
    def __init__(self) -> None:
        self.test_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'test')
        self._base_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'test', 'crossner', 'base')
        self._ec_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'test', 'crossner-ec')
        self._tc_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'test', 'crossner-tc')
        self._results_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'test', 'crossner-results')

        self._sources = ['ai', 'literature', 'music', 'politics', 'science']

    def evaluate(self, model_name: str) -> str:
        tables = {
            'ai': {'method': [],'recall': [], 'precision': [], 'f1': []},
            'literature': {'method': [],'recall': [], 'precision': [], 'f1': []},
            'music': {'method': [],'recall': [], 'precision': [], 'f1': []},
            'politics': {'method': [],'recall': [], 'precision': [], 'f1': []},
            'science': {'method': [],'recall': [], 'precision': [], 'f1': []},
            'avg': {'method': [],'recall': [], 'precision': [], 'f1': []},
        }

        result_jsonl = path.join(self._ec_dir, model_name, 'cross.base.jsonl')
        all_precision, all_recall, all_f1 = 0, 0, 0
        for source in self._sources:
            metric = BaseMetric(source=source, result_jsonl=result_jsonl, base_json=path.join(self._base_dir, 'cross.json'))
            precision, recall, f1 = metric.calc_metric()
            all_precision += precision
            all_recall += recall
            all_f1 += f1
            tables[source]['method'].append(f'(Base Extractor)')
            tables[source]['precision'].append(precision)
            tables[source]['recall'].append(recall)
            tables[source]['f1'].append(f1)
        tables['avg']['method'].append(f'(Base Extractor)')
        tables['avg']['precision'].append(all_precision / 5)
        tables['avg']['recall'].append(all_recall / 5)
        tables['avg']['f1'].append(all_f1 / 5)

        result_jsonl = path.join(self._tc_dir, model_name, 'cross.ec.jsonl')
        all_precision, all_recall, all_f1 = 0, 0, 0
        for source in self._sources:
            metric = BaseMetric(source=source, result_jsonl=result_jsonl, base_json=path.join(self._base_dir, 'cross.json'))
            precision, recall, f1 = metric.calc_metric()
            all_precision += precision
            all_recall += recall
            all_f1 += f1
            tables[source]['method'].append(f'(EC)')
            tables[source]['precision'].append(precision)
            tables[source]['recall'].append(recall)
            tables[source]['f1'].append(f1)
        tables['avg']['method'].append(f'(EC)')
        tables['avg']['precision'].append(all_precision / 5)
        tables['avg']['recall'].append(all_recall / 5)
        tables['avg']['f1'].append(all_f1 / 5)

        result_jsons = listdir(path=path.join(self._results_dir, model_name))
        for result_json in tqdm(result_jsons, total=len(result_jsons)):
            chunk_size = result_json.split('.')[2]
            doc_count = result_json.split('.')[3]
            all_precision, all_recall, all_f1 = 0, 0, 0
            for source in self._sources:
                metric = RCMetric(source=source, result_jsonl=path.join(
                    self._results_dir, model_name, result_json
                ),
                base_json=path.join(self._base_dir, 'cross.json'),
                tc_json=path.join(self._tc_dir, model_name, f'cross.tc.{chunk_size}.{doc_count}.json'))
                precision, recall, f1 = metric.calc_metric()
                all_precision += precision
                all_recall += recall
                all_f1 += f1
                tables[source]['method'].append(f'({chunk_size}:{doc_count})')
                tables[source]['precision'].append(precision)
                tables[source]['recall'].append(recall)
                tables[source]['f1'].append(f1)
            tables['avg']['method'].append(f'({chunk_size}:{doc_count})')
            tables['avg']['precision'].append(all_precision / 5)
            tables['avg']['recall'].append(all_recall / 5)
            tables['avg']['f1'].append(all_f1 / 5)

        with open(path.join(self.test_dir, f'{model_name}.txt'), 'w') as f:
            lines = []
            for table_name, table in tables.items():
                table = tabulate(table, headers='keys').split('\n')
                for i in range(len(table)):
                    table[i] += '\n'
                table_width = len(table[1])
                lines.append(f'||{table_name}' + f'=' * (table_width - len(f'||{table_name}') - 1) + '\n')
                lines.extend(table)
            f.writelines(lines)
        return str(path.join(self.test_dir, f'{model_name}.txt'))