import json
import random
from os import path
from tqdm import tqdm
from typing import Dict
from uuid import uuid5, NAMESPACE_X500

class Builder:
    schema = {
        "Amenity": ["Amenity", "facility", "service", "convenience"],
        "title": ["title", "heading"],
        "机构": ["机构", "组织机构"],
        "actor": ["actor", "performer"],
        "地理位置": ["地理位置", "位置", "地点", "地方"],
        "percent": ["percent", "percentage", "proportion", "fraction"],
        "manufacturing process": ["manufacturing process", "production process", "fabrication process", "assembly process"],
        "quantity": ["quantity", "amount", "number", "volume"],
        "media": ["media"],
        "vehicle range": ["vehicle range", "driving range", "range of motion", "travel distance"],
        "trailer": ["trailer", "preview", "teaser", "promo"],
        "Chemical": ["Chemical"],
        "position of vehicle": ["position of vehicle", "vehicle orientation", "vehicle alignment", "vehicle placement"],
        "Location": ["Location", "place", "site", "position"],
        "Anatomy": ["Anatomy"],
        "姓名": ["姓名", "人名", "名字"],
        "location": ["location", "place", "site", "position"],
        "民族": ["民族"],
        "vehicle model": ["vehicle model", "car model", "automobile design"],
        "bus": ["bus"],
        "facility": ["facility", "establishment"],
        "motorcycle": ["motorcycle", "motorbike", "bike", "scooter"],
        "SUV": ["SUV", "sport utility vehicle"],
        "专业": ["专业", "学科", "领域", "行业"],
        "cell type": ["cell type", "cell kind", "cell category", "cell variety"],
        "rating": ["rating"],
        "product": ["product", "item", "goods"],
        "Cuisine": ["Cuisine", "meal", "food"],
        "work of art": ["work of art", "artwork", "art piece", "masterpiece"],
        "food": ["food", "Cuisine", "meal"],
        "Hours": ["Hours", "time"],
        "song": ["song"],
        "money": ["money", "currency"],
        "国籍": ["国籍"],
        "exact location": ["exact location"],
        "vehicle velocity": ["vehicle velocity", "speed", "rate of motion"],
        "material": ["material", "substance", "compound"],
        "protein": ["protein", "polypeptide", "biomolecule"],
        "organization": ["organization", "organisation", "institution", "association", "agency"],
        "manufacturing standard": ["manufacturing standard", "production guideline", "industry standard", "quality standard"],
        "geographical social political": ["geographical social political", "socio-political geographic", "culturally geographic", "regionally significant"],
        "vehicle type": ["vehicle type", "model of vehicle", "category of vehicle", "class of vehicle"],
        "application": ["application", "app", "software", "program"],
        "process parameter": ["process parameter", "process variable", "operating parameter", "control parameter"],
        "cell line": ["cell line"],
        "人名": ["人名", "姓名", "名字", "称呼"],
        "DNA": ["DNA", "deoxyribonucleic acid", "genetic material", "genetic code"],
        "event": ["event"],
        "电影": ["电影", "影片", "片子", "电影作品"],
        "公司": ["公司", "企业"],
        "geographical phenomenon": ["geographical phenomenon", "natural phenomenon", "geophysical event", "environmental occurrence"],
        "MPV": ["MPV", "multi-purpose vehicle", "people carrier"],
        "orientation of vehicle": ["orientation of vehicle"],
        "engineering feature": ["engineering feature", "engineering attribute"],
        "游戏": ["游戏"],
        "ordinal": ["ordinal", "sequential", "ordered", "ranked"],
        "sedan": ["sedan", "saloon", "family car"],
        "Disease": ["Disease", "illness"],
        "enabling technology": ["enabling technology"],
        "brand of vehicle": ["brand of vehicle"],
        "GENE": ["GENE", "genetic material", "genetic code"],
        "date": ["date"],
        "景点": ["景点", "旅游景点", "名胜", "观光地"],
        "area": ["area", "region", "zone"],
        "RNA": ["RNA", "ribonucleic acid", "nucleic acid", "genetic material"],
        "disease": ["disease", "illness"],
        "truck": ["truck", "lorry", "freighter"],
        "Rating": ["Rating"],
        "machine or equipment": ["machine or equipment", "apparatus", "machinery", "device"],
        "instrument": ["instrument", "machine or equipment"],
        "time": ["time"],
        "vintage car": ["vintage car", "classic car"],
        "concept or principle": ["concept or principle"],
        "plot": ["plot"],
        "vehicle": ["vehicle", "transport"],
        "law": ["law"],
        "Price": ["Price", "cost", "value", "expense"],
        "coupe": ["coupe", "sport coupe", "compact car"],
        "review": ["review"],
        "biology": ["biology"],
        "plant": ["plant", "flora"],
        "组织机构": ["组织机构", "组织", "机构"],
        "astronomical object": ["astronomical object", "space object"],
        "character": ["character", "personage", "role", "figure"],
        "sports car": ["sports car", "sporty car"],
        "职位": ["职位", "岗位", "职务", "工作"],
        "人物": ["人物", "角色"],
        "cardinal": ["cardinal", "chief", "principal", "primary"],
        "else": ["else", "otherwise", "moreover", "besides", "miscellaneous", "misc"],
        "national religious political": ["national religious political"],
        "machanical property": ["machanical property", "material property", "physical property"],
        "director": ["director", "filmmaker", "manager", "supervisor"],
        "process characterization": ["process characterization"],
        "color of vehicle": ["color of vehicle"],
        "animal": ["animal", "creature", "fauna"],
        "biomedical": ["biomedical", "medical", "health-related"],
        "学历": ["学历", "教育程度", "学位", "文凭"],
        "river": ["river", "waterway"],
        "书名": ["书名", "图书名称", "书籍标题", "作品名"],
        "person": ["person", "individual", "human", "being"],
        "Dish": ["Dish"],
        "estate car": ["estate car"],
        "hatchback": ["hatchback"],
        "mythical figure": ["mythical figure", "legendary character", "mythological entity", "fabled being"],
        "road": ["road", "way"],
        "地址": ["地址", "地点", "住址"],
        "roadster": ["roadster"],
        "year": ["year"],
        "政府": ["政府", "官方"],
        "籍贯": ["籍贯", "出生地", "家乡", "原籍"],
        "van": ["van"],
        "average ratings": ["average ratings"],
        "genre": ["genre", "category", "type", "style"],
        "language": ["language", "tongue", "lingo"],
        "Restaurant Name": ["Restaurant Name", "dining establishment"]
    }

    def __init__(self) -> None:
        random.seed(42)
        self._iepile_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'train', 'iepile')
        self._augmentation_dir = path.join(path.dirname(path.dirname(path.dirname(__file__))), 'data', 'train', 'iepile-augmentation')

    def build(self) -> Dict[str, int]:
        lens = {
            'train': 0,
            'dev': 0
        }
        for division in ['train']:
            save = []
            with open(path.join(self._iepile_dir, f'{division}.ner.jsonl')) as s:
                lines = s.readlines()
                for line in tqdm(lines, total=len(lines)):
                    # 重构实体输出
                    line = json.loads(line.strip())
                    output_count = 0
                    old_output = json.loads(line['output'])
                    for entity_type, entities in old_output.items():
                        new_entities = []
                        for entity in entities:
                            output_count += 1
                            new_entities.append(entity)
                        old_output[entity_type] = new_entities


                    instruction = json.loads(line['instruction'])
                    task = line['task']
                    source = line['source'] + '_aug'

                    # 打乱schema，增加难度
                    random.shuffle(instruction['schema'])
                    
                    # 70%的数据随机替换类型，增加难度
                    if random.random() < 0.7:
                        new_schema = []
                        rev_schema = {}
                        for entity_type in instruction['schema']:
                            new_entity_type = random.choice(Builder.schema[entity_type])
                            if not rev_schema.get(new_entity_type, None):
                                rev_schema[new_entity_type] = entity_type
                                new_schema.append(new_entity_type)
                        
                        # 是否随机插入相同类型，增加相似类型抽取的概率
                        while random.random() < 0.05:
                            entity_type = random.choice(instruction['schema'])
                            new_entity_type = random.choice(Builder.schema[entity_type])
                            if not rev_schema.get(new_entity_type, None):
                                rev_schema[new_entity_type] = entity_type
                                new_schema.append(new_entity_type)
                        
                        # 对齐schema顺序后插入
                        output = {}
                        for entity_type in new_schema:
                            output[entity_type] = old_output[rev_schema[entity_type]]
                        instruction['schema'] = new_schema
                        save.append({
                            'id': str(uuid5(NAMESPACE_X500, str(instruction['input']))),
                            'task': task,
                            'source': source,
                            'instruction': json.dumps(instruction, ensure_ascii=False),
                            'output': json.dumps(output, ensure_ascii=False)
                        })
                    else:
                        # 对齐schema顺序后插入
                        output = {}
                        for entity_type in instruction['schema']:
                            output[entity_type] = old_output[entity_type]
                        save.append({
                            'id': str(uuid5(NAMESPACE_X500, str(instruction['input']))),
                            'task': task,
                            'source': source,
                            'instruction': json.dumps(instruction, ensure_ascii=False),
                            'output': json.dumps(output, ensure_ascii=False)
                        })

            lens[division] = len(save)

            with open(path.join(self._augmentation_dir, f'{division}.ner.json'), 'w') as t:
                json.dump(save, t, ensure_ascii=False, indent=1)

        return lens