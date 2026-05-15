import numpy as np
import os
import torch
import certifi
from PIL import Image
from ybjy_config import CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED_NAME

HF_CACHE_DIR = CACHE_DIR / 'huggingface'
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', str(HF_CACHE_DIR))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TRANSFORMERS_CACHE', str(HF_CACHE_DIR / 'transformers'))
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())

import open_clip


YBJY_TEXT_ALIASES = {
    '人': ['person', 'human being'],
    '人物': ['person', 'human being'],
    '男人': ['man', 'male person'],
    '女人': ['woman', 'female person'],
    '小孩': ['child', 'kid'],
    '儿童': ['child', 'kid'],
    '老人': ['elderly person', 'old person'],
    '非洲人': ['African person', 'Black person'],
    '亚洲人': ['Asian person'],
    '戴眼镜的人': ['person wearing glasses', 'a person with glasses', 'a woman wearing glasses', 'a man wearing glasses'],
    '戴眼镜的非洲人': ['African person wearing glasses', 'Black person wearing glasses', 'person with dark skin wearing glasses'],
    '戴眼镜的女人': ['woman wearing glasses', 'female person with glasses'],
    '戴眼镜的男人': ['man wearing glasses', 'male person with glasses'],
    '困倦的人': ['sleepy person', 'tired person', 'drowsy person'],
    '读书的人': ['person reading a book', 'person reading'],
    '奔跑的人': ['running person', 'person running'],
    '坐着的人': ['sitting person', 'person sitting'],
    '站着的人': ['standing person', 'person standing'],
    '狗': ['dog'],
    '小狗': ['dog', 'puppy'],
    '黄狗': ['yellow dog', 'golden dog', 'tan dog', 'light brown dog', 'dog with yellow fur'],
    '黄色的狗': ['yellow dog', 'golden dog', 'tan dog', 'light brown dog', 'dog with yellow fur'],
    '金色的狗': ['golden dog', 'yellow dog', 'tan dog', 'light brown dog'],
    '金毛': ['golden retriever'],
    '黑狗': ['black dog', 'black Labrador', 'dark dog'],
    '白狗': ['white dog', 'white puppy', 'samoyed'],
    '棕色的狗': ['brown dog', 'tan dog', 'light brown dog'],
    '趴着的狗': ['dog lying down', 'lying dog', 'dog resting on the ground'],
    '躺着的狗': ['dog lying down', 'lying dog', 'dog resting'],
    '坐着的狗': ['sitting dog', 'dog sitting'],
    '奔跑的狗': ['running dog', 'dog running'],
    '玩球的狗': ['dog playing with a ball', 'dog holding a ball'],
    '草地上的狗': ['dog on grass', 'dog in a grassy field'],
    '猫': ['cat'],
    '小猫': ['cat', 'kitten'],
    '趴着的猫': ['cat lying down', 'lying cat', 'cat resting'],
    '睡觉的猫': ['sleeping cat', 'cat sleeping'],
    '玩耍的猫': ['playing cat', 'cat playing'],
    '鸟': ['bird'],
    '飞翔的鸟': ['flying bird', 'bird flying'],
    '青蛙': ['frog'],
    '马': ['horse'],
    '奔跑的马': ['running horse', 'horse running'],
    '飞机': ['airplane', 'airliner'],
    '汽车': ['car', 'automobile'],
    '跑车': ['sports car'],
    '卡车': ['truck', 'trailer truck'],
    '船': ['ship', 'boat'],
    '帆船': ['sailboat', 'schooner'],
}

YBJY_ENTITY_ALIASES = {
    '人': ['person'],
    '人物': ['person'],
    '男人': ['man'],
    '女人': ['woman'],
    '小孩': ['child'],
    '儿童': ['child'],
    '老人': ['elderly person'],
    '非洲人': ['African person', 'Black person'],
    '亚洲人': ['Asian person'],
    '狗': ['dog'],
    '小狗': ['puppy', 'dog'],
    '猫': ['cat'],
    '小猫': ['kitten', 'cat'],
    '鸟': ['bird'],
    '马': ['horse'],
    '汽车': ['car'],
    '跑车': ['sports car'],
    '卡车': ['truck'],
    '船': ['ship', 'boat'],
    '帆船': ['sailboat'],
}

YBJY_MODIFIER_ALIASES = {
    '黄': ['yellow', 'golden', 'tan'],
    '黄色': ['yellow', 'golden', 'tan'],
    '金色': ['golden', 'yellow'],
    '黑': ['black', 'dark'],
    '黑色': ['black', 'dark'],
    '白': ['white'],
    '白色': ['white'],
    '棕': ['brown', 'tan'],
    '棕色': ['brown', 'tan'],
    '红': ['red'],
    '红色': ['red'],
    '蓝': ['blue'],
    '蓝色': ['blue'],
    '绿': ['green'],
    '绿色': ['green'],
    '戴眼镜': ['wearing glasses', 'with glasses'],
    '眼镜': ['wearing glasses', 'with glasses'],
    '困倦': ['sleepy', 'tired', 'drowsy'],
    '清晰': ['clear'],
}

YBJY_ACTION_ALIASES = {
    '趴': ['lying down', 'resting on the ground'],
    '趴着': ['lying down', 'resting on the ground'],
    '躺': ['lying down', 'resting'],
    '躺着': ['lying down', 'resting'],
    '坐': ['sitting'],
    '坐着': ['sitting'],
    '站': ['standing'],
    '站着': ['standing'],
    '跑': ['running'],
    '奔跑': ['running'],
    '走': ['walking'],
    '走路': ['walking'],
    '跳': ['jumping'],
    '跳跃': ['jumping'],
    '飞': ['flying'],
    '飞翔': ['flying'],
    '睡': ['sleeping'],
    '睡觉': ['sleeping'],
    '读书': ['reading a book'],
    '阅读': ['reading'],
    '玩球': ['playing with a ball', 'holding a ball'],
    '玩耍': ['playing'],
}

YBJY_SCENE_ALIASES = {
    '草地': ['on grass', 'in a grassy field'],
    '室内': ['indoors'],
    '室外': ['outdoors'],
    '街道': ['on a street'],
    '海边': ['at the beach'],
    '水里': ['in water'],
    '森林': ['in a forest'],
}


def build_ybjy_text_values(clean_text):
    text_values = [clean_text]
    matched_keys = []
    for key in sorted(YBJY_TEXT_ALIASES.keys(), key=len, reverse=True):
        if key in clean_text:
            if any(key != matched_key and key in matched_key for matched_key in matched_keys):
                continue
            matched_keys.append(key)
            text_values.extend(YBJY_TEXT_ALIASES[key])
    text_values.extend(build_ybjy_composed_text_values(clean_text))
    return list(dict.fromkeys([value for value in text_values if value]))[:36]


YBJY_TEXT_ALIASES.update(
    {
        '\u8f66': ['car', 'vehicle'],
        '\u81ea\u884c\u8f66': ['bicycle', 'bike'],
        '\u9a91\u81ea\u884c\u8f66': ['person riding a bicycle', 'person cycling', 'cyclist riding a bike'],
        '\u4eba\u9a91\u81ea\u884c\u8f66': ['person riding a bicycle', 'person cycling', 'cyclist riding a bike'],
        '\u4e00\u4e2a\u4eba\u9a91\u81ea\u884c\u8f66': ['person riding a bicycle', 'person cycling', 'cyclist riding a bike'],
        '\u96e8\u5929\u8857\u9053\u91cc\u7684\u8f66': ['car on a rainy street', 'vehicle on a wet street', 'car in rainy weather'],
        '\u9a91\u81ea\u884c\u8f66\u7684\u4eba': ['person riding a bicycle', 'person cycling', 'cyclist riding a bike'],
        '\u9a91\u8f66\u7684\u4eba': ['person riding a bicycle', 'person cycling', 'cyclist riding a bike'],
    }
)

YBJY_ENTITY_ALIASES.update(
    {
        '\u8f66': ['car', 'vehicle'],
        '\u81ea\u884c\u8f66': ['bicycle', 'bike'],
    }
)

YBJY_MODIFIER_ALIASES.update(
    {
        '\u96e8\u5929': ['rainy', 'wet'],
        '\u4e0b\u96e8': ['rainy', 'wet'],
    }
)

YBJY_ACTION_ALIASES.update(
    {
        '\u9a91': ['riding'],
        '\u9a91\u8f66': ['riding a bicycle', 'cycling'],
        '\u9a91\u81ea\u884c\u8f66': ['riding a bicycle', 'cycling'],
    }
)

YBJY_SCENE_ALIASES.update(
    {
        '\u96e8\u5929': ['in rainy weather', 'on a wet day'],
        '\u4e0b\u96e8': ['in rainy weather', 'on a wet day'],
        '\u6e7f\u8def\u9762': ['on a wet road'],
    }
)


def collect_ybjy_aliases(clean_text, alias_map):
    values = []
    matched_keys = []
    for key in sorted(alias_map.keys(), key=len, reverse=True):
        if key in clean_text:
            if any(key != matched_key and key in matched_key for matched_key in matched_keys):
                continue
            matched_keys.append(key)
            values.extend(alias_map[key])
    return list(dict.fromkeys(values))


def build_ybjy_composed_text_values(clean_text):
    entities = collect_ybjy_aliases(clean_text, YBJY_ENTITY_ALIASES)
    modifiers = collect_ybjy_aliases(clean_text, YBJY_MODIFIER_ALIASES)
    actions = collect_ybjy_aliases(clean_text, YBJY_ACTION_ALIASES)
    scenes = collect_ybjy_aliases(clean_text, YBJY_SCENE_ALIASES)
    composed_values = []
    people_entities = [value for value in entities if value in {'person', 'man', 'woman', 'child', 'elderly person'}]
    has_bicycle = any(value in {'bicycle', 'bike'} for value in entities)
    has_riding = any(value in {'riding', 'riding a bicycle', 'cycling'} for value in actions)
    if people_entities and has_bicycle and has_riding:
        for person_entity in people_entities[:3]:
            composed_values.append(f'{person_entity} riding a bicycle')
            composed_values.append(f'{person_entity} cycling')
    for entity in entities[:4]:
        if entity in {'bicycle', 'bike'} and has_riding:
            continue
        for modifier in modifiers[:4]:
            composed_values.append(f'{modifier} {entity}')
        for action in actions[:4]:
            composed_values.append(f'{entity} {action}')
            composed_values.append(f'{action} {entity}')
        for scene in scenes[:3]:
            composed_values.append(f'{entity} {scene}')
        for modifier in modifiers[:3]:
            for action in actions[:3]:
                composed_values.append(f'{modifier} {entity} {action}')
                composed_values.append(f'{entity} {action} {modifier}')
    return composed_values


def build_ybjy_prompt_rows(text_values):
    prompt_rows = []
    has_ascii_alias = any(value.isascii() for value in text_values[1:])
    for value_index, value in enumerate(text_values):
        is_ascii_value = value.isascii()
        if value_index == 0 and has_ascii_alias and not is_ascii_value:
            weight = 0.45
        elif value_index == 0:
            weight = 1.15
        elif value_index <= 4:
            weight = 1.55
        else:
            weight = 0.95
        prompt_rows.extend(
            [
                (value, weight),
                (f'a photo of {value}', weight),
                (f'a clear photo showing {value}', weight * 0.95),
                (f'a natural image showing {value}', weight * 0.9),
            ]
        )
    return prompt_rows[:120]


class FrozenYbjyClipService:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.preprocess = None
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self._load_model_once()

    def _load_model_once(self):
        if self.model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED_NAME,
                cache_dir=str(CACHE_DIR / 'open_clip'),
                device=self.device
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess

    def _normalize_feature(self, feature_array):
        feature_array = np.array(feature_array, dtype=np.float32)
        norm_value = np.linalg.norm(feature_array)
        if norm_value == 0:
            return feature_array.tolist()
        return (feature_array / norm_value).tolist()

    def encode_image_to_feature(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature_tensor = self.model.encode_image(image_tensor)
        feature_array = feature_tensor.squeeze(0).cpu().numpy()
        return self._normalize_feature(feature_array)

    def encode_text_to_feature(self, text_value):
        clean_text = text_value.strip()
        text_values = build_ybjy_text_values(clean_text)
        prompt_rows = build_ybjy_prompt_rows(text_values)
        text_prompts = [row[0] for row in prompt_rows]
        prompt_weights = torch.tensor([row[1] for row in prompt_rows], dtype=torch.float32, device=self.device)
        text_tokens = self.tokenizer(text_prompts).to(self.device)
        with torch.no_grad():
            feature_tensor = self.model.encode_text(text_tokens)
            feature_tensor = feature_tensor / feature_tensor.norm(dim=-1, keepdim=True)
            prompt_weights = prompt_weights / prompt_weights.sum()
            feature_tensor = (feature_tensor * prompt_weights.unsqueeze(1)).sum(dim=0)
        feature_array = feature_tensor.cpu().numpy()
        return self._normalize_feature(feature_array)


frozen_ybjy_clip_service = FrozenYbjyClipService()
