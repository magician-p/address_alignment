import json

from transformers import AutoModelForTokenClassification, AutoTokenizer

from check_address import check_address
from config import *
from get_address import get_address
from predict import Predictor


def address_alignment(texts, model, tokenizer, id2label):
    predictor = Predictor(model, tokenizer, id2label)
    predlabels = predictor.predict(texts)
    informations = get_address(texts, predlabels)
    fixed_informations = check_address(informations)
    return fixed_informations

if __name__ == '__main__':
    texts = [
        "中国浙江省杭州市余杭区葛墩路27号楼傅婷15830444519",
        "北京市市辖区通州区永乐店镇27号楼汪明13334219987",
        "北京市东风街道27号楼汪明13334219987"
        # "高霞 13139243427北京市市辖区东风街道27号楼",
        # "新疆维吾尔自治区划阿拉尔市金杨镇27号楼 刘燕 14727827196",
        # "甘肃省南市文县碧口镇27号楼陈桂兰 13939269190",
        # "陕西省渭南市华阴市罗镇27号楼 赵鑫15687584092",
        # "邱金凤18582166250西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        # "广州市花都区花东镇27号楼张荣   18736672007",
        # "北京市拱辰街道27号楼2单元 陈桂兰 13939269190",
        # "河北省唐山市玉田县林北仓镇27号楼5单元 张荣 18736672007",
        # "河北省朔州市玉田县林南仓镇27号楼5单元 张荣 18736672007"
    ]
    model = AutoModelForTokenClassification.from_pretrained(BEST_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL_PATH if pretrained_exists() else PRETRAINED_MODEL_NAME
    )
    with open(PROCESSED_DATA_PATH / 'id2label.json', 'r', encoding='utf-8') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    fixed = address_alignment(texts, model, tokenizer, id2label)
    print(fixed)