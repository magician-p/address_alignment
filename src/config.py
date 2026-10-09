from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# 数据集路径
DATA_ROOT = PROJECT_ROOT / "data"
# 原始数据
RAW_DATA_PATH = DATA_ROOT / "raw" / 'data.jsonl'
# 处理后的数据
PROCESSED_DATA_PATH = DATA_ROOT / "processed"

# 预训练模型加载路径
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "pretrained"
# 预训练模型下载路径
PRETRAINED_MODEL_NAME = "uer/roberta-small-wwm-chinese-cluecorpussmall"

# 检查点保存路径
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = PROJECT_ROOT / "model"

# 日志文件
LOG_ROOT = PROJECT_ROOT / "logs"

# 训练超参
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-5
SAVE_STEPS = 100
EARLY_STOPPING_PATIENCE = 3

# 预训练模型必须存在的文件，缺一个就当作未下载
PRETRAINED_REQUIRED_FILES = (
    'config.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.txt',
    'pytorch_model.bin',
    'model.safetensors'
)

def pretrained_exists():
    return all((PRETRAINED_MODEL_PATH / f).exists() for f in PRETRAINED_REQUIRED_FILES)

@dataclass
class Dataformat:
    Prov: str = ''
    City: str = ''
    District: str = ''
    Town: str = ''
    Detail: str = ''
    Name: str = ''
    Phone: str = ''

MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'Wh199802191835',
    'database': 'address',
    'charset': 'utf8mb4'
}