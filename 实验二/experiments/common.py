"""实验任务及与历史数据相同的配置名。"""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Task:
    configuration: str
    execute: Callable[[], dict]


def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding='utf-8'))


def json_value(value):
    """将坐标元组转换为 JSON 数据中采用的列表。"""
    return json.loads(json.dumps(value, ensure_ascii=False))
