import json
from pathlib import Path
from typing import Union


def load_json(json_path: Union[str, Path]) -> Union[dict, list]:
    """讀取JSON檔案

    Args:
        json_path (Union[str, Path]): 檔案路徑

    Returns:
        Union[dict, list]: JSON物件
    """

    data = None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data