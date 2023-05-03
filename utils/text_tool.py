from typing import List


def load_lines( txt_path: str) -> List[str]:
    """讀取多行文字

    Args:
        txt_path (str): 文字檔路徑

    Returns:
        List[str]: 文字列表
    """

    lines: List[str] = list()
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    return lines
