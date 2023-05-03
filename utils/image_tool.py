import base64
import cv2
import os
import numpy as np
from pathlib import Path


def base64_to_img(base64_str: str) -> np.ndarray:
    """Base64編碼轉ndarray

    Args:
        base64_str (str): Base64編碼字串

    Returns:
        np.ndarray: 影像ndarray
    """

    img_str = base64.b64decode(base64_str)
    np_arr = np.fromstring(img_str, np.int8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img


def save_img(img: np.ndarray, save_path: str) -> None:
    """儲存圖片

    Args:
        img (np.ndarray): 影像ndarray
        save_path (str): 儲存路徑
    """

    # 1. 檢查路徑
    if not Path(save_path).parent.exists():
        os.makedirs(Path(save_path).parent)

    # 2. 寫入
    cv2.imwrite(str(save_path), img)
