import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from utils.image_tool import base64_to_img, save_img
from utils.json_util import load_json
from utils.text_tool import load_lines

DEFAULT_IDX = 255


def txt_to_label(txt_path: str) -> Dict[str, int]:

    if txt_path is None:
        return {}

    labels = load_lines(txt_path)
    label_to_idx: Dict[str, int] = dict()
    for idx, label in enumerate(labels):
        label_to_idx[label] = idx + 1

    return label_to_idx


def gen_ori_img(
    anno: dict,
    target_shape: Tuple[int,
                        int]) -> Tuple[np.ndarray, int, int, int, int, float]:

    # 1. Base64 轉圖片
    img_base64 = anno.get('imageData')
    img = base64_to_img(img_base64)

    # 2. 計算新的圖片尺寸比例
    h, w, _ = img.shape
    w_target, h_target = target_shape

    h_ratio = h_target / h
    w_ratio = w_target / w
    ratio = min(h_ratio, w_ratio)

    new_h = int(h * ratio)
    new_w = int(w * ratio)

    # 3. 縮放圖片
    img_resize = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. 計算邊界
    top = int((h_target - new_h) / 2)
    bottom = h_target - new_h - top
    left = int((w_target - new_w) / 2)
    right = w_target - new_w - left

    # 5. 填充邊界影像
    img_filled = cv2.copyMakeBorder(img_resize,
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    cv2.BORDER_CONSTANT,
                                    value=(127, 127, 127))

    return img_filled, top, bottom, left, right, ratio


def gen_mask(anno: dict, label_to_idx: Dict[str, int],
             target_shape: Tuple[int, int], top: int, left: int,
             ratio: float) -> np.ndarray:

    # 1. 取得圖片尺寸
    height = anno.get('imageHeight')
    width = anno.get('imageWidth')

    # 2. 產生Mask
    # 產生空白影像
    mask = np.zeros((target_shape[1], target_shape[0], 1), np.uint8)

    # 繪製標記資料(實心填充輪廓)
    shapes = anno.get('shapes')
    for shape in shapes:

        # 取得標籤索引
        label = shape.get('label')
        label_idx = label_to_idx.get(label) or DEFAULT_IDX
        # label_idx = 255

        # 取得輪廓座標點
        points = shape.get('points')
        resize_points: List[List[float, float]] = list()
        for point in points:

            # 計算新的座標點
            x, y = point
            x = (x * ratio) + left
            y = (y * ratio) + top
            resize_points.append([x, y])

        # 繪製輪廓
        np_points = np.array(resize_points, np.int32)
        mask = cv2.drawContours(mask, [np_points], -1, label_idx, -1)

    return mask


def anno_to_mask(
    anno_dir: str,
    label_path: str,
    out_img_dir: str,
    out_mask_dir: str,
    target_shape: Tuple[int, int] = (512, 512)) -> None:

    # 1. 搜尋檔案路徑
    anno_paths = Path(anno_dir).rglob('*.json')

    # 2. 讀取標籤資料
    label_to_idx = txt_to_label(label_path)

    # 3. 處理標記資料
    for anno_path in anno_paths:

        # 讀取標記資料
        anno = load_json(anno_path)
        if not anno:
            continue

        # 原始照片
        img_filled, top, bottom, left, right, ratio = gen_ori_img(
            anno, target_shape)
        img_spath = Path(out_img_dir, f'{anno_path.stem}.jpg')
        save_img(img_filled, img_spath)

        # Mask
        mask = gen_mask(anno, label_to_idx, target_shape, top, left, ratio)
        mask_spath = Path(out_mask_dir, f'{anno_path.stem}_mask.png')
        save_img(mask, mask_spath)


if __name__ == '__main__':

    # 1. 設定參數
    import argparse
    parser = argparse.ArgumentParser(
        description='A tool for convert labelme json to mask image')
    parser.add_argument('-a',
                        '--anno_dir',
                        type=str,
                        default='./json',
                        required=False,
                        help='Directory to label json files')
    parser.add_argument('-l',
                        '--label_path',
                        type=str,
                        default='./label.txt',
                        required=False,
                        help='Path to label text(one label per line)')
    parser.add_argument('-oi',
                        '--out_img_dir',
                        type=str,
                        default='./data/imgs',
                        required=False,
                        help='Directory to output image files')
    parser.add_argument('-om',
                        '--out_mask_dir',
                        type=str,
                        default='./data/masks',
                        required=False,
                        help='Directory to output mask files')

    args = parser.parse_args()

    # 2. 執行轉換
    anno_to_mask(args.anno_dir, args.label_path, args.out_img_dir,
                 args.out_mask_dir)
