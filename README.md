# labelme2mask

一個把 Labelme coco json 轉換成 mask image 的工具

## 安裝

```shell
# poetry
poetry install # 正式
poetry install --dev # 開發

# pip
pip install -r requirements.txt # 正式
pip install -r requirements-dev.txt # 開發
```

## 使用說明

- 執行步驟

1. 將使用 Labelme 標註好的 json 檔案放到 `json/` 的資料夾下
2. 準備標籤種類的文字檔，並命名為 `label.txt`
3. 執行 `convert.py`
4. 輸出結果會在 data/ 資料夾下，會有 `imgs/` 與 `masks/` 兩個資料夾

- 標籤種類文字檔說明
  一行代表一個種類，由上至下會由程式自動分配一個索引值，代表他在 Mask Image 中的值。

例如有兩個種類，一個是蘋果（apple）、一個是香蕉（banana），則檔案應該按照下面格式撰寫：

```
apple
banana
```

轉換過後 apple 就會是 0，banana 就會是 1。

- 參數說明

| 參數名稱                | 型態 | 必填 | 預設值       | 說明                         |
| ----------------------- | ---- | ---- | ------------ | ---------------------------- |
| `-a`, `--anno_dir`      | str  |      | ./json       | Labelme 標註 json 檔案資料夾 |
| `-l`, `--label_path`    | str  |      | ./label.txt  | 標籤種類                     |
| `-oi`, `--out_img_dir`  | str  |      | ./data/imgs  | 影像輸出目錄                 |
| `-om`, `--out_mask_dir` | str  |      | ./data/masks | Mask 輸出目錄                |
