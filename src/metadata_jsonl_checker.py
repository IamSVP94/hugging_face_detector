from pathlib import Path

import cv2
import pandas as pd

from utills import plt_show_img

jsonl_path = '/home/vid/hdd/file/project/85_РУСАГРО/24_МешкиПодсчет/dataset/bag_person_labelme_format/test/metadata.jsonl'

jsonObj = pd.read_json(path_or_buf=jsonl_path, lines=True)

for idx, line in jsonObj.iterrows():
    img_path = Path(jsonl_path).parent / line['file_name']
    img = cv2.imread(str(img_path))
    for box in line['objects']['bbox']:
        x_min, y_min, w, h = box
        cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0))
    plt_show_img(img)
    if idx > 3:
        exit()
