import json
import jsonlines
from tqdm import tqdm
from pathlib import Path
from utills import glob_search


def make_dataset_metadata_file(
        labelme_src_dir, labels,
        start_obj_id: int = 0, start_image_id: int = 0,
):
    imgs = glob_search(labelme_src_dir, shuffle=True)
    img_dicts = []

    p_bar = tqdm(imgs)
    for image_id, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img_dict = {'file_name': str(img_path.name), 'image_id': start_image_id}
        marks_path = img_path.with_suffix('.json')
        if marks_path.exists():
            with open(marks_path) as label_json:
                labelme_body = json.load(label_json)
                img_dict['width'] = labelme_body['imageWidth']
                img_dict['height'] = labelme_body['imageHeight']

                objects_id, objects_area, objects_bbox, objects_category = [], [], [], []
                for l in labelme_body['shapes']:
                    category = labels.index(l['label'])
                    objects_category.append(category)

                    points = l["points"]
                    xs, ys = tuple(map(int, [p[0] for p in points])), tuple(map(int, [p[1] for p in points]))
                    abs_xmin, abs_ymin, abs_xmax, abs_ymax = min(xs), min(ys), max(xs), max(ys)
                    abs_width, abs_height = abs_xmax - abs_xmin, abs_ymax - abs_ymin

                    bbox = [abs_xmin, abs_ymin, abs_width, abs_height]
                    objects_bbox.append(bbox)
                    area = abs_width * abs_height
                    objects_area.append(area)

                    objects_id.append(start_obj_id)
                    start_obj_id += 1
                else:
                    img_dict['objects'] = {
                        'id': objects_id,
                        'area': objects_area,
                        'bbox': objects_bbox,
                        'category': objects_category
                    }
                    img_dicts.append(img_dict)
        start_image_id += 1
    else:
        dst_json_path = Path(labelme_src_dir) / 'metadata.jsonl'
        dst_json_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(str(dst_json_path), 'w') as wf:
            wf.write_all(img_dicts)
    return start_image_id, start_obj_id


labels = ['bag', 'person']
start_image_id, start_obj_id = make_dataset_metadata_file(
    labelme_src_dir='/home/vid/hdd/file/project/85_РУСАГРО/24_МешкиПодсчет/dataset/bag_person_labelme_format/train/',
    labels=labels,
)
make_dataset_metadata_file(
    labelme_src_dir='/home/vid/hdd/file/project/85_РУСАГРО/24_МешкиПодсчет/dataset/bag_person_labelme_format/test/',
    labels=labels,
    start_obj_id=start_obj_id, start_image_id=start_image_id,
)
