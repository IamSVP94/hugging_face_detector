import cv2
import albumentations

from utills import plt_show_img

img_path = '/home/vid/hdd/file/project/85_РУСАГРО/24_МешкиПодсчет/dataset/bag_person_coco_format/train/img_13_RUSAGRO_10_00002.jpg'
txt_path = '/home/vid/hdd/file/project/85_РУСАГРО/24_МешкиПодсчет/dataset/bag_person_coco_format/train/img_13_RUSAGRO_10_00002.txt'
labels = ['bag', 'person']

img = cv2.imread(str(img_path))

bboxes = []
with open(txt_path, 'r') as rf:
    marks = rf.read().splitlines()
    h_img, w_img, _ = img.shape
    for line in marks:
        cl, x_c, y_c, w, h = map(float, line.split(' '))

        x_min = w_img * (x_c - w / 2)
        x_max = w_img * (x_c + w / 2)
        y_min = h_img * (y_c - h / 2)
        y_max = h_img * (y_c + h / 2)
        box = [x_min, y_min, x_max, y_max, cl]
        bboxes.append(box)

w, h = 864, 480
transform = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.RandomSizedBBoxSafeCrop(p=1, height=h, width=w, erosion_rate=0.1),
        albumentations.BBoxSafeRandomCrop(p=0.5, erosion_rate=0.1),
        albumentations.Resize(height=h, width=w, always_apply=True),
    ],
    bbox_params=albumentations.BboxParams(
        format="pascal_voc", min_visibility=0.2,
    ),
)

transformed = transform(image=img, bboxes=bboxes)
drawed = transformed['image']
h_img, w_img, _ = drawed.shape
for box in transformed['bboxes']:
    x_min, y_min, x_max, y_max, cl = map(int, box)
    cv2.rectangle(drawed, (x_min, y_min), (x_max, y_max), (0, 255, 0))
plt_show_img(drawed)
