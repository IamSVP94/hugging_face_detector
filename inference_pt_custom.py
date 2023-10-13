import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
from transformers import pipeline
from utills import cv2_add_title, plt_show_img

img_path = '/home/vid/hdd/file/project/85_РУСАГРО/24_МешкиПодсчет/dataset/bag_person_labelme_format/test/img_16_RUSAGRO_4_00203.jpg'
pretrained_path = '/home/vid/hdd/projects/PycharmProjects/hugging_face_detector/output/checkpoint-800/'

image = Image.open(img_path)

# obj_detector = pipeline("object-detection", model=pretrained_path)
obj_detector = pipeline("object-detection")

image_processor = AutoImageProcessor.from_pretrained(pretrained_path)
model = AutoModelForObjectDetection.from_pretrained(pretrained_path)


def draw(img, box, title=None, color=(0, 255, 0), thickness=3, font_scale=1):
    xmin, ymin, xmax, ymax = list(map(int, box.tolist()))
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    if title is not None:
        img = cv2_add_title(
            img, title, [xmin, ymin, xmax, ymax], where='top', color=color, filled=True,
            font=cv2.FONT_HERSHEY_COMPLEX, font_scale=font_scale, thickness=thickness, )
    return img


with torch.no_grad():
    inputs = image_processor(images=image, return_tensors="pt")
    # print(33, inputs)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

    # torch.onnx.export(
    #     model, (inputs['pixel_values'], None),
    #     f="/home/vid/hdd/projects/PycharmProjects/hugging_face_detector/save/test-ddetr_bagperson.onnx",
    #     input_names=['pixel_values'],
    #     output_names=['logits', 'pred_boxes'],
    #     dynamic_axes={"pixel_values": {0: "batch_size", 1: "image_channel", 2: "image_height", 3: "image_width"}},
    #     do_constant_folding=True,
    #     opset_version=16
    # )

nms_idxs = torchvision.ops.nms(boxes=results['boxes'], scores=results['scores'], iou_threshold=0.1)

results["scores"], results["labels"], results["boxes"] = results["scores"][nms_idxs], results["labels"][nms_idxs], \
    results["boxes"][nms_idxs]

img = cv2.imread(str(img_path))
drawed = img.copy()
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    drawed = draw(drawed, box, title=f'{model.config.id2label[label.item()]}: {score:.2f}', font_scale=0.7, thickness=3)
    print(f"Detected {model.config.id2label[label.item()]} with confidence "
          f"{round(score.item(), 3)} at location {box}"
          )
else:
    plt_show_img(drawed, mode='plt')
