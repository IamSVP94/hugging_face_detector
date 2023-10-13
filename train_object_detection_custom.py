import evaluate
import numpy as np
import albumentations
from datasets import load_dataset
from transformers import AutoModelForObjectDetection, TrainingArguments, AutoImageProcessor, Trainer

# /run/user/1000/gvfs/smb-share:server=s003.vmx.org,share=dataset/bag_person_labelme_format/
dataset = load_dataset(
    'imagefolder',
    data_dir='/home/vid/hdd/file/project/85_РУСАГРО/24_МешкиПодсчет/dataset/bag_person_labelme_format/',
)
metric = evaluate.load('repllabs/mean_average_precision')

categories = ['bag', 'person']
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}


# item = dataset['train'][0]
# img = np.array(item['image'])[:, :, ::-1]
# labels = [id2label[id] for id in item['objects']['category']]
# drawed = draw(img=img, item=item, labels=labels, where='bottom')
# plt_show_img(drawed)
# exit()


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [{"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)} for id_, cat_, ar_, box_ in
               zip(image_ids, categories, area, bboxes)]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

height, width, erosion_rate = 480, 864, 0.1  # random
transform = albumentations.Compose(
    [
        albumentations.RandomSizedBBoxSafeCrop(p=0.5, height=height, width=width, erosion_rate=erosion_rate),
        albumentations.BBoxSafeRandomCrop(p=0.5, erosion_rate=erosion_rate),
        albumentations.RandomBrightnessContrast(p=0.3),
        albumentations.HorizontalFlip(p=0.5),
        # albumentations.Resize(height=height, width=width, always_apply=True),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"], min_visibility=0.2, ),
)

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir=f'/home/vid/hdd/projects/PycharmProjects/hugging_face_detector/output_bag_person_50/',
    run_name='DETR_experiment_bag_person_50',
    label_names=categories,
    num_train_epochs=300,
    learning_rate=1e-5,
    weight_decay=1e-4,
    lr_scheduler_type='cosine',  # cosine, constant
    auto_find_batch_size=True,
    report_to='clearml',
    fp16=True,
    save_steps=200,
    logging_steps=200,
    save_strategy='epoch',
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    # load_best_model_at_end=False,
    #     resume_from_checkpoint
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"].with_transform(transform_aug_ann),
    eval_dataset=dataset["test"].with_transform(transform_aug_ann),
    compute_metrics=compute_metrics,
    tokenizer=image_processor,
)

trainer.train()
