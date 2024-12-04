import os
import cv2
import numpy as np
import albumentations as A


img_dir = 'data/cropped_images_corrected/'
label_dir = 'data/cropped_labels_corrected/'
output_img_dir = 'data/augmented_images/'
output_label_dir = 'data/augmented_labels/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def clip_or_exclude_bbox(bbox, image_width, image_height):
    # (class, x_center, y_center, width, height)
    class_id, x_center, y_center, width, height = bbox
    
    # Verificar si el bbox está completamente fuera de los límites
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        return None  # Excluir este bbox completamente

    # Recortar si está parcialmente fuera de los límites
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return (class_id, x_center, y_center, width, height)

def filter_bboxes(bboxes):
    return [bbox for bbox in bboxes if 0 <= bbox[1] <= 1 and 0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1 and 0 <= bbox[4] <= 1]

num_augmentations = 9

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, clip=True))

def read_yolo_label(label_path):
    with open(label_path, 'r') as file:
        labels = [line.strip().split() for line in file]
    return [(int(float(lbl[0])), float(lbl[1]), float(lbl[2]), float(lbl[3]), float(lbl[4])) for lbl in labels]

def save_yolo_label(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

for filename in os.listdir(img_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        image = cv2.imread(img_path)
        labels = read_yolo_label(label_path)

        bboxes = [(lbl[1], lbl[2], lbl[3], lbl[4], lbl[0]) for lbl in labels]
        bboxes = filter_bboxes(bboxes)
        print(label_path)

        for i in range(num_augmentations):
            # Transform 
            transformed = transform(image=image, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = [(bbox[4], bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in transformed['bboxes']]

            # Recortar o excluir los bounding boxes
            clipped_bboxes = [clip_or_exclude_bbox(bbox, transformed_image.shape[1], transformed_image.shape[0]) for bbox in transformed_bboxes]
            
            # Excluir None (bounding boxes fuera de los límites)
            clipped_bboxes = [bbox for bbox in clipped_bboxes if bbox is not None]

            # Save jpg
            output_img_path = os.path.join(output_img_dir, f"{filename.replace('.jpg', '')}_aug_{i}.jpg")
            cv2.imwrite(output_img_path, transformed_image)

            # Save txt
            output_label_path = os.path.join(output_label_dir, f"{filename.replace('.jpg', '')}_aug_{i}.txt")
            save_yolo_label(output_label_path, transformed_bboxes)

