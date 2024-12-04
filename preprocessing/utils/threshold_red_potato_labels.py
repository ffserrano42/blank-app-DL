import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


class ThresholdAnnotator:
    def __init__(
        self,
        label_path,
        image_path,
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.image = cv2.imread(image_path)
        self.image_height, self.image_width = self.image.shape[:2]
        self.labels = None
        self.corrected_labels = None


    def load_labels(self):
        self.labels = []
        with open(self.label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(
                    float, line.split()
                )
                x1 = int((x_center - width / 2) * self.image_width)
                y1 = int((y_center - height / 2) * self.image_height)
                x2 = int((x_center + width / 2) * self.image_width)
                y2 = int((y_center + height / 2) * self.image_height)

                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                self.labels.append((class_id, x1, y1, x2, y2))


    def show_labels_on_image(self, color: tuple=(0,255,0)):
        image = self.image.copy()
        for _, x1, y1, x2, y2 in self.labels:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Labels...", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mean_color(self):
        image = self.image.copy()
        mean_colors = []
        for _, x1, y1, x2, y2 in self.labels:
            region = image[y1:y2, x1:x2]
            mean_color = cv2.mean(region)[:3]
            mean_colors.append(mean_color)
        mean_colors = np.array(mean_colors, dtype=np.uint8)
        mean_colors = np.round(mean_colors / 10) * 10
        mean_colors = mean_colors.astype(np.uint8)
        uniques, counts = np.unique(mean_colors, axis=0, return_counts=True)
        max_index = np.argmax(counts)
        max_g = uniques[max_index]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        distances = np.linalg.norm(image_rgb - max_g, axis=2)
        threshold = np.percentile(distances, 50)
        masc = distances <= threshold
        image_rgb[masc] = [255, 0, 0]
        image_result = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Threshold...", image_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





if __name__ == '__main__':
    name = "16_crop_11"
    myInstance = ThresholdAnnotator(
        label_path=f"data/cropped_labels_corrected/{name}.txt",
        image_path=f"data/cropped_images/{name}.jpg",
    )
    myInstance.load_labels()
    myInstance.show_labels_on_image()
    myInstance.mean_color()
