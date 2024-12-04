import cv2
import matplotlib.pyplot as plt
import os


class LabelVerifier:
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

    def check_and_correct_labels(self):
        self.corrected_labels = []
        for class_id, x1, y1, x2, y2 in self.labels:
            x1 = max(0, min(x1, self.image_width - 1))
            x2 = max(0, min(x2, self.image_width - 1))
            y1 = max(0, min(y1, self.image_height -1))
            y2 = max(0, min(y2, self.image_height -1))

            width = (x2 -x1) / self.image_width
            height = (y2 - y1) / self.image_height
            x_center = ((x1 + x2) / 2) / self.image_width
            y_center = ((y1 + y2) / 2) / self.image_height

            width = max(0, width)
            height = max(0, height)
            x_center = max(0, min(x_center, 1))
            y_center = max(0, min(y_center, 1))

            self.corrected_labels.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height
                )
            )
        return self.corrected_labels

    def show_corrected_labels(self):
        self.my_boxes = []
        image = self.image.copy()
        for c_, xc, yc, w, h in self.corrected_labels:
            w = float(w) * 640
            h = float(h) * 640
            x1 = float(xc) * 640 - (w/2)
            y1 = float(yc) * 640 - (h/2)
            x2 = x1 + float(w)
            y2 = y1 + float(h)
            self.my_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        # for box in self.my_boxes:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # cv2.imshow("Corrected labels ...", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def save_corrected_annotations(self):
        my_set = set(self.corrected_labels)
        filename = os.path.basename(self.image_path).replace(
            '.jpg', '.txt'
        )
        output_path = os.path.join(
            "data/cropped_labels_corrected",
            filename
        )
        with open(output_path, "w") as f:
            for c_, xc, yc, w, h in self.corrected_labels:
                f.write(f"{c_} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


if __name__ == '__main__':

    my_labels_paths = os.listdir("data/cropped_labels")
    names = []
    for path_ in my_labels_paths:
        names.append(str(path_).split(".")[0])

    for path_ in names:
        myInstance = LabelVerifier(
            label_path=f"data/cropped_labels/{path_}.txt",
           image_path=f"data/cropped_images/{path_}.jpg"
    )
        myInstance.load_labels()
        # myInstance.show_labels_on_image()
        myInstance.check_and_correct_labels()
        print(myInstance.labels)
        print(myInstance.corrected_labels)
        myInstance.show_corrected_labels()
        myInstance.save_corrected_annotations()

    # # Check.
    # myInstance = LabelVerifier(
    #     label_path="data/cropped_labels_corrected/72_crop_8.txt",
    #     image_path="data/cropped_images/72_crop_8.jpg"
    # )
    # myInstance.load_labels()
    # myInstance.show_labels_on_image()

