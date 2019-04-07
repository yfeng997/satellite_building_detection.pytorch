import os
import json
import cv2

images_dir = './tmp_images'
annotations_dir = './tmp_annotations'
vis_dir = './vis'

for root, dirs, files in os.walk(images_dir):
    for filename in files:
        if filename.endswith('.jpg'):
            annotation_file = os.path.join(annotations_dir, filename[:-4]+'.json')
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            raw_image = cv2.imread(os.path.join(images_dir, filename))
            for anno in annotation:
                bbox = anno['bbox']
                label = anno['class']
                x = int(bbox[0] * 224)
                y = int(bbox[1] * 224)
                w = int(bbox[2] * 224)
                h = int(bbox[3] * 224)
                if label == 1:
                    cv2.rectangle(raw_image, (x, y), (x+w, y+h), (255,0,0), 1)
                else:
                    cv2.rectangle(raw_image, (x, y), (x+w, y+h), (0,255,0), 1)
            cv2.imwrite(os.path.join(vis_dir, filename), raw_image)