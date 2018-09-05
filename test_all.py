import os
import numpy as np

import test_img


def nme(landmarks1, landmarks2, iod):
    landmarks1 = np.array(landmarks1)
    landmarks2 = np.array(landmarks2)
    return np.linalg.norm(landmarks1 - landmarks2) / iod


def run(image_dir,
        annotation_fp,
        model_dir,
        output_dir,
        thresholds=(0.8, 0.8, 0.8),
        min_size=20,
        factor=0.7):
    with open(annotation_fp, 'r') as f:
        annotations = {
            line.rstrip().split(' ')[0]:
                list(map(float, line.rstrip().split(' ')[5:]))
            for line in f.readlines()
        }
    scores = []
    non_detections = 0
    for filename, annotations in annotations.items():
        iod = np.linalg.norm(
            np.array(annotations[0:2]) - np.array(annotations[2:4])
        )
        rectangles, points = test_img.main(
            image_path=os.path.join(image_dir, filename),
            model_dir=model_dir,
            thresholds=thresholds,
            min_size=min_size,
            factor=factor,
            save_image=True,
            save_name=os.path.join(output_dir, filename)
        )
        if len(points) is 0:
            non_detections += 1
            continue
        scores.append(nme(annotations, points[0], iod))
    print("Avg Error: " + str(np.average(scores)))
    print("Non-Detections: " + str(non_detections))


run(image_dir='/home/matt/Desktop/MTCNN_thermal_16bit/images',
    annotation_fp='/home/matt/Desktop/MTCNN_thermal_16bit/test_annotations.txt',
    model_dir='/home/matt/Desktop/MTCNN_thermal_16bit/models',
    output_dir='/home/matt/Desktop/MTCNN_thermal_16bit/results')
