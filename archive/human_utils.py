import json
import numpy as np
import torch

# merged_{x}_{x + w // 2}_{x + w}_{y}_{y + h // 2 + 1}_{y + h}.jpg

def extract_yolo_data(data):
    array = data[0]
    scores = []
    classification = []
    bboxes = []
    if array.ndim == 2:
        for area in range(array.shape[0]):
            x1 = array[area, 0].item() - (array[area, 2].item() // 2)
            y1 = array[area, 1].item() - (array[area, 3].item() // 2)
            x2 = array[area, 0].item() + (array[area, 2].item() // 2)
            y2 = array[area, 1].item() + (array[area, 3].item() // 2)

            scores.append(array[area, 4].item())
            classification.append(np.argmax(array[area, 5:]).item())
            bboxes.append([x1, x2, y1, y2])
    else:
        x1 = array[0].item() - (array[2].item() // 2)
        y1 = array[1].item() - (array[3].item() // 2)
        x2 = array[0].item() + (array[2].item() // 2)
        y2 = array[1].item() + (array[3].item() // 2)

        scores.append(array[4].item())
        classification.append(np.argmax(array[5:]).item())
        bboxes.append([x1, x2, y1, y2])
    return [scores,
            classification,
            bboxes]
            # data[1],
            # data[2],
            # data[3],
            # data[4],
            # data[5],
            # data[6]]



