import argparse
import csv
import glob
import os
import time
import json

import cv2
import numpy as np
import torch


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_id, class_name = row
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def detect_image(image_path, model_path, class_list):
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=',', quotechar='"'))

    labels = {}
    for value, key in classes.items():
        labels[key - 1] = value

    from retinanet.retinanet import model as model_package
    model = model_package.resnet50(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    img_index = 0
    img_num = len(glob.glob1(image_path, '*.jpg'))
    t0 = time.time()
    for img_name in os.listdir(image_path):
        if not img_name.endswith('jpg'):
            continue
        prediction_filename = f'retinanet_{os.path.splitext(img_name)[0]}.json'
        # if os.path.exists(os.path.join(image_path, prediction_filename)):
        #     continue
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        t1 = time_sync()
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():
            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()
            scores, classification, transformed_anchors = model(image.cuda().float())
            idxs = np.where(scores.cpu() > 0.5)
            bboxes_x_xx_y_yy = []
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                # label_name = labels[int(classification[idxs[0][j]])]
                # score = scores[j]
                # caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                # draw_caption(image_orig, (x1, y1, x2, y2), caption)
                # cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                bboxes_x_xx_y_yy.append([x1, x2, y1, y2])
            img_index += 1
            # with open(os.path.join(image_path, prediction_filename), 'w') as f:
            #     json.dump({'scores': scores[idxs].cpu().numpy().tolist(),
            #                "classification": classification[idxs].cpu().numpy().astype(np.int).tolist(),
            #                'bboxes_x_xx_y_yy': bboxes_x_xx_y_yy}, f)

            t2 = time_sync()
            print(f'{img_index}/{img_num} ({t2 - t1:.3f}s) Saving {prediction_filename}')
            # cv2.imshow('detections', image_orig)
            # cv2.waitKey(0)
    print(f'Retinanet detection done!. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list)
