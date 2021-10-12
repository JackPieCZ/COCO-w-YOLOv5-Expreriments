import argparse
import csv
import time
import tqdm

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


def detect_image(image_path, model_path, class_list, score_tres):
    img_arrays, w_starts, w_centers, w_ends, h_starts, h_centers, h_ends = image_path
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

    t0 = time.time()
    results = []
    min_side = 608
    max_side = 1024
    for ind, img_name in enumerate(tqdm.tqdm(img_arrays, desc='RetinaNet processing')):
        # if os.path.exists(os.path.join(image_path, prediction_filename)):
        #     continue
        image = img_name[..., ::-1].astype('float32')
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        # rescale the image so the smallest side is min_side
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
            idxs = np.where(scores.cpu() > score_tres)
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
            # with open(os.path.join(image_path, prediction_filename), 'w') as f:
            #     json.dump({'scores': scores[idxs].cpu().numpy().tolist(),
            #                "classification": classification[idxs].cpu().numpy().astype(np.int).tolist(),
            #                'bboxes_x_xx_y_yy': bboxes_x_xx_y_yy}, f)
            results.append([
                scores[idxs].cpu().tolist(),
                classification[idxs].cpu().numpy().astype(np.int).tolist(),
                bboxes_x_xx_y_yy,
                w_starts[ind],
                w_centers[ind],
                w_ends[ind],
                h_starts[ind],
                h_centers[ind],
                h_ends[ind]])
            # print(f'image {ind}/{len(img_arrays)}')
            # cv2.imshow('detections', image_orig)
            # cv2.waitKey(0)
    # print(f'Retinanet detection done!. ({time.time() - t0:.3f}s)')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list)
