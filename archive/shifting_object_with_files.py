import glob
import logging
import os
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from human_utils import load_detection_data
from retinanet.visualize_single_image_without_nms_with_files import detect_image as retinadetect
from yolov5.detect_without_nms_with_files import run as yolodetect


def load_image(file_path):
    pil_image = Image.open(file_path)
    numpy_image = np.asarray(pil_image, np.int32)
    print(f'loaded image, dimensions: {numpy_image.shape}')
    return numpy_image


def export_image(numpy_image, output_path, show=False):
    pil_image = Image.fromarray(np.uint8(numpy_image))
    pil_image.save(output_path)
    if show:
        pil_image.show()


def get_global_data(human_path, bg_path, target_dir):
    human_matrix = load_image(human_path)
    bg_matrix = load_image(bg_path)
    alpha_mask = human_matrix[..., 3:] / 255.

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    return human_matrix, bg_matrix, alpha_mask


def process_single_image(coords):
    x, y = coords
    local_bg_matrix = bg_matrix.copy()
    if x >= 0:
        w_start = 0
        bg_w_start = x
        if x + w <= bg_w:
            w_end = w
            bg_w_end = x + w
        else:
            w_end = bg_w - x
            bg_w_end = bg_w
    else:
        w_start = abs(x)
        w_end = w
        bg_w_start = 0
        bg_w_end = w - w_start
    if y >= 0:
        h_start = 0
        bg_h_start = y
        if y + h <= bg_h:
            h_end = h
            bg_h_end = y + h
        else:
            h_end = bg_h - y
            bg_h_end = bg_h
    else:
        h_start = abs(y)
        h_end = h
        bg_h_start = 0
        bg_h_end = h - h_start

    local_bg_matrix[bg_h_start:bg_h_end, bg_w_start:bg_w_end] = local_bg_matrix[bg_h_start:bg_h_end,
                                                                bg_w_start:bg_w_end] * (1. - alpha_mask[h_start:h_end,
                                                                                             w_start:w_end])
    local_bg_matrix[bg_h_start:bg_h_end, bg_w_start:bg_w_end] += human_matrix[h_start:h_end, w_start:w_end, :3]

    differences = np.mean(np.abs(local_bg_matrix - bg_matrix), 2)
    nonzero = np.nonzero(differences)
    bb_y1 = min(nonzero[0])
    bb_y2 = max(nonzero[0])
    bb_x1 = min(nonzero[1])
    bb_x2 = max(nonzero[1])

    # fig, ax = plt.subplots()
    # ax.plot([1919, 1920], [1079, 1080])
    # ax.imshow(local_bg_matrix)
    #
    # ax.add_patch(Rectangle((bg_w_start, bg_h_start),
    #                        (bg_w_end - bg_w_start),
    #                        (bg_h_end - bg_h_start),
    #                        fill=False,
    #                        edge_color='blue'))
    # ax.add_patch(Rectangle((bb_x1, bb_y1),
    #                        (bb_x2 - bb_x1),
    #                        (bb_y2 - bb_y1),
    #                        fill=False,
    #                        edge_color='red'))
    # plt.show()
    #     # quit()
    filename = f'merged_{bb_x1}_{x + w // 2}_{bb_x2}_{bb_y1}_{y + h // 2 + 1}_{bb_y2}.jpg'
    image_filepath = os.path.join(target_dir, filename)
    if os.path.isfile(image_filepath):
        return
    export_image(local_bg_matrix, os.path.join(target_dir, filename))


def process_images():
    jobs = []
    for x in range(-w // 2, bg_w - (w // 2), 10):
        for y in range(-h // 2, bg_h - (h // 2) - 1, 10):
            jobs.append((x, y))

    total_iterations = len(list(range(-w // 2, bg_w - (w // 2), 10))) \
                       * len(list(range(-h // 2, bg_h - (h // 2), 10)))

    with Pool() as pool:
        mapped_pool = pool.imap(process_single_image, jobs)
        pbar = tqdm.tqdm(range(len(jobs)))
        for idx in pbar:
            next(mapped_pool)
            pbar.set_description(f'[{idx:05d}/{total_iterations:05d}]')


def yolo_gen(target_dir):
    sec_per_img = 0.1525
    files_num = len(glob.glob1(target_dir, '*.jpg'))
    print('Loading images into YOLOv5...')
    print(f'Expected duration {files_num * sec_per_img} seconds')
    yolodetect(weights='yolov5s.pt',
               source=target_dir)


def retina_gen(target_dir):
    print('Loading images into RetinaNet...')
    retinadetect(
        image_path=target_dir,
        model_path='coco_resnet.pt',
        class_list='retinanet/classes.json')


def bb_iou(bb1, bb2):
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def detector_process(detector, debug=False):
    if detector == 'retina':
        preposition = 'retinanet_'
        extension = '.json'
        iou_tres = 0.45
        score_tres = 0.5
        matrix_filename = 'retina_iou_matrix.npy'
    elif detector == 'yolo':
        preposition = 'yolo_'
        extension = '.pt'
        iou_tres = 0.45
        score_tres = 0.25
        matrix_filename = 'yolo_score_matrix.npy'
    else:
        print('Unknown detector')
        exit()
    files_num = len(glob.glob1(target_dir, '*' + extension))
    ind = 0
    score_matrix = np.zeros((bg_h, bg_w))
    for filepath in glob.glob(os.path.join(target_dir, '*' + extension)):
        scores, classif, bboxes, w_start, w_center, w_end, h_start, h_center, h_end = load_detection_data(filepath)
        boxGT = {'x1': w_start, 'x2': w_end, 'y1': h_start, 'y2': h_end}
        iou_passed_areas = []
        idxs = [idx for idx, val in enumerate(scores) if val > score_tres]
        ious = []
        if debug:
            print(f'{len(idxs)=}')
            imgpath = filepath.replace(preposition, '').replace(extension, '.jpg')
            img = load_image(imgpath)
            fig, ax = plt.subplots()
            ax.plot([1919, 1920], [1079, 1080])
            ax.imshow(img)
            ax.add_patch(Rectangle((boxGT['x1'], boxGT['y1']),
                                   (boxGT['x2'] - boxGT['x1']),
                                   (boxGT['y2'] - boxGT['y1']),
                                   fill=False,
                                   edgecolor='blue'))

        for area in idxs:
            if classif[area] == 0:
                x1, x2, y1, y2 = bboxes[area]
                if x1 > w_end:
                    continue
                if y1 > h_end:
                    continue
                if x2 < w_start:
                    continue
                if y2 < h_start:
                    continue
                boxPred = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                iou = bb_iou(boxGT, boxPred)
                if iou > iou_tres:
                    iou_passed_areas.append(area)
                ious.append(iou)
                if debug:
                    ax.add_patch(Rectangle((boxPred['x1'], boxPred['y1']),
                                           (boxPred['x2'] - boxPred['x1']),
                                           (boxPred['y2'] - boxPred['y1']),
                                           fill=False,
                                           edgecolor='red'))
        max_score = 0.0
        for area in iou_passed_areas:
            if scores[area] > max_score:
                max_score = scores[area]
        if debug:
            if max_score < 0.5:
                plt.show()
                print(f'{max_score=}')
                if ious:
                    print(f'{ious=}')
                print(f'{len(iou_passed_areas)=}')
                print(f'{boxGT=}')
                print(f'{boxPred=}')
                print(filepath)
                plt.close()
        score_matrix[h_center, w_center] = max_score
        if ind % 10 == 0:
            print(f'{detector} processing {ind}/{files_num}')
        ind += 1

    score_matrix = enlarge_matrix_data(score_matrix)
    np.save(os.path.join(target_dir, matrix_filename), score_matrix)
    # if debug:
    print(np.mean(score_matrix))
    print(np.amax(score_matrix))
    print(np.amin(score_matrix))
    ax = plt.subplot()
    im = ax.imshow(score_matrix)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def enlarge_matrix_data(matrix):
    patched_matrix = np.zeros(matrix.shape)
    patched_matrix2 = np.zeros(matrix.shape)

    for i in range(1, 10):
        patched_matrix2[:, i::10] = matrix[:, 0::10]
    for i in range(1, 10):
        patched_matrix2[i::10] = matrix[0::10]

    patched_matrix = np.zeros(matrix.shape)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x, y] != 0:
                patch = matrix[x, y]
                for patch_x in range(x, x + 10):
                    for patch_y in range(y, y + 10):
                        patched_matrix[patch_x, patch_y] = patch

    assert patched_matrix == patched_matrix2

    return patched_matrix


if __name__ == '__main__':
    matplotlib.use('module://backend_interagg')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('whatever')

    human_path = 'humans/human1.png'
    bg_path = 'bgs/park1.jpg'
    target_dir = os.path.join('/', os.sep, 'mnt', 'datagrid', 'personal', 'kolarj55', 'outpainting',
                              'datasets', 'shifted_human', 'v2')
    human_matrix, bg_matrix, alpha_mask = get_global_data(human_path, bg_path, target_dir)
    h, w, _ = human_matrix.shape
    bg_h, bg_w, _ = bg_matrix.shape

    # process_images()
    # yolo_gen(target_dir)
    retina_gen(target_dir)
    # detector_process(detector='retina', debug=False)
    # detector_process(detector='yolo', debug=False)
