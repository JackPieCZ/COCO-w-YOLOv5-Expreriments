import random
import argparse
import datetime
import logging
import os
import time
import glob
from shutil import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from p_tqdm import p_map
from tqdm import tqdm

from human_utils import extract_yolo_data
from retinanet.visualize_single_image_without_nms_without_files import detect_image as retinadetect
from yolov5.detect_without_nms_without_files import run as yolodetect


def tprint(text):
    clock = datetime.timedelta(seconds=time.time() - t0)
    print(f'[{clock}] {text}')


def load_image(file_path):
    pil_image = Image.open(file_path)
    numpy_image = np.asarray(pil_image, np.int8)
    return numpy_image


def get_global_data(human_path, bg_path):
    human_matrix = load_image(human_path)
    bg_matrix = load_image(bg_path)
    alpha_mask = human_matrix[..., 3:] / 255.

    return human_matrix, bg_matrix, alpha_mask


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


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

    differences = local_bg_matrix - bg_matrix
    nonzero = np.nonzero(differences)
    bb_y1 = min(nonzero[0])
    bb_y2 = max(nonzero[0])
    bb_x1 = min(nonzero[1])
    bb_x2 = max(nonzero[1])

    return [
        local_bg_matrix,
        bb_x1,
        x + w // 2,
        bb_x2,
        bb_y1,
        y + h // 2 + 1,
        bb_y2
    ]


def process_images():
    jobs = []
    for x in range(-w // 2, bg_w - (w // 2), 10):
        for y in range(-h // 2, bg_h - (h // 2) - 1, 10):
            jobs.append((x, y))
    logging.debug('Splitting coords into chunks')
    chunks = list(divide_chunks(jobs, chunk_size))
    tprint('Coords split into chunks')

    for idx, chunk in enumerate(chunks):
        tprint(f'STARTING CHUNK {idx}/{len(jobs) // chunk_size}')
        logging.info(f'Starting chunk {idx}/{len(jobs) // chunk_size}')
        process_chunk_of_images(chunk)

    tprint('Both plots finished')
    mat_filename = f'{human_path[human_path.find("/") + 1:human_path.find(".")]}_{bg_path[bg_path.find("/") + 1:bg_path.find(".")]}'
    yolo_version = yolo_weights[yolo_weights.find('yolov5') + 6:yolo_weights.find('.pt')]
    np.save(os.path.join(target_dir, mat_filename + '_yolov5' + yolo_version + '_score_matrix.npy'), yolo_score_matrix)
    np.save(os.path.join(target_dir, mat_filename + '_retinaNet_score_matrix.npy'), retina_score_matrix)

    yolo_score_matrix[bg_h, bg_w] = 1.0
    retina_score_matrix[bg_h, bg_w] = 1.0
    ax = plt.subplot()
    im = ax.imshow(yolo_score_matrix)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(f'YOLOv5{yolo_version} score matrix with {mat_filename}')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, cax=cax, label='Objectness Score')
    plt.savefig(os.path.join(target_dir, mat_filename + '_yolov5' + yolo_version + '_score_matrix.png'))
    plt.show()

    ax = plt.subplot()
    im = ax.imshow(retina_score_matrix)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(f'RetinaNet score matrix with {mat_filename}')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    plt.colorbar(im, cax=cax, label='Objectness Score')
    plt.savefig(os.path.join(target_dir, mat_filename + '_retinaNet_score_matrix.png'))
    plt.show()

    copy('debug.log', target_dir)
    logging.info(f'Finished in {datetime.timedelta(seconds=time.time() - t0)}')


def process_chunk_of_images(chunk):
    global yolo_score_matrix
    global retina_score_matrix

    chunk_data = p_map(process_single_image, chunk, desc='Processing chunk of images')
    tprint('Image data processed')
    logging.info('Created images to process with shifted object')
    chunk_data = [list(x) for x in zip(*chunk_data)]

    tprint('Starting Yolo')
    logging.info('Starting to process data in Yolo')
    chunk_results = yolodetect(weights=yolo_weights,
                               source=chunk_data,
                               score_tres=yolo_score_tres)
    logging.debug('Detection data processing')
    detector_data = [extract_yolo_data(x) for x in chunk_results]
    score_matrix = detector_process(results=detector_data)
    tprint('Updating Yolo score matrix')
    logging.info('Updating Yolo score matrix')
    yolo_score_matrix += score_matrix

    tprint('Starting RetinaNet')
    logging.info('Starting to process data in RetinaNet')
    detector_data = retinadetect(image_path=chunk_data,
                                 model_path=retina_weights,
                                 class_list='retinanet/classes.json',
                                 score_tres=retina_score_tres)
    logging.debug('Detection data processing')
    score_matrix = detector_process(results=detector_data)
    tprint('Updating RetinaNet score matrix')
    logging.info('Updating RetinaNet score matrix')
    retina_score_matrix += score_matrix


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


def detector_process(results):
    score_matrix = np.zeros((bg_h, bg_w))
    for result in tqdm(results, desc='Processing detection results'):
        scores, classif, bboxes, w_start, w_center, w_end, h_start, h_center, h_end = result
        boxGT = {'x1': w_start, 'x2': w_end, 'y1': h_start, 'y2': h_end}
        iou_passed_areas = []

        for area in range(len(scores)):
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

        max_score = 0.0
        for area in iou_passed_areas:
            if scores[area] > max_score:
                max_score = scores[area]
        score_matrix[h_center:h_center + 10, w_center:w_center + 10] = max_score
    return score_matrix


def get_random_file(ext, top):
    _top = top
    ct, limit = 0, 5000
    while True:
        if ct > limit:
            return 'No file found after %d iterations.' % limit
        ct += 1
        try:
            dirs = next(os.walk(top))[1]
        except StopIteration:  # access denied and other exceptions
            top = _top
            continue
        i = random.randint(0, len(dirs))
        if i == 0:  # use .
            files = glob.glob(top + '/*.' + ext)
            if not files:
                top = _top
                continue
            i = random.randint(0, len(files)-1)
            return files[i]
        top += '/' + dirs[i-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for visualizing objecness score of input image being '
                                                 'detected by YOLOv5 and RetinaNet at every possible position '
                                                 'by 10 pixels at background')
    parser.add_argument('--job', type=int, help='ID of instance to process')
    parser.add_argument('--yolo_weights', type=str, default='yolov5s.pt',
                        help='Path to YOLOv5 model (e.g. yolov5s(6).pt, yolov5m(6).pt, yolov5l(6).pt,yolov5x(6).pt)')
    parser.add_argument('--retina_weights', type=str, default='coco_resnet.py', help='Path to RetinaNet model')
    parser.add_argument('--output', type=str,
                        default=os.path.join('/', os.sep, 'mnt', 'datagrid', 'personal', 'kolarj55', 'shifting_object'),
                        help='Path to output folder')
    parser.add_argument('--chunks_size', type=int, default=1000,
                        help='Number of jobs to split the process from 20 000 jobs (e.g. 1000 -> 20 000/1000 = 20 chunks of work')
    parser.add_argument('--yolo_score_tres', type=float, default=0.25, help='Score threshold for YOLOv5 detector')
    parser.add_argument('--retina_score_tres', type=float, default=0.45, help='Score threshold for RetinaNet detector')

    # places_path = os.path.join('/', os.sep, 'mnt', 'datagrid', 'public_datasets', 'places365', 'val_256')
    # bg_path = random.choice(os.listdir(places_path))

    ### Driver's code ###
    chunk_size = 1000
    yolo_score_tres = 0.25
    retina_score_tres = 0.5
    iou_tres = 0.45

    yolo_weights = 'yolov5s.pt'  # other options: yolo5m6 (medium), yolo5l6 (large), yolo5x6 (xlarge)
    retina_weights = 'coco_resnet.pt'
    target_dir = os.path.join('/', os.sep,
                              'mnt',
                              'datagrid',
                              'personal',
                              'kolarj55',
                              'shifting_object')
    human_path = 'humans/human1.png'
    bg_path = 'bgs/park1.jpg'
    ### End of driver code ###

    mpl.use('module://backend_interagg')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename='debug.log')
    logger = logging.getLogger('__name__')
    t0 = time.time()
    logging.debug('Script is starting')

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    human_matrix, bg_matrix, alpha_mask = get_global_data(human_path, bg_path)
    h, w, _ = human_matrix.shape
    bg_h, bg_w, _ = bg_matrix.shape
    yolo_score_matrix = np.zeros((bg_h, bg_w))
    retina_score_matrix = np.zeros((bg_h, bg_w))

    logging.info(f'Using object {human_path}')
    logging.info(f'Using background {bg_path}')
    logging.info(
        f'Chunk size: {chunk_size}, Yolo Score Threshold: {yolo_score_tres}, RetinaNet Score Threshold: {retina_score_tres}, IoU Threshold: {iou_tres}')
    logging.info(f'Weights used: {yolo_weights} {retina_weights}')
    logging.debug('Global variables prepared')

    process_images()
