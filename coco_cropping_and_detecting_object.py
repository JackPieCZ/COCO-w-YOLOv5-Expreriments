import os
import time
import argparse
import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.patches import Rectangle

from PIL import Image
from pycocotools.coco import COCO
from yolo.detect_without_nms_without_filesv2 import run as yolodetect


# from retinanet.visualize_single_image_without_nms_without_files import detect_image as retinadetect


def tprint(text):
    """
    Prints given text with timestamp
    Args:
        text: text that should be printed
    Returns None
    """
    clock = datetime.timedelta(seconds=time.time() - t0)
    print(f'[{clock}] {text}')


def load_image(file_path):
    """
    Loads image from file_path input into numpy array (int16)
    Args:
        file_path: path to image that should be loaded
    Returns numpy array
    """
    arr = np.asarray(Image.open(file_path), np.int16)
    if arr.ndim == 2: # if the picture is in greyscale, convert to rgb
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def show_image(array, label=None, bbox=None):
    """
    Displays given array with matplotlib
    Args:
        array: numpy array
    Optional:
        label - label text above the image
        bbox - shows rectangle of given bounding box
    Returns None
    """
    ax = plt.subplot()
    if bbox:
        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, ec='red'))
    ax.imshow(array)
    ax.set_title(label)
    plt.show()


def get_img_info(img_index, catId, imgsIdxs, show=False):
    """
    Gets needed information for given image (id, filepath, size, bounding box)
    Args:
        img_index: image index
        catId: category of given image
        imgsIdxs: list of image indexes
    Optional:
        show: show image with bounding box with matplotlib
    Returns image id, filepath, size, bounding box coordinates
    If the bounding box is smaller than 80px returns None
    """
    img = coco.loadImgs(imgsIdxs[img_index])[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catId)
    anns = coco.loadAnns(annIds)
    bboxes = []
    for i in anns:
        if max(i['bbox'][2], i['bbox'][3]) > 80:
            bboxes.append([round(num) for num in i['bbox']])

    if bboxes:
        img_path = os.path.join(imgDir, img['file_name'])
        img_size = (img['width'], img['height'])
        img_id = img['id']
        if show:
            _, ax = plt.subplots()
            for x, y, w, h in bboxes:
                ax.add_patch(Rectangle((x, y), w, h, fill=False, ec='blue'))
            I = load_image(img_path)
            ax.imshow(I)
            plt.show()
        plt.close('all')
        return img_id, img_path, img_size, bboxes
    else:
        return


def intersection(bb1, bb2):
    """
    Compares the intersection of two bounding boxes
    Args:
        bb1: the first bounding box (x,y,w,h)
        bb2: the other bounding box (x,y,w,h)
    Returns float value of intersection between bb1 and bb2
    """
    bb1_x1 = bb1[0]
    bb1_x2 = bb1_x1 + bb1[2]
    bb1_y1 = bb1[1]
    bb1_y2 = bb1_y1 + bb1[3]

    bb2_x1 = bb2[0]
    bb2_x2 = bb2_x1 + bb2[2]
    bb2_y1 = bb2[1]
    bb2_y2 = bb2_y1 + bb2[3]

    dx = min(bb1_x2, bb2_x2) - max(bb1_x1, bb2_x1)
    dy = min(bb1_y2, bb2_y2) - max(bb1_y1, bb2_y1)

    if dx >= 0 and dy >= 0:
        return dx * dy
    else:
        return 0.0


def bb_iou(bb1, bb2):
    """
    Compute intersection over union between two bounding boxes
    Args:
        bb1: bounding box {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
        bb2: bounding box {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
    Returns float value of IoU
    """
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
    assert 0.0 <= iou <= 1.0
    return iou


def show_safe_bboxes(bboxes, img_array, img_size):
    """
    Displays surrounding bounding boxes for given image
    Args:
        bboxes: a list of bounding boxes (x,y,w,h)
        img_array: numpy array of image
        img_size: size of the image in pixels (w,h)
    Returns None
    """
    ax = plt.subplot()
    ax.imshow(img_array)
    for x, y, w, h in bboxes:
        desired_width = h / img_size[1] * img_size[0]
        left_safe_box = [max(x - desired_width, 0), y, x - max(x - desired_width, 0), h]
        right_safe_box = [x + w, y, min(desired_width, img_size[0] - (x + w)), h]
        ax.add_patch(
            Rectangle((left_safe_box[:2]), left_safe_box[2], left_safe_box[3], fill=False, ec='orange'))
        ax.add_patch(
            Rectangle((right_safe_box[:2]), right_safe_box[2], right_safe_box[3], fill=False, ec='yellow'))
    plt.show()
    plt.close('all')


def generate_safe_bboxes(bbox, img_size):
    """
    Computes coordinates for surrounding bounding boxes of given bounding box
    Args:
        bbox: bounding box that is in the center (x,y,w,h)
        img_size: size of image where the bounding box is defined (w,h)
    Returns list of coordinates for both left and right bounding boxes
    """
    x, y, w, h = bbox
    desired_width = h / img_size[1] * img_size[0]
    left_safe_box = [max(x - desired_width, 0), y, x - max(x - desired_width, 0), h]
    right_safe_box = [x + w, y, min(desired_width, img_size[0] - (x + w)), h]
    return left_safe_box, right_safe_box


def generate_step_frames(num_steps, bbox, img_size, img_array, show=False, img_step=5):
    """
    Creates list of image arrays for differently cropped amounts (e.g. steps)
    Args:
        num_steps: number of different crops
        bbox: coordinates of bounding box in original image (x,y,w,h)
        img_size: size of original image (w,h)
        img_array: numpy array of original image
    Optional:
        show: display the process in matplotlib
        img_step: how many steps should be skipped for displaying
    Returns
    """

    img_aspect_ratio = img_size[0] / img_size[1]
    x, y, w, h = bbox
    bbox_area = w * h
    bbox_aspect_ratio = w / h
    frames = []
    bboxes = []
    bbox_sizes = []
    if img_aspect_ratio > bbox_aspect_ratio:
        bg_h = h
        bg_w = round(img_size[0] / img_size[1] * h)
        rest_x = img_size[0] - x - w
        moving_pixels = bg_w + w
        step = moving_pixels / num_steps
        border = 0

        up_border_bg_start = 0
        down_border_bg_end = h
        up_border_pic_start = y
        down_border_pic_end = y + h
    else:
        bg_h = round(img_size[1] / img_size[0] * w)
        bg_w = w
        rest_x = img_size[0] - w - x
        rest_y = img_size[1] - h - y
        moving_pixels = w * 2
        step = moving_pixels / num_steps
        border = (bg_h - h) // 2

        if y > border:
            up_border_pic_start = y - border
            up_border_bg_start = 0
        else:
            up_border_pic_start = 0
            up_border_bg_start = border - y
        if rest_y > border:
            down_border_pic_end = up_border_pic_start + bg_h - up_border_bg_start
            down_border_bg_end = bg_h
        else:
            down_border_pic_end = img_size[1]
            down_border_bg_end = border + h + rest_y

    ### FIRST FRAME ###
    frame0 = np.zeros((bg_h, bg_w, 3), dtype=np.int16)
    start_bg_0 = 0
    end_bg_0 = min(bg_w, rest_x)
    start_pic_0 = x + w
    end_pic_0 = min(x + w + rest_x, x + w + bg_w)
    frame0[up_border_bg_start:down_border_bg_end, start_bg_0:end_bg_0, :] = img_array[
                                                                            up_border_pic_start:down_border_pic_end,
                                                                            start_pic_0:end_pic_0, :]
    frames.append(frame0)
    bbox = [0, 0, 0, 0]
    bboxes.append(bbox)
    bbox_sizes.append(-100)
    if show:
        show_image(frame0, label='Frame 0', bbox=bbox)

    ### FROM SECOND FRAME TO MIDDLE FRAME ###
    for i in range(1, num_steps):# // 2 + 1):
        frame = np.zeros((bg_h, bg_w, 3), dtype=np.int16)
        start_pic = round(x + w - i * step)
        if start_pic < 0:
            end_pic = bg_w + start_pic
            start_pic = 0
        else:
            end_pic = min(start_pic + bg_w, img_size[0])
        if start_pic > 0:
            start_bg = 0
            end_bg = min(bg_w, end_pic - start_pic)
        else:
            start_bg = abs(round(x + w - i * step))
            end_bg = min(bg_w, start_bg + end_pic)
        frame[up_border_bg_start:down_border_bg_end, start_bg:end_bg, :] = img_array[
                                                                           up_border_pic_start:down_border_pic_end,
                                                                           start_pic:end_pic, :]
        # end_bbox = round(i * step)
        # if end_bbox < w:
        #     start_bbox = 0
        # elif end_bbox > bg_w:
        #     end_bbox = bg_w
        #     start_bbox = x - start_pic + start_bg
        # else:
        #     start_bbox = end_bbox - w
        start_bbox = x - start_pic + start_bg
        end_bbox = start_bbox + w
        if start_bbox < 0:
            start_bbox = 0
        if end_bbox > bg_w:
            end_bbox = bg_w
        bbox = [start_bbox, border, end_bbox - start_bbox, bg_h - border * 2]
        used_bbox_size = bbox[2] * bbox[3]
        percentage_of_bbox = used_bbox_size // (bbox_area / 100)
        if i < (num_steps // 2 + 1):
            bbox_size = (int(round(-100 + percentage_of_bbox, -1)))
        else:
            bbox_size = (int(round(100 - percentage_of_bbox, -1)))
        if i % img_step == 0 and show:
            show_image(frame, f'Frame {i}', bbox=bbox)
        if bbox_sizes[-1] != bbox_size:
            frames.append(frame)
            bboxes.append(bbox)
            bbox_sizes.append(bbox_size)
        # else:
        #     if i == (num_steps // 2):
        #         frames.append(frame)
        #         bboxes.append(bbox)
        #         bbox_sizes.append(bbox_size)
    # ### MIDDLE FRAME ###
    #
    # ### FROM MIDDLE TO LAST FRAME ###
    # for i in range(num_steps // 2 + 1, num_steps):
    #     frame = np.zeros((bg_h, bg_w, 3), dtype=np.int16)
    #     start_pic = max(round(start_pic_0 - i * step), 0)
    #     end_pic = min(img_size[0], round(x + w + bg_w - i * step))
    #     start_bg = bg_w - (end_pic - start_pic)
    #     end_bg = bg_w
    #     if start_bg < 0:
    #         end_pic += start_bg
    #         start_bg = 0
    #     frame[up_border_bg_start:down_border_bg_end, start_bg:end_bg, :] = img_array[
    #                                                                        up_border_pic_start:down_border_pic_end,
    #                                                                        start_pic:end_pic, :]
    #     start_bbox = x - start_pic + start_bg
    #     end_bbox = start_bbox + w
    #     if end_bbox > bg_w:
    #         end_bbox = bg_w
    #     bbox = [start_bbox, border, end_bbox - start_bbox, bg_h - border * 2]
    #     used_bbox_size = bbox[2] * bbox[3]
    #     percentage_of_bbox = used_bbox_size / (bbox_area / 100)
    #     bbox_size = (int(round(100 - percentage_of_bbox, -1)))
    #     if i % img_step == 0 and show:
    #         show_image(frame, f'Frame {i}', bbox=bbox)
    #
    #     if bbox_sizes[-1] != bbox_size:
    #         frames.append(frame)
    #         bboxes.append(bbox)
    #         bbox_sizes.append(bbox_size)
    #
    ### LAST FRAME ###
    frame21 = np.zeros((bg_h, bg_w, 3), dtype=np.int16)
    start_bg_20 = max(0, bg_w - x)
    end_bg_20 = bg_w
    start_pic_20 = max(0, x - bg_w)
    end_pic_20 = x
    frame21[up_border_bg_start:down_border_bg_end, start_bg_20:end_bg_20, :] = img_array[
                                                                               up_border_pic_start:down_border_pic_end,
                                                                               start_pic_20:end_pic_20, :]
    frames.append(frame21)
    bbox = [0, 0, 0, 0]
    bboxes.append(bbox)
    bbox_sizes.append(100)
    if show:
        show_image(frame21, f'Frame {num_steps}', bbox=bbox)

    plt.close('all')
    return frames, bboxes, bbox_sizes


def detector_process(num_steps, step_detections, step_frames, step_bboxes, show=False, img_step=5):
    ious = []
    posterior_probs = []
    objectnesses = []
    correct_cats = []
    for i in range(num_steps):
        iou = 0
        posterior_prob = 0
        objectness = 0
        correct_cat = False

        if step_detections[i] != []:
            x1, y1, w, h = step_bboxes[i]
            bbox_GT = {'x1': x1, 'x2': x1 + w, 'y1': y1, 'y2': y1 + h}
            correct_cat = True

            if len(step_detections[i]) > 1:
                temp_iou = 0
                det_ind = 0
                for ind, det in enumerate(step_detections[i]):
                    x1, y1, x2, y2 = [int(num) for num in det[:4]]
                    bbox_pred = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                    iou = bb_iou(bbox_pred, bbox_GT)
                    if iou > temp_iou:
                        temp_iou = iou
                        det_ind = ind
                step_detections[i] = [step_detections[i][det_ind]]
            det = step_detections[i][0]

            x1, y1, x2, y2 = [int(num) for num in det[0:4]]
            bbox_pred = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
            iou = bb_iou(bbox_pred, bbox_GT)
            posterior_prob = det[4]
            objectness = det[5]
            if i % img_step == 0 and show:
                ax = plt.subplot()
                ax.imshow(step_frames[i])
                ax.add_patch(Rectangle((bbox_pred['x1'], bbox_pred['y1']), bbox_pred['x2'] - bbox_pred['x1'],
                                       bbox_pred['y2'] - bbox_pred['y1'], ec='blue', fill=False))
                ax.add_patch(Rectangle((bbox_GT['x1'], bbox_GT['y1']), w, h, ec='red', fill=False))
                plt.show()
                if i == 0 or i == num_steps-1:
                    pass

        ious.append(iou)
        posterior_probs.append(posterior_prob)
        objectnesses.append(objectness)
        correct_cats.append(correct_cat)
    plt.close('all')
    return ious, posterior_probs, objectnesses, correct_cats


def mean_step_values(lst, num_steps, filter_zeros):
    mean_results = []
    for i in range(num_steps):
        if filter_zeros and i not in [0, num_steps-1]:
            filtered_lst = list(filter(lambda a: a != 0, lst[i]))
        else:
            filtered_lst = lst[i]
        if len(filtered_lst) > 0:
            mean_result = sum(filtered_lst) / len(filtered_lst)
        else:
            mean_result = None
        mean_results.append(mean_result)
    return mean_results


def save_individual_plot_values(filepath, lst):
    with open(filepath, 'w') as f:
        for el in lst:
            f.write(str(el) + '\n')
        f.close()


def check_if_data(lists):
    desc = ""
    for lst in lists:
        if lst == [None] * len(lst):
            desc += ''


def check_category(cat, yolo_weights, num_steps, debug=False):
    catId = coco.getCatIds(catNms=[cat])
    true_catId = categories.index(cat)
    imgsIdxs = coco.getImgIds(catIds=catId)
    print(f'Starting class {true_catId} - {cat}')

    used_imgs = 0
    x_ax_range = [i for i in range(-100, 101, 10)]
    x_ax_ious = [[] for _ in range(num_steps)]
    x_ax_posterior_probs = [[] for _ in range(num_steps)]
    x_ax_objectnesses = [[] for _ in range(num_steps)]
    x_ax_correct_cats = [[] for _ in range(num_steps)]
    t1 = time.time()

    for img_index in range(0, len(imgsIdxs)):
        tprint(f'{img_index} / {len(imgsIdxs)} - images used {used_imgs}')
        data = get_img_info(img_index, catId, imgsIdxs, show=debug)
        if data:
            img_id, img_path, img_size, bboxes = data
            img_array = load_image(img_path)
            if debug:
                show_safe_bboxes(bboxes, img_array, img_size)

            for bbox in bboxes:
                left_safe_box, right_safe_box = generate_safe_bboxes(bbox, img_size)
                bboxes_to_avoid = bboxes[:bboxes.index(bbox)] + bboxes[bboxes.index(bbox) + 1:]
                if sum(intersection(bbox, x) for x in bboxes_to_avoid) + sum(
                        intersection(left_safe_box, x) for x in bboxes_to_avoid) + sum(
                    intersection(right_safe_box, x) for x in bboxes_to_avoid) == 0.0:
                    step_frames, step_bboxes, bbox_coverage = generate_step_frames(num_steps - 1, bbox, img_size,
                                                                                   img_array,
                                                                                   show=debug, img_step=1)
                    step_detections = yolodetect(weights=yolo_weights, source=step_frames,
                                                 classes=true_catId)  # , conf_thres=0.1, )

                    ious, posterior_probs, objectnesses, correct_cats = detector_process(len(step_frames),
                                                                                         step_detections, step_frames,
                                                                                         step_bboxes, show=debug,
                                                                                         img_step=1)

                    for i in range(len(step_frames)):
                        slot = x_ax_range.index(bbox_coverage[i])
                        x_ax_ious[slot].append(ious[i])
                        x_ax_posterior_probs[slot].append(posterior_probs[i])
                        x_ax_objectnesses[slot].append(objectnesses[i])
                        x_ax_correct_cats[slot].append(correct_cats[i])
                    used_imgs += 1

    additional_desc = ''
    mean_ious = mean_step_values(x_ax_ious, num_steps, filter_zeros=True)
    if mean_ious == [None] * len(mean_ious):
        additional_desc += 'IoU Data Not Available\n'

    mean_posterior_probs = mean_step_values(x_ax_posterior_probs, num_steps, filter_zeros=True)
    if mean_posterior_probs == [None] * len(mean_posterior_probs):
        additional_desc += 'Posterior Probability Data Not Available\n'

    mean_objectnesses = mean_step_values(x_ax_objectnesses, num_steps, filter_zeros=True)
    if mean_objectnesses == [None] * len(mean_objectnesses):
        additional_desc += 'Objectness Data Not Available\n'

    mean_correct_cats = mean_step_values(x_ax_correct_cats, num_steps, filter_zeros=False)

    ax = plt.subplot()
    possible_color_palette = ["1f77b4", "ff7f0e", "561d25", "aaf683", "941b0c"]
    ax.plot(x_ax_range, mean_ious, '-o', label='IoU')  # 1F77B4
    ax.plot(x_ax_range, mean_posterior_probs, '-d', label='~p(K|x)')  # FF7F0E
    ax.plot(x_ax_range, mean_objectnesses, '-h', label='Objectness')  # 2CA02C
    ax.plot(x_ax_range, mean_correct_cats, '-D', label='Detection of Correct Class')  # D62728
    plt.text(0.5, 0.5, additional_desc, horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, alpha=0.3)
    plt.xlabel('% of the Picture Cropped Out')
    plt.ylabel('Values')
    plt.title(f'Average Detector Values of {used_imgs} {cat} Images')
    plt.xlim([-100, 100])
    plt.ylim([0, 1])
    ax.xaxis.set_minor_locator(plticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(plticker.AutoMinorLocator())
    loc_y = plticker.MultipleLocator(base=0.1)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc_y)
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plot_path = os.path.join(target_dir, 'bash_jobs', f'job_{true_catId}',
                             f'{true_catId}_{cat}_category_plot.png')
    plt.savefig(plot_path, dpi=600)
    ious_plot_path = os.path.join(target_dir, 'bash_jobs', f'job_{true_catId}',
                                  f'{true_catId}_{cat}_category_plot_ious.txt')
    posterior_probs_plot_path = os.path.join(target_dir, 'bash_jobs', f'job_{true_catId}',
                                  f'{true_catId}_{cat}_category_plot_posterior_pros.txt')
    objectnesses_plot_path = os.path.join(target_dir, 'bash_jobs', f'job_{true_catId}',
                                          f'{true_catId}_{cat}_category_plot_objectnesses.txt')
    correct_cat_plot_path = os.path.join(target_dir, 'bash_jobs', f'job_{true_catId}',
                                         f'{true_catId}_{cat}_category_plot_correct_classes.txt')
    save_individual_plot_values(ious_plot_path, mean_ious)
    save_individual_plot_values(posterior_probs_plot_path, mean_posterior_probs)
    save_individual_plot_values(objectnesses_plot_path, mean_objectnesses)
    save_individual_plot_values(correct_cat_plot_path, mean_correct_cats)

    tprint(f'Category plot completed in {time.time() - t1}')
    plt.show()
    plt.close('all')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=int, help="Coco's ID of the category to be processed")
    parser.add_argument('--num_steps', type=int, default=21, help='Number of steps/crops to make from the input image')
    parser.add_argument('--yolo_weights', type=str, default='yolov5s.pt',
                        help='Select YOLOv5 model file (e.g. yolov5s(6).pt, yolov5m(6).pt, yolov5l(6).pt,yolov5x(6).pt)')
    parser.add_argument('--debug', default=False)
    opt = parser.parse_args()
    return opt


def main(category=None,
         num_steps=21,
         yolo_weights='yolov5s.pt',
         debug=False):
    if category is not None:
        assert 0 <= category <= 79
        assert num_steps >= 3
        print(f'Executing category {category}')
        check_category(categories[category],
                       os.path.join(os.getcwd(), 'models', yolo_weights),
                       num_steps, debug)
    else:
        print('No category assigned. Starting from category 0...')
        for i in range(0, 80):
            check_category(categories[i], yolo_weights, num_steps, debug)


if __name__ == '__main__':
    opt = parse_opt()
    mpl.use('module://backend_interagg')

    annFile = os.path.join('/', os.sep, 'mnt', 'datagrid', 'public_datasets', 'COCO', 'annotations',
                           'instances_val2017.json')
    imgDir = os.path.join('/', os.sep, 'mnt', 'datagrid', 'public_datasets', 'COCO', 'val2017')
    target_dir = os.path.join('/', os.sep, 'mnt', 'home.stud', 'kolarj55', 'detector_improve_iopainting')
    paths = [annFile, imgDir, target_dir]
    for path in paths:
        if not os.path.exists(path):
            print(f"{path} does not exist.")
            exit(1)

    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    categories = [cat['name'] for cat in cats]
    supercategories = set([cat['supercategory'] for cat in cats])

    t0 = time.time()
    # main(**vars(opt))
    main()
