# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
#

# General package imports
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
from shapely.geometry import Polygon
import sys
from typing import List

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Object detection tools and helper functions
import misc.object_detection_tools as tools

def measure_detection_performance(detections: List[list], labels: List[object], labels_valid: List[object], min_iou: float = 0.5) -> List[List[float]]:
    """Returns the computed detection performance measures.

    Args:
        detections (List[list]): Nested list of detections, each with the form: `[id, x, y, z, h, w, l, yaw]`.
        labels (List[object]): List of `label` instances, each with a `box` attribute containing the 3D dimensions and heading angle of the predicted bounding box.
        labels_valid (List[object]): List of `label` instances, each with a `box` attribute containing the 3D dimensions and heading angle of the ground-truth bounding box.
        min_iou (float): The minimum IoU threshold to use for determining matches.

    Returns:
        List[List[float]]: A nested list of detection metrics computed for each pair of predicted and ground-truth bounding boxes.
    """
    true_positives = 0
    center_devs = []
    ious = []

    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:
            box = label.box
            box_1 = tools.compute_box_corners(box.center_x, box.center_y, box.width, box.length, box.heading)
            for det in detections:
                _id, x, y, z, _h, w, l, yaw = det
                box_2 = tools.compute_box_corners(x, y, w, l, yaw)
                dist_x = np.array(box.center_x - x).item()
                dist_y = np.array(box.center_y - y).item()
                dist_z = np.array(box.center_z - z).item()
                try:
                    poly_1 = Polygon(box_1)
                    poly_2 = Polygon(box_2)
                    intersection = poly_1.intersection(poly_2).area
                    union = poly_1.union(poly_2).area
                    iou = intersection / union
                except Exception as err:
                    print(f"Encountered '{err}' error in IoU calculation")
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                    true_positives += 1
        if matches_lab_det:
            best_match = max(matches_lab_det, key=itemgetter(0))
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    all_positives = sum(labels_valid)
    false_negatives = all_positives - true_positives
    false_positives = len(detections) - true_positives
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    return det_performance

def compute_performance_stats(det_performance_all: List[list]):
    """Computes and visualises the evaluation metrics given the detection scores.

    Args:
        det_performance_all (List[list]): The nested list of detection scores, assumed to be computed with `measure_detection_performance`.
    """
    ious = []
    center_devs = []
    pos_negs = []

    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])

    try:
        sum_all_pos, sum_tp, sum_fn, sum_fp = np.asarray(pos_negs).sum(axis=0)
    except TypeError:
        sum_all_pos, sum_tp, sum_fn, sum_fp = 0, 0, 0, 0

    precision = sum_tp / float(sum_tp + sum_fp)
    recall = sum_tp / float(sum_tp + sum_fn)
    print(f"Precision: {precision}, Recall: {recall}")

    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tupl in center_devs:
        for elem in tupl:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)

    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)
    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)
    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)
    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)

    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = [
        'Detection Precision', 'Detection Recall',
        'Intersection over Union (IoU)', 'Position Errors in $X$',
        'Position Errors in $Y$', 'Position Errors in $Z$'
    ]
    textboxes = [
        '', '', '', '\n'.join(
            (r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ),
             r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ),
             r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))), '\n'.join(
                 (r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ),
                  r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ),
                  r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))), '\n'.join(
                      (r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ),
                       r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ),
                       r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))
    ]
    f, a = plt.subplots(2, 3, figsize=(24, 20))
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx], fontsize=20)
        if textboxes[idx]:
            ax.text(0.05,
                    0.95,
                    textboxes[idx],
                    transform=ax.transAxes,
                    fontsize=16,
                    verticalalignment='top',
                    bbox=props)
    plt.tight_layout()
    if matplotlib.rcParams['backend'] != 'agg':
        plt.show()

    DIR_OUT = os.path.join(PACKAGE_PARENT, 'out')
    os.makedirs(DIR_OUT, exist_ok=True)
    fname_out = datetime.now().strftime("%Y-%m-%d-Output-1-Detection-Performance-Metrics.png")
    fp_out = os.path.join(DIR_OUT, fname_out)
    plt.savefig(fp_out)