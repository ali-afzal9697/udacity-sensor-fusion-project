# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
#

# General package imports
import easydict
import numpy as np
import torch
from typing import List

# Add project directory to PYTHONPATH to enable relative imports
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Model-related imports
from tools.object_detection_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.object_detection_models.darknet.utils.evaluation_utils import post_processing_v2
from tools.object_detection_models.resnet.models import fpn_resnet
from tools.object_detection_models.resnet.utils.evaluation_utils import decode, post_processing
from tools.object_detection_models.resnet.utils.torch_utils import _sigmoid


def load_configs_model(model_name: str = 'darknet', configs: easydict.EasyDict = None) -> easydict.EasyDict:
    """Load model configurations into an EasyDict instance.

    Args:
        model_name (str): The desired pre-trained model to load, can be one of: ['darknet', 'fpn_resnet'].
        configs (easydict.EasyDict): The EasyDict instance to update.

    Returns:
        easydict.EasyDict: The updated EasyDict instance containing the model parameters to use.
    """
    if not configs:
        configs = easydict.EasyDict()

    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))

    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'object_detection_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.saved_fn = 'darknet'
        configs.rel_results_folder = 'results_sequence_1_darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.min_iou = 0.5
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False
        configs.save_test_output = True
    elif model_name == 'fpn_resnet':
        print("student task ID_S3_EX1-3")
        configs.model_path = os.path.join(parent_path, 'tools', 'object_detection_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet'
        configs.saved_fn = 'fpn-resnet'
        configs.rel_results_folder = 'results_sequence_1_resnet'
        configs.pretrained_path = configs.pretrained_filename
        configs.num_layers = 18
        configs.K = 50
        configs.no_cuda = False
        configs.gpu_idx = 0
        configs.num_samples = None
        configs.num_workers = 1
        configs.batch_size = 1
        configs.nms_thresh = 0.4
        configs.peak_thresh = 0.2
        configs.conf_thresh = 0.5
        configs.min_iou = 0.5
        configs.save_test_output = False
        configs.output_format = 'image'
        configs.output_video_fn = 'out_fpn_resnet'
        configs.output_width = 608
        configs.pin_memory = True
        configs.distributed = False
        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50
        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2
        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
        configs.num_input_features = 4
    else:
        raise ValueError(f"Error: Invalid model name '{model_name}'")

    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset')
    if configs.save_test_output:
        configs.result_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn, configs.rel_results_folder)

    configs.no_cuda = True
    configs.gpu_idx = 0
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    return configs


def load_configs(model_name: str = 'fpn_resnet', configs: easydict.EasyDict = None) -> easydict.EasyDict:
    """Returns the modified EasyDict instance with model parameters.

    Args:
        model_name (str): The model to load the configuration settings for.
        configs (easydict.EasyDict): The EasyDict instance to store the configuration settings.

    Returns:
        easydict.EasyDict: The configured EasyDict instance.
    """
    if not configs:
        configs = easydict.EasyDict()

    configs.lim_x = [0, 50]
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0]
    configs.bev_width = 608
    configs.bev_height = 608

    configs = load_configs_model(model_name, configs)

    configs.output_width = 608
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
    return configs


def create_model(configs: easydict.EasyDict) -> torch.nn.Module:
    """Returns a torch.nn.Module instance from the specification in configs.

    Args:
        configs (easydict.EasyDict): The EasyDict instance specifying the object detection neural network architecture to build.

    Returns:
        torch.nn.Module: The instantiated neural network submodule configured with the torch.nn.Module base class.
    """
    exists = os.path.isfile(configs.pretrained_filename)
    assert exists, f"No file at {configs.pretrained_filename}"

    if configs.arch == 'darknet' and configs.cfgfile:
        print('Using the DarkNet model architecture')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)
    elif 'fpn_resnet' in configs.arch:
        print('Using the ResNet model architecture with Feature Pyramid')
        model = fpn_resnet.get_pose_net(num_layers=configs.num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)
        print("student task ID_S3_EX1-4")
    else:
        assert False, f"Undefined model backbone: '{configs.arch}'"

    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print(f"Loaded weights from '{configs.pretrained_filename}'\n")

    configs.device = torch.device('cpu' if configs.no_cuda else f"cuda:{configs.gpu_idx}")
    model = model.to(device=configs.device)
    model.eval()
    return model


def detect_objects(input_bev_maps: List[np.ndarray], model: torch.nn.Module, configs: easydict.EasyDict) -> List[list]:
    """Perform inference and post-process the object detections.

    Args:
        input_bev_maps (List[np.ndarray]): The BEV images to perform inference over, i.e., the images containing objects to detect.
        model (torch.nn.Module): The pre-trained object detection net as a torch.nn.Module instance.
        configs (easydict.EasyDict): The EasyDict instance containing the confidence and NMS threshold values and a path to the pre-trained model.

    Returns:
        List[list]: The nested list of predictions, where each prediction has the form: [id, x, y, z, h, w, l, yaw].
    """
    with torch.no_grad():
        outputs = model(input_bev_maps)

        if 'darknet' in configs.arch:
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            print(output_post)
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])
        elif 'fpn_resnet' in configs.arch:
            print("student task ID_S3_EX1-5")
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(hm_cen=outputs['hm_cen'], cen_offset=outputs['cen_offset'],
                                direction=outputs['direction'], z_coor=outputs['z_coor'], dim=outputs['dim'],
                                K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections=detections, configs=configs)
            detections = detections[0][1]

    print("student task ID_S3_EX2")
    objects = []

    if not detections.any():
        return objects

    for obj in detections:
        _id, _x, _y, _z, _h, _w, _l, _yaw = obj
        x = _y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
        y = _x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
        y -= (configs.lim_y[1] - configs.lim_y[0]) / 2
        w = _w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
        l = _l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
        z = _z
        yaw = _yaw
        h = _h

        if ((x >= configs.lim_x[0] and x <= configs.lim_x[1]) and (
                y >= configs.lim_y[0] and y <= configs.lim_y[1]) and (z >= configs.lim_z[0] and z <= configs.lim_z[1])):
            objects.append([1, x, y, z, h, w, l, yaw])

    return objects