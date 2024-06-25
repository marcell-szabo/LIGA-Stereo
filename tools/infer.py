import argparse
from liga.config import cfg, cfg_from_list, cfg_from_yaml_file, update_cfg_by_args, log_config_to_file
from pathlib import Path
import numpy as np
from liga.utils import common_utils
from liga.models import load_data_to_gpu, build_network
from liga.datasets import build_dataloader
import torch
from flask import Flask, request, jsonify
import os
import cv2
import time
from liga.utils import box_utils
from liga.ops.iou3d_nms import iou3d_nms_utils
from liga.utils.open3d_utils import save_point_cloud, save_box_corners
from liga.visualization.bev import BEVVisualizer
from mmdetection_kitti.mmdet.utils.det3d.kitti_utils import boxes3d_to_bev_torch
import copy

app = Flask(__name__)
upload_folder = os.path.join('data', 'kitti', 'training')
app.config['UPLOAD'] = upload_folder

model, test_set, test_loader = None, None, None

def load_model():
    global model, test_loader, test_set
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='pytorch')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to evaluate')
    parser.add_argument('--ckpt_id', type=int, default=None, help='checkpoint id to evaluate')
    parser.add_argument('--exp_name', type=str, default=None, help='exp path for this experiment')
    parser.add_argument('--trainval', action='store_true', default=False, help='')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    update_cfg_by_args(cfg, args)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '_'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    assert args.ckpt or args.ckpt_id, "pls specify ckpt or ckpt_dir or ckpt_id"
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    logger = common_utils.create_logger(rank=cfg.LOCAL_RANK)
    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        training=False,
        dist=dist_test, workers=args.workers, logger=logger
    )
    print(f'testset {test_set}')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    model.eval()
    return model


def infer(model, test_loader, start_time):
    for  batch_dict in test_loader:
        load_data_start_time = time.time_ns()
        print(f'start infer: {(load_data_start_time - start_time) / 10**9}')

        load_data_to_gpu(batch_dict)
        print(f'load data gpu time: {(time.time_ns() - load_data_start_time) / 10**9}')
        infer_start_time = time.time_ns()
        with torch.no_grad():
            print(batch_dict)
            pred_dicts, ret_dict = model(batch_dict)
        infer_end_time = time.time_ns()
        print(f'infer time: {(infer_end_time - infer_start_time) / 10**9}')
        disp_dict = {}

        transform_start_time = time.time_ns()
        calib = batch_dict['calib'][0]
        pred_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(pred_dicts[0]['pred_boxes'].cpu().numpy(), calib)
        # gt_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(batch_dict['gt_boxes'][0,:,:7].cpu().numpy(), calib)
        pred_box_corners = box_utils.boxes3d_to_corners3d_kitti_camera(pred_boxes_cam)
        # gt_box_corners = box_utils.boxes3d_to_corners3d_kitti_camera(gt_boxes_cam)

        try:
            # points = batch_dict['points'][:, 1:].cpu().numpy()
            # save_point_cloud(points, 'temp/pc.ply')
            # gt_lineset = save_box_corners(gt_box_corners, 'temp/gt_box.ply')
            
            # pred_lineset = save_box_corners(pred_box_corners, 'temp/pred_box.ply')

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            # img = (img.astype(np.float32) / 255 - mean) / std
            img = batch_dict['left_img'][0].permute(1, 2, 0).cpu().numpy().copy()
            #img = batch_dict['left_img'][0].cpu().numpy()
            # Convert img to Mat format
            img = (((img * std) + mean) * 255).astype(np.uint8)

            # gt_boxes_img, gt_box_corners_img = calib.corners3d_to_img_boxes(gt_box_corners)
            pred_boxes_img, pred_box_corners_img = calib.corners3d_to_img_boxes(pred_box_corners)


            def draw_image_3d_rect(img, corners_img, color):
                edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7)]
                for edge in edges:
                    pt1 = tuple(map(int, corners_img[edge[0]]))
                    pt2 = tuple(map(int, corners_img[edge[1]]))
                    cv2.line(img, pt1, pt2, color=color, thickness=2)

            # for i in gt_box_corners_img:
            #     draw_image_3d_rect(img, i, (0, 255, 0))

            for i in pred_box_corners_img:
                draw_image_3d_rect(img, i, (255, 0, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f'transorm time: {(time.time_ns() - transform_start_time) / 10**9}')
            return img
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(f'nogood: {e}')


def infer2(model, start_time):
    calib = test_set.get_calib(0)
    left_img = test_set.get_image(0, 2)
    right_img = test_set.get_image(0, 3)
    batch_dict = {
            'frame_id': 0,
            'calib': calib,
            'calib_ori': copy.deepcopy(calib),
            'left_img': left_img,
            'right_img': right_img,
            'image_shape': left_img.shape,
        }
    batch_dict = test_set.collate_batch([batch_dict])
    load_data_start_time = time.time_ns()
    print(f'start infer: {(load_data_start_time - start_time) / 10**9}')

    load_data_to_gpu(batch_dict)
    print(f'load data gpu time: {(time.time_ns() - load_data_start_time) / 10**9}')
    infer_start_time = time.time_ns()
    with torch.no_grad():
        pred_dicts, ret_dict = model(batch_dict)
    infer_end_time = time.time_ns()
    print(f'infer time: {(infer_end_time - infer_start_time) / 10**9}')
    disp_dict = {}
    for i in range(len(pred_dicts)):
        print(f'{i}. dict {pred_dicts[i]["pred_boxes_2d"]}')
        print(f'{i}. dict {pred_dicts[i]["pred_scores_2d"]}')

    transform_start_time = time.time_ns()
    calib = batch_dict['calib'][0]
    # bev_boxes = box_utils.boxes3d_lidar_to_aligned_bev_boxes(pred_dicts[0]['pred_boxes'].cpu())
    # print(bev_boxes)
    pred_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(pred_dicts[0]['pred_boxes'].cpu().numpy(), calib)
    pred_box_corners = box_utils.boxes3d_to_corners3d_kitti_camera(pred_boxes_cam)
    
    bev_boxes = boxes3d_to_bev_torch(torch.from_numpy(pred_boxes_cam))
    # print(bev_boxes)

    # ground truths
    # gt_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(batch_dict['gt_boxes'][0,:,:7].cpu().numpy(), calib)
    # gt_box_corners = box_utils.boxes3d_to_corners3d_kitti_camera(gt_boxes_cam)

    visualizer = BEVVisualizer(fig_cfg={'figsize': (5,5)})
    # set bev image in visualizer
    visualizer.set_bev_image()
    # draw bev bboxes
    visualizer.draw_bev_bboxes(pred_boxes_cam, 'feed/bev/bev.png', edge_colors='orange')
    try:
        # points = batch_dict['points'][:, 1:].cpu().numpy()
        # save_point_cloud(points, 'temp/pc.ply')
        # gt_lineset = save_box_corners(gt_box_corners, 'temp/gt_box.ply')
        
        # pred_lineset = save_box_corners(pred_box_corners, 'temp/pred_box.ply')

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # img = (img.astype(np.float32) / 255 - mean) / std
        img = batch_dict['left_img'][0].permute(1, 2, 0).cpu().numpy().copy()
        #img = batch_dict['left_img'][0].cpu().numpy()
        # Convert img to Mat format
        img = (((img * std) + mean) * 255).astype(np.uint8)

        # gt_boxes_img, gt_box_corners_img = calib.corners3d_to_img_boxes(gt_box_corners)
        pred_boxes_img, pred_box_corners_img = calib.corners3d_to_img_boxes(pred_box_corners)


        def draw_image_3d_rect(img, corners_img, color):
            edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)]
            for edge in edges:
                pt1 = tuple(map(int, corners_img[edge[0]]))
                pt2 = tuple(map(int, corners_img[edge[1]]))
                cv2.line(img, pt1, pt2, color=color, thickness=2)

        # for i in gt_box_corners_img:
        #     draw_image_3d_rect(img, i, (0, 255, 0))

        for i in pred_box_corners_img:
            draw_image_3d_rect(img, i, (255, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(f'transorm time: {(time.time_ns() - transform_start_time) / 10**9}')
        return img
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f'nogood: {e}')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model
        image_l = request.files['image_l']
        image_r = request.files['image_r']
        calib = request.files['calib']
        
        saving_file_start_time = time.time_ns()
        
        image_l.save(dst=os.path.join(upload_folder, 'image_2', '0.png'))
        image_r.save(dst=os.path.join(upload_folder, 'image_3', '0.png'))
        calib.save(dst=os.path.join(upload_folder, 'calib', '0.txt'))
        
        print(f'file saving time: {(time.time_ns() - saving_file_start_time) / 10**9}')
        saving_file_start_time = time.time_ns()

        # result_img = infer(model, test_loader, saving_file_start_time)
        result_img = infer2(model, saving_file_start_time)

        print(f'full infer function time: {(time.time_ns() - saving_file_start_time) / 10**9}')
        saving_file_start_time = time.time_ns()

        cv2.imwrite('feed/images/result.png', result_img.astype(np.uint8))
        print(f'result saving time: {(time.time_ns() - saving_file_start_time) / 10**9}')

        return jsonify({'message': 'Prediction completed successfully'})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello'})

if __name__ == '__main__':
    load_model()
    print('init: start serving')
    app.run(host='0.0.0.0', port=5000)
