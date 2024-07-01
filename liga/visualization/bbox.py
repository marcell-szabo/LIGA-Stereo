import torch
import numpy as np
import cv2
from liga.utils.calibration_kitti import Calibration
from typing import Tuple

def draw_image_3d_rect(img, corners_img, color: Tuple[int], alpha: float):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)]
    overlay = img.copy()
    for edge in edges:
        pt1 = tuple(map(int, corners_img[edge[0]]))
        pt2 = tuple(map(int, corners_img[edge[1]]))
        cv2.line(img, pt1, pt2, color=color, thickness=2)
    alpha = float(alpha)
    cv2.addWeighted(overlay, 1 - alpha, img, alpha, 0, img)
    
    # Calculate the top-left corner of the bounding box to place the text
    top_left = tuple(map(int, corners_img[5]))
    
    # Display the prediction score on the image
    text = f"{alpha:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_origin = (top_left[0], top_left[1] - text_size[1] - 2)

    cv2.putText(img, text, text_origin, font, font_scale, (255, 0, 0), font_thickness)


def draw_bbox(img: torch.Tensor, calib: Calibration, box_corners: np.array, pred_scores: np.array, destination: str):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # img = (img.astype(np.float32) / 255 - mean) / std
    # img = batch_dict['left_img'][0].permute(1, 2, 0).cpu().numpy().copy()
    img = img.permute(1,2,0).cpu().numpy().copy()
    # Convert img to Mat format
    img = (((img * std) + mean) * 255).astype(np.uint8)

    # gt_boxes_img, gt_box_corners_img = calib.corners3d_to_img_boxes(gt_box_corners)
    pred_boxes_img, pred_box_corners_img = calib.corners3d_to_img_boxes(box_corners)
    for i, corners in enumerate(pred_box_corners_img):
        score = pred_scores[i]
        draw_image_3d_rect(img, corners, (255, 0, 0), score)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(destination, img.astype(np.uint8))

