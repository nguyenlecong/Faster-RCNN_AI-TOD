import os

import cv2
import torchvision
from seaborn import color_palette

import config
class_map = config.class_map


def create_folder(root, train_mode):
    max_model = 0
    for root, j, _ in os.walk(root):
        for dirs in j:
            try:
                temp = int(dirs)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1

    log_path = os.path.join(root, str(max_model))
    os.makedirs(log_path)

    if train_mode:
        weight_path = os.path.join(log_path, 'weights')
        os.makedirs(weight_path)
    else:
        weight_path = None

    return log_path, weight_path

def nms(boxes, scores, iou_thresh=0.5):
    return torchvision.ops.nms(boxes, scores, iou_thresh)

def plot_result(image, boxes, cls, scr):
    color_map = make_color_map()
    color = color_map[cls]

    x1, y1, x2, y2 = boxes.cpu().numpy().astype("int")
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    cv2.putText(image, ' '.join([cls, scr]), (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)
    # cv2.putText(image, ' '.join([cls, scr]), (x1, y1-10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1,
    #             cv2.LINE_AA)

def make_color_map():
    '''
        Create a color map for each class
    '''
    names = sorted(set(list(class_map.keys())))
    n = len(names)
    cp = color_palette("Paired", n)

    cp[:] = [tuple(int(255*c) for c in rgb) for rgb in cp]

    return dict(zip(names, cp))
