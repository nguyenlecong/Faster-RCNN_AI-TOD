import os

import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from utils import plot_result, nms, create_folder
from dataloader import CustomDataset, get_transform

from modules import utils


class_map = config.class_map


def infer(model, img):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        pred = model([img.to(device)])

    image = img.numpy().transpose(1, 2, 0)
    image = (image*255).astype('uint8')

    boxes = pred[0]['boxes']
    scores = pred[0]['scores']
    keep = nms(boxes, scores, 0.1)

    labels = pred[0]['labels']
    
    scr_th = 0.2
    for i in keep:
        cls = list(class_map.keys())[int(labels[i])]
        scr = round(float(scores[i]), 2)
        
        if scr >= scr_th:
            plot_result(image, boxes[i], cls, str(scr))

    image = image[:,:,::-1]
    cv2.imwrite('predictions/tmp.png', image)

    # plt.figure(figsize=(20,20))
    # plt.imshow(image[:,:,::-1])

def test(model):
    with torch.no_grad():
        path, _ = create_folder('predictions', False)
        print('Results at', path)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.eval()
        
        for i in tqdm(range(len(test_dataset))):
            img, target = test_dataset[i]

            img_name = target['image_name']
            save_path = os.path.join(path, f'{img_name}.txt')
            file = open(save_path, "w")

            pred = model([img.to(device)])[0]
            
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            zip_lists = zip(labels, scores, boxes)

            r = len(boxes)
            for i, x in enumerate(zip_lists):                
                class_name = list(class_map.keys())[int(x[0])]
                score = round(float(x[1]), 2)

                x, y, x2, y2 = x[2].cpu().numpy().astype("int")
                w = x2 - x
                h = y2 - y

                s = ' '.join(str(i) for i in [class_name, score, x, y, w, h])

                if i<r-1:
                    s += '\n'

                file.writelines(s)
            
            file.close()


if __name__ == '__main__':
    test_batch = config.test_batch
    test_dataset = CustomDataset('/hdd/thaihq/qnet_search/ori_data/test', get_transform(train=False))
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch,
                                                    shuffle=False, num_workers=4,
                                                    collate_fn=utils.collate_fn)
    
    weight_path = config.test_weight_path
    print('Infer with', weight_path)

    model_1 = torch.load(weight_path)

    img, target = test_dataset[2500]
    print(target['image_name'])

    infer(model_1, img)
    test(model_1)