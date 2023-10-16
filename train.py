import os

import torch
import torchvision
import torch.utils.data
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


from modules import utils
from modules.engine import train_one_epoch, evaluate

import config
from utils import create_folder
from early_stopping import EarlyStopping
from dataloader import CustomDataset, get_transform


def create_model():
    backbone = torchvision.models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features
    backbone.out_channels = 512

    anchor_size = config.anchor_size
    anchor_ratio = config.anchor_ratio
    anchor_generator = AnchorGenerator(sizes=(anchor_size,),
                                    aspect_ratios=(anchor_ratio,))

    class_map = config.class_map
    num_classes = len(class_map)

    min_size = config.min_size
    max_size = config.max_size

    box_detections_per_img = config.detections_per_img

    model = FasterRCNN(backbone=backbone,
                    num_classes=num_classes,
                    min_size=min_size,
                    max_size=max_size,
                    rpn_anchor_generator=anchor_generator,
                    box_detections_per_img=box_detections_per_img
                    )
    return model

def create_dataloader():
    train_batch = config.train_batch

    train_dataset = CustomDataset('/hdd/thaihq/qnet_search/ori_data/train', get_transform(train=True))
    val_dataset = CustomDataset('/hdd/thaihq/qnet_search/ori_data/val', get_transform(train=False))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch,
                                            shuffle=True, num_workers=4,
                                            collate_fn=utils.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch//2,
                                                shuffle=False, num_workers=4,
                                                collate_fn=utils.collate_fn)

    
    return train_data_loader, val_data_loader

def val(model, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    for images, targets in metric_logger.log_every(data_loader, print_freq=100):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(
            v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(images, targets)
            # losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        
    return metric_logger

def train():
    model = create_model()
    train_data_loader, val_data_loader = create_dataloader()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # construct an optimizer - SGD follow Faster R-CNN paper
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    stopper = EarlyStopping(config.patience)

    log_folder, weight_folder = create_folder('training', True)


    print('Start training')

    num_epochs = config.num_epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every 100 iterations
        train_loss = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=100)  # val_data_loader for debugging

        train_log_path = os.path.join(log_folder, 'train_log.txt')
        stopper.log(train_log_path, str(train_loss))

        last_weight_path = os.path.join(weight_folder, 'last.pt')
        torch.save(model, last_weight_path)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the val dataset
        print('Start validating')

        val_loss = val(model, val_data_loader, device)

        val_log_path = os.path.join(log_folder, 'val_log.txt')
        stopper.log(val_log_path, str(val_loss))

        # loss = float(str(val_loss).split('  ')[0].split(' ')[1]) # for debugging
        loss = float(str(val_loss).split('  ')[0].split(' ')[2][1:-1])  # Average

        stop = stopper(epoch, round(loss, 2))
        if stop:
            best_weight_path = os.path.join(weight_folder, 'best.pt')
            torch.save(model, best_weight_path)
            break

        
if __name__ == '__main__':
    train()