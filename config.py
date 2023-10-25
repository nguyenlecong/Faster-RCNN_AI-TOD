# Model Config
min_size = 800  # 600
max_size = 800
anchor_ratio = (0.5, 1.0, 2.0)

# anchor_size = (32, 64, 128, 256, 512)  # default
anchor_size = (8, 16, 32, 64, 128, 256, 512)

detections_per_img = 1500

# Dataset Config

# XView
# class_map = {0: 'airplane', 1: 'ship', 2: 'storage-tank', 3: 'vehicle'}
# train_ratio = 0.7
# val_ratio = 0.2

# AI-TOD
class_map = {'airplane': 1, 'bridge': 2 , 'storage-tank': 3, 
             'ship': 4, 'swimming-pool': 5,  'vehicle': 6,
             'person': 7, 'wind-mill': 8}

train_batch = 4  # val_batch = train_batch // 2
test_batch = 1

# Train/Test mode
mode = 'train'
patience = 10  # 1 for debugging
num_epochs = 10000
learning_rate = 0.01

# Test
test_weight_path = 'training/1/weights/best.pt'
