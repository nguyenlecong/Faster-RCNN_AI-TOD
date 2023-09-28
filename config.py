# Model Config
min_size = 600  # model 1+4
# min_size = 1200  # model 2+3

anchor_ratio = (0.5, 1.0, 2.0)

# anchor_size = (64, 128, 256, 512)  # model 1
# anchor_size = (32, 64, 128, 256, 512)  # model 2
anchor_size = (8, 16, 32, 64, 128, 256, 512)  # model 3+4

# Dataset Config
# class_map = {0: 'airplane', 1: 'ship', 2: 'storage-tank', 3: 'vehicle'}  # model 1+2+3
class_map = {'airplane': 1, 'bridge': 2 , 'storage-tank': 3, 
             'ship': 4, 'swimming-pool': 5,  'vehicle': 6,
             'person': 7, 'wind-mill': 8}  # model 4
train_ratio = 0.7
val_ratio = 0.2
train_val_batch = 16
test_batch = 1

# Train/Test mode
mode = 'train'

num_epochs = 20