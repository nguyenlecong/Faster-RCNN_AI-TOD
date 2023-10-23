import os
import sys

import matplotlib.pyplot as plt


def plot(log_folder, mode):
    path = os.path.join(log_folder, f'{mode}_log.txt')
    if not os.path.exists(path):
        return
    
    file = open(path, 'r')
    lines = file.readlines()
    
    if mode == 'train':
        idx = 1  # 0 is lr
        
        lr = [float(str(i).split('  ')[0].split(' ')[1]) for i in lines]
        plt.figure(figsize=(15, 10), tight_layout=True)
        plt.plot(range(1, len(lr)+1), lr, label='learning rate')
        plt.title('Learning rate')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        save_path = os.path.join(log_folder, f'learning_rate.png')
        plt.savefig(save_path)
        # plt.show()

    elif mode == 'val':
        idx = 0

    loss = [float(str(i).split('  ')[idx].split(' ')[2][1:-1]) for i in lines]

    plt.figure(figsize=(15, 10), tight_layout=True)
    plt.plot(range(1,len(loss)+1), loss, label='loss')

    min_loss = min(loss)
    min_index = loss.index(min(loss))
    plt.plot(min_index, min_loss, '*', label='best value')
    
    plt.title(f'{mode} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    save_path = os.path.join(log_folder, f'{mode}_loss.png')
    plt.savefig(save_path)

    # plt.show()

if __name__ == '__main__':
    idx = sys.argv[1]
    log_folder = f'training/{idx}'
    plot(log_folder, mode='train')
    plot(log_folder, mode='val')
