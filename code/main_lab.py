from resnet import ResNetCIFAR
from train import *
import torch
import numpy as np
from sklearn.manifold import TSNE
from dynamicConv import *
from attacks import *

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_val_cifar():
    net = ResNetCIFAR(num_layers=20, K=5)
    net = net.to(device)

    epoch = 100
    batch_size = 128
    lr = 0.1
    regularize_strength = 1e-4
    momentum = 0.9

    train(net, epoch, batch_size, lr, regularize_strength, momentum, log_every_n=50)

'''
K=5 0.9190
K=4 0.9241
K=3 0.9246
K=2 0.9194
'''
def report_TSNE():
    net = ResNetCIFAR(num_layers=20, K=4, log=True)
    net.to(device)
    net.load_state_dict(torch.load("pretrained_model_K4.pt"))
    # test(net)
    log(net)
    
def adversarial_attack():
    for i in range(2, 6):
        for j in range(2, 6):
            if i == j:
                continue
            attack_log(i, j)
            # attack(i, j, 'FGSM')
            # attack(i, j, 'PGD')
            # attack(i, j, 'MI-FGSM')
    # attack(5, 5, 'FGSM')
    # attack(5, 5, 'PGD')
    # attack_log(5, 5)
    # attack(5, 5, 'MI-FGSM')

if __name__ == '__main__':
    train_val_cifar()
    report_TSNE()
    adversarial_attack()

