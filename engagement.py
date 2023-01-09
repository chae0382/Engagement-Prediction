import os
import gc
import cv2
import wandb
import torch
import argparse
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch import nn
from PIL import Image
from time import time
from dataset import DAiSEE
from torchvision import datasets
from facenet_pytorch import MTCNN
from models.gazeheadnet import GazeHeadNet
from models.fer_net import Encoder, Regressor
from torchvision.transforms import ToPILImage, ToTensor

from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser()

# Hyperparameter setting
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument("-lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("-b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("-b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("-weight_decay", type=float, default=1e-4, help="Weight decay")

# GPU and random seed setting
parser.add_argument('-num_workers', type=int, default=2)
parser.add_argument('-seed', type=int, default=1243)
parser.add_argument("-gpu", type=str, default="0", help="GPU ID")

# Path
parser.add_argument('-dataset_path', type=str, default='')
parser.add_argument('-result_path', type=str, default='./result', help="Path to save results")

# Log
parser.add_argument('-save_freq', type=int, default=1000, help="Frequency to save model")
parser.add_argument("-project_name", type=str, default="engagement", help="project name for wandb")
parser.add_argument("-exp_name", type=str, default="base", help="Experiment name for wandb")
parser.add_argument('-print_freq', type=int, default=20, metavar='', help="Frequency to print training process")
parser.add_argument("-sample_interval", type=int, default=100, help="Sample download")

# Checkpoints
parser.add_argument('-load_step', type=int, default=0, help='Checkpoint step')

args = parser.parse_args()
print(args)

import warnings

warnings.filterwarnings("ignore")

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'

import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# =============================================================================
# 각자의 Wandb ID Key로 대체
# 참고: https://sooftware.io/wandb/
# =============================================================================
# Wandb setting
os.environ['WANDB_API_KEY'] = "######"
wandb.init(project=args.project_name, entity="####", name=args.exp_name)
wandb.config.update(args)
# =============================================================================
# ## example
# os.environ['WANDB_API_KEY'] = "0ee23525f6f4ddbbab74086ddc0b2294c7793e80"
# wandb.init(project=args.project_name, entity="psj", name=args.exp_name)
# wandb.config.update(args)
# =============================================================================
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# img_size = (480, 640)
img_size = (224, 224)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])

train_subsample = 0.3
train_dataset = DAiSEE("Train", transform)
train_dataset = Subset(train_dataset,
                       np.linspace(start=0, stop=len(train_dataset), num=int(train_subsample * len(train_dataset)),
                                   endpoint=False, dtype=np.uint32))

test_subset = 0.1
test_dataset = DAiSEE("Test", transform)
test_dataset = Subset(test_dataset,
                      np.linspace(start=0, stop=len(test_dataset), num=int(test_subset * len(test_dataset)),
                                  endpoint=False, dtype=np.uint32))

# val_dataset = DAiSEE("Validation", transform)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))
# print(len(val_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

# FER network
fer_encoder, fer_regressor = Encoder().to(device), Regressor().to(device)
fer_encoder.load_state_dict(torch.load('weights/AffectNet_KDH_GAN_enc_alexnet_10.t7'),
                            strict=False)  # Load the weight of model
fer_regressor.load_state_dict(torch.load('weights/AffectNet_KDH_GAN_reg_alexnet_10.t7'), strict=False)
fer_encoder.eval()
fer_regressor.eval()

# Gaze network
gazenet = GazeHeadNet().to(device)
gazenet.load_state_dict(torch.load('weights/gazeheadnet.pth.tar'), strict=False)
gazenet.eval()


def merge(gaze, fer):
    # gaze_vector = np.array(gaze)
    # fer_vector = np.array(fer)
    eng_vector = torch.cat([gaze, fer], dim=1)
    return eng_vector


class eng_network(nn.Module):
    def __init__(self):
        super(eng_network, self).__init__()  ##########################Check
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


eng_model = eng_network().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(eng_model.parameters(), lr=args.lr, betas=(args.b1, args.b2),
                             weight_decay=args.weight_decay)

eng_model.train()

mtcnn = MTCNN(device='cuda', post_process=False)
transform_to_PIL = ToPILImage()
transform_to_tensor = ToTensor()

# Result path
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

result_path = os.path.join(args.result_path, args.project_name, args.exp_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)

mean = 0.
std = 0.

total_data = 0
total_correct = 0

for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(train_loader):
        eng_model.train()
        iteration = epoch * len(train_loader) + i

        images, labels = images.to(device), labels.to(device)
        assert images is not None and labels is not None, "DATA!!!"  # 0.

        # Face --> (gaze, va)
        gaze = gazenet(images)
        fer_z = fer_encoder(images)
        va = fer_regressor(fer_z)

        # (gaze, va) --> engagement
        input_ = merge(gaze, va)
        out = eng_model(input_)
        _, prediction = torch.max(out.data, 1)

        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = (prediction == labels).sum().item()
        num_data = images.size(0)
        acc = (correct / num_data) * 100

        total_correct += correct
        total_data += num_data
        total_acc = (total_correct / total_data) * 100

        log_dict = {"Loss": loss,
                    "acc": acc,
                    "total_acc": total_acc}
        wandb.log(log_dict.copy())

        # Print the training process
        if iteration % args.print_freq == 0:
            logging.info(
                "[Epoch %d/%d] [Iteration %d/%d] %s" % (epoch, args.epochs, iteration, args.epochs * len(train_loader),
                                                        ' '.join(["[%s: %.5f]" % (k, v) for k, v in log_dict.items()])
                                                        )
                )

        # Save the model every specified iteration.
        if iteration != 0 and iteration % args.save_freq == 0:
            ckpt_path = os.path.join(result_path, "model_%d.pth" % iteration)
            torch.save(eng_model.state_dict(), ckpt_path)

        if iteration % args.sample_interval == 0:
            correct_test = 0
            total_test = 0
            for idx, (x_test, y_test) in enumerate(test_loader):
                x_test, y_test = x_test.to(device), y_test.to(device)

                with torch.no_grad():
                    eng_model.eval()

                    gaze = gazenet(x_test)
                    fer_z = fer_encoder(x_test)
                    va = fer_regressor(fer_z)

                    # (gaze, va) --> engagement
                    input_ = merge(gaze, va)
                    out = eng_model(input_)
                    _, prediction = torch.max(out.data, 1)

                    correct_test += (prediction == y_test).sum().item()
                    total_test += x_test.size(0)

            wandb.log({"valid acc": (correct_test / total_test) * 100})

# gc.collect()
torch.cuda.empty_cache()
logging.info("Saving the final model...")
ckpt_path = os.path.join(result_path, "model_final.pth")
torch.save(eng_model.state_dict(), ckpt_path)
logging.info("Training finished.")

logging.info("")
logging.info("Start final evaluation")
for idx, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

    with torch.no_grad():
        eng_model.eval()

        gaze = gazenet(x_test)
        fer_z = fer_encoder(x_test)
        va = fer_regressor(fer_z)

        # (gaze, va) --> engagement
        input_ = merge(gaze, va)
        out = eng_model(input_)
        _, prediction = torch.max(out.data, 1)

        correct_test += (prediction == y_test).sum().item()
        total_test += x_test.size(0)

logging.info("")
logging.info("Final result")
logging.info("Accuracy: %.4f" % (correct_test / total_test) * 100)
