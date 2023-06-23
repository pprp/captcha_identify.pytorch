# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import datasets
import models
import os, shutil
import argparse
import test
import torchvision
import settings
from PIL import Image
from torch.nn.parallel import DataParallel

def clean_unvalid_png():
    print("clean_unvalid_png")
    for dir in [settings.TRAIN_DATASET_PATH, settings.TEST_DATASET_PATH, settings.PREDICT_DATASET_PATH]:
        ok = 0
        fail = 0
        dir = os.path.abspath(dir)
        for file in os.listdir(dir):
            file = os.path.join(dir, file)
            try:
                image = Image.open(file) 
                ok += 1
            except Exception as e:
                fail += 1
                # print(e)
                os.remove(file)
        print(f"{dir}   [+{ok} / -{fail}]")


if __name__ == "__main__":
    clean_unvalid_png()