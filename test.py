# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import settings
import datasets
from models import *
import one_hot_encoding
import argparse
import torch_util
import os
from models import *
from tqdm import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cpu")

def main(model_path):
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path, map_location=device))
    print("load cnn net.")

    test_dataloader = datasets.get_test_data_loader()

    correct = 0
    total = 0

    pBar = tqdm(total=test_dataloader.__len__())

    for i, (images, labels) in enumerate(test_dataloader):
        pBar.update(1)

        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 0:settings.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, settings.ALL_CHAR_SET_LEN:2 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * settings.ALL_CHAR_SET_LEN:3 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * settings.ALL_CHAR_SET_LEN:4 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if(predict_label == true_label):
            correct += 1
        # if(total%200==0):
            # print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

def test_data(model_path):
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path, map_location=device))
    test_dataloader = datasets.get_test_data_loader()

    correct = 0
    total = 0

    for i, (images, labels) in enumerate(test_dataloader):

        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 0:settings.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, settings.ALL_CHAR_SET_LEN:2 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * settings.ALL_CHAR_SET_LEN:3 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * settings.ALL_CHAR_SET_LEN:4 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if(predict_label == true_label):
            correct += 1
        # if(total%200==0):
            # print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    return 100 * correct / total
    # print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test path")
    parser.add_argument('--model-path', type=str, default="weights/cnn_1.pt")

    args = parser.parse_args()
    main(args.model_path)


