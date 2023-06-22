# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
#from visdom import Visdom # pip install Visdom
import settings
import datasets
from models import CNN, RES18
import argparse

def main():
    cnn = CNN()
    cnn.eval()

    parser = argparse.ArgumentParser(description="model path")
    parser.add_argument("--model-path", type=str, default="cnn_best.pt")
    
    cnn.load_state_dict(torch.load(args.model_path))
                                                                                                                                                         
    predict_dataloader = datasets.get_predict_data_loader()

    #vis = Visdom()
    for i, (images, labels) in enumerate(predict_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 0:settings.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, settings.ALL_CHAR_SET_LEN:2 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * settings.ALL_CHAR_SET_LEN:3 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * settings.ALL_CHAR_SET_LEN:4 * settings.ALL_CHAR_SET_LEN].data.numpy())]

        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print(c)
        #vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()


