# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import datasets
from models import *
# import torch_util
import os, shutil
import argparse
import test
import torchvision
import settings

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters
num_epochs = 300
batch_size = 256
learning_rate = 0.001

# device = torch_util.select_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main(args):
    # cnn = CNN().to(device)
    cnn = RES18().to(device)
    
    cnn.train()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    if args.resume:
        cnn.load_state_dict(torch.load(args.model_path, map_location=device))

    max_acc = 0
    # Train the Model
    train_dataloader = datasets.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            cnn = CNN().to(device)

            # images = Variable(images)
            # labels = Variable(labels.float())
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            predict_labels = cnn(images)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 2 == 0:
                print("epoch: %03g \t step: %03g \t loss: %.5f \t\r" % (epoch, i+1, loss.item()))
                torch.save(cnn.state_dict(), "./weights/cnn_%03g.pt" % epoch)
        print("epoch: %03g \t step: %03g \t loss: %.5f \t" % (epoch, i, loss.item()))
        torch.save(cnn.state_dict(), "./weights/cnn_%03g.pt" % epoch)
        acc = test.test_data("./weights/cnn_%03g.pt" % epoch)
        if max_acc < acc:
            print("update accuracy %.5f." % acc)
            max_acc = acc
            shutil.copy("./weights/cnn_%03g.pt" % epoch, "./weights/cnn_best.pt")
        else:
            print("do not update %.5f." % acc)
        
    torch.save(cnn.state_dict(), "./weights/cnn_last.pt")
    print("save last model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load path")
    parser.add_argument('--model-path', type=str, default="./weights/cnn_0.pt")
    parser.add_argument('--resume',action='store_true')
    
    args = parser.parse_args()
    main(args)


