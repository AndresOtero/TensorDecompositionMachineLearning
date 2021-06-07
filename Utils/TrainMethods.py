from __future__ import print_function

import csv
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn

from Nets.ParallelizedTensorsNet import ParallelizedTensorNet
from Nets.TRNetSerialized import TRNetSerialized, TRNetSerializedCell
from Utils import Constant, EnumModel, EnumDataset
from Utils.NetParamsFactory import NetParamsFactory
from Utils.PreProcessText import PreProcessText
from Utils.DataLoaderUtil import DataLoaderUtil
from Utils.NetParams import NetParams
from Utils.OptimizerFactory import OptimizerFactory
from Utils.TimerUtil import TimerUtil

def train_vision(model, device, train_loader, optimizer, epoch, log_while_training, log_interval):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and log_while_training:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTrain  Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))


def test_vision(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)


def train_text(model, device, train_loader, optimizer, epoch, log_while_training, log_interval):
    model.train()
    batch_idx = 0
    correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch.text)
        target =batch.label.clone().detach().type(torch.LongTensor)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_idx += 1
        if batch_idx % log_interval == 0 and log_while_training:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(text), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTrain  Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))

def test_text(model, device, test_loader):
    model.eval()
    #test_loader.to(device)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            output = model(batch.text)
            target = batch.label.clone().detach().type(torch.LongTensor)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        -test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return -test_loss, 100. * correct / len(test_loader.dataset)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train_text_binary(model, device, train_loader, optimizer, epoch, log_while_training, log_interval):
    model.train()
    batch_idx = 0
    epoch_loss = 0
    epoch_acc = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch.text).squeeze(1)
        loss = criterion(output, batch.label)
        loss.backward()
        optimizer.step()
        batch_idx += 1
        if batch_idx % log_interval == 0 and log_while_training:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch.text), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        acc = binary_accuracy(output, batch.label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    epoch_loss/=len(train_loader)
    epoch_acc /=len(train_loader)
    print(f'\tTrain Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}%')

def test_text_binary(model, device, test_loader):
    model.eval()
    #test_loader.to(device)
    epoch_loss = 0
    epoch_acc = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in test_loader:
            output = model(batch.text).squeeze(1)
            loss =  criterion(output, batch.label)
            acc = binary_accuracy(output, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    epoch_loss/=len(test_loader)
    epoch_acc /=len(test_loader)
    print(f'\t Test. Loss: {epoch_loss:.3f} |  test. Acc: {epoch_acc * 100:.2f}%')

    return -epoch_loss, 100. *epoch_acc