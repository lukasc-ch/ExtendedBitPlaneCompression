# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

import torch
import numpy as np

import tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob
import csv

import sys
sys.path.append('./quantLab')

def getModel(modelName, epoch=None, returnPath=False):
    import torchvision as tv 
    import quantlab.ImageNet.topology as topo
    
    loss_func =  torch.nn.CrossEntropyLoss()
    model = None
    
    if modelName == 'alexnet':
        model = tv.models.alexnet(pretrained=True)
    elif modelName == 'squeezenet':
        model = tv.models.squeezenet1_1(pretrained=True)
    elif modelName == 'resnet34':
        model = tv.models.resnet34(pretrained=True)
    elif modelName == 'vgg16':
        model = tv.models.vgg16_bn(pretrained=True)
    elif modelName == 'mobilenet2':
        model = tv.models.mobilenet_v2(pretrained=True)
        
    elif modelName == 'alexnet-cust':
        path = './quantLab/ImageNet/log/exp00/'
        model = topo.AlexNetBaseline(capacity=1)
        if epoch == None: epoch = 54
        tmp = torch.load(path + '/save/epoch%04d.ckpt' % epoch, map_location=torch.device('cpu'))
        model.load_state_dict(tmp['net'])
        
    elif modelName == 'mobilenetV2-cust':
        path = './quantLab/ImageNet/log/exp05/'
        model = topo.MobileNetV2Baseline(capacity=1, expansion=6)
        if epoch == None: epoch = 200
        tmp = torch.load(path + '/save/epoch%04d.ckpt' % epoch, map_location=torch.device('cpu'))
        model.load_state_dict(tmp['net'])
    
    assert(model != None)
    if returnPath:
        return model, loss_func, path
    else:
        return model, loss_func

def getTensorBoardData(path, traceName):
    """Reads values form Tensorboard log files."""
    if not os.path.isfile(path):
        pathOpts = glob.glob(path + '/events.out.tfevents.*')
        assert(len(pathOpts) == 1)
        path = pathOpts[0]
    
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    # Show all tags in the log file: print(event_acc.Tags())
    
    trace  = event_acc.Scalars(traceName)
    values = [v.value for v in trace]
    steps  = [v.step for v in trace]
    return steps, values

def getFMs(model, loss_func, training=True, numBatches=1, batchSize=10, computeGrads=False, safetyFactor=0.75):
    
    # CREATE DATASET LOADERS
    import quantLab.quantlab.ImageNet.preprocess as pp
    datasetTrain, datasetVal, _ = pp.load_datasets('./ilsvrc12/', augment=False)
    if training:
        dataset = datasetTrain
        model.train()
    else:
        dataset = datasetVal
        model.eval()
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)

    # SELECT MODULES
    msReLU = list(filter(lambda m: type(m) == torch.nn.modules.ReLU or type(m) == torch.nn.modules.ReLU6, model.modules()))
    msConv = list(filter(lambda m: type(m) == torch.nn.modules.Conv2d, model.modules()))
    msBN = list(filter(lambda m: type(m) == torch.nn.modules.BatchNorm2d, model.modules()))

    #register hooks to get intermediate outputs: 
    def setupFwdHooks(modules):
      outputs = []
      def hook(module, input, output):
          outputs.append(output.detach().contiguous().clone())
      for i, m in enumerate(modules): 
        m.register_forward_hook(hook)
      return outputs

    #register hooks to get gradient maps: 
    def setupGradHooks(modules):
      grads = []
      def hook(module, gradInput, gradOutput):
          assert(len(gradInput) == 1)
          grads.insert(0, gradInput[0].contiguous().clone())
      for i, m in enumerate(modules): 
        m.register_backward_hook(hook)
      return grads

    outputsReLU = setupFwdHooks(msReLU)
    outputsConv = setupFwdHooks(msConv)
    outputsBN   = setupFwdHooks(msBN)
    gradsReLU   = setupGradHooks(msReLU)
    
    # PASS IMAGES THROUGH NETWORK
    outputSetsMaxMulti, gradSetsMaxMulti = [], []
    outputSets, gradSets = [outputsReLU, outputsConv, outputsBN], [gradsReLU]
    dataIterator = iter(dataLoader)

    for _ in range(numBatches):
      for outputs in outputSets: 
        outputs.clear()
      for grads in gradSets: 
        grads.clear()

      (image, target) = next(dataIterator)

      if training:
        model.train()
        outp = model(image)
        if computeGrads:
            loss = loss_func(outp, target)
            loss.backward()
      else: 
        model.eval()
        outp = model(image)

      tmp = [[outp.max().item() for outp in outputs] 
             for outputs in outputSets]
      outputSetsMaxMulti.append(tmp)
      if computeGrads:
          tmp = [[grad.max().item() for grad in grads] 
                 for grads in gradSets]
          gradSetsMaxMulti.append(tmp)

    outputSetsMax = [np.array([om2[i] for om2 in outputSetsMaxMulti]).max(axis=0) for i in range(len(outputSets))]
    if computeGrads:
        gradSetsMax = [np.array([om2[i] for om2 in gradSetsMaxMulti]).max(axis=0) for i in range(len(gradSets))]
    
    # NORMALIZE
    for outputs, outputsMax in zip(outputSets, outputSetsMax):
      for op, opmax in zip(outputs,outputsMax):
        op.mul_(safetyFactor/opmax)

    if computeGrads:
        for grads, gradsMax in zip(gradSets, gradSetsMax):
          for op, opmax in zip(grads,gradsMax):
            op.mul_(safetyFactor/opmax)
    else: 
        gradsReLU = []
    
    return outputsReLU, outputsConv, outputsBN, gradsReLU