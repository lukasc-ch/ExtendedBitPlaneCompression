# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

import pandas as pd
import datetime
import multiprocessing 
import tqdm

import bpcUtils
import dataCollect
import analysisTools
import referenceImpl

analyzer = analysisTools.Analyzer(quantMethod='fixed16', compressor=None)

def zeroRLE_default(x, wordwidth): 
    return referenceImpl.zeroRLE(x, maxZeroBurstLen=16, wordwidth=wordwidth)
def ZVC_default(x, wordwidth):
    return referenceImpl.ZVC(x, wordwidth=wordwidth)
def BPC_default(x, wordwidth):
    return bpcUtils.BPC(x, chunkSize=8, variant='baseline', wordwidth=wordwidth)
def BPCplus_default(x, wordwidth):
    # bpcVariant in ['ours-02', 'baseline', 'ours-03']
    return bpcUtils.BPCplus(x, chunkSize=8, maxRLEBurstLen=16, 
                            variant='baseline', bpcVariant='ours-04', wordwidth=wordwidth)

comprMethodsByName = {'zero-RLE': zeroRLE_default, 
                      'ZVC': ZVC_default, 
                      'BPC': BPC_default, 
                      'ours': BPCplus_default}
training=True
batchSize = 250
modelNames = ['alexnet', 'resnet34', 'squeezenet', 'vgg16', 'mobilenet2']#, 'alexnet-cust', 'mobilenetV2-cust']

for modelName in modelNames:
    print('running on model: %s' % modelName)
    model, loss_func = dataCollect.getModel(modelName)
    outputsReLU, _, _, gradsReLU = dataCollect.getFMs(model, loss_func, 
                                                      training=training, computeGrads=True,
                                                      numBatches=1, batchSize=batchSize)#250)
    numLayers = len(outputsReLU)
    print('done getting FMs, now compressing...')

    def configsGen():

        if training: 
            dataDescrOpts = ['outputs', 'gradients']
        else:
            dataDescrOpts = ['outputs']

        for netName in [modelName]:
            for dataDescr in dataDescrOpts:
                if dataDescr == 'outputs':
                    quantMethods = ['fixed8', 'fixed12', 'fixed16', 'float16']
                elif dataDescr == 'gradients':
                    quantMethods = ['fixed16', 'float16']
                else: 
                    assert(False)

                for qm in quantMethods:
                    for comprName in comprMethodsByName.keys():
                        for layerIdx in range(numLayers):
                            for intraBatchIdx in range(batchSize):
                                yield netName, dataDescr, qm, comprName, layerIdx, intraBatchIdx

    def do_work(config):
        netName, dataDescr, qm, comprName, layerIdx, intraBatchIdx = config
        if dataDescr == 'outputs':
            dataTensor = outputsReLU
        elif dataDescr == 'gradients': 
            dataTensor = gradsReLU
        else:
            assert(False)
        return analyzer.getComprProps(dataTensor[layerIdx][intraBatchIdx], 
                                      quant=qm, 
                                      compressor=lambda x: comprMethodsByName[comprName](
                                          x, wordwidth=bpcUtils.getWW(qm))
                                     )

    configs = list(configsGen())
    pool = multiprocessing.Pool(processes=20)
    comprPropsByConfig = [r for r in tqdm.tqdm(pool.imap(do_work, configs), total=len(configs))]
    # comprPropsByConfig = [do_work(c) for c in configs] # for debugging...
    pool.close()

    df = pd.DataFrame.from_records(configs, 
                                   columns=['modelName', 'dataDescr', 'quantMethod', 
                                            'comprName', 'layerIdx', 'intraBatchIdx'])
    df = df.join(pd.DataFrame.from_records(comprPropsByConfig, 
                                           columns=['comprRatio', 'comprSize', 'comprSizeBaseline']))

    df.to_pickle('results/results-%s.pkl' % (datetime.datetime.now().strftime('%y%m%d-%H%M%S')))
