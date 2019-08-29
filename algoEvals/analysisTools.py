# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

import functools
import numpy as np
from bpcUtils import quantize

class Analyzer:
    def __init__(self, quantMethod, compressor):
        """ quantMethod: provides the default quantization method. Can e.g. in ['float32', 'float16', 'fixed16', 'fixed12', 'fixed8', ...]
            compressor: default compressor to use. A function taking a tensor and returning a string of bits. 
        """
        self.quantMethod_default = quantMethod
        self.compressor_default = compressor
    
    @functools.lru_cache(maxsize=8*200*100)
    def getComprProps(self, t, quant=None, compressor=None):

      if quant == None:
        quant = self.quantMethod_default
      if compressor == None: 
        compressor = self.compressor_default
        
      t, numBit, dtype = quantize(t, quant=quant)

      valueStream = t.view(-1).numpy().astype(dtype)

      codeStream = compressor(valueStream)
      if len(codeStream) > 0:
        comprRatio = len(valueStream)*numBit/len(codeStream) #strmLen
      else: 
        comprRatio = None
      comprSize = len(codeStream) #strmLen
      comprSizeBaseline = len(valueStream)*numBit
      return comprRatio, comprSize, comprSizeBaseline

    def getComprRatio(self, t, quant=None, compressor=None):
        
      if quant == None:
        quant = self.quantMethod_default
      if compressor == None: 
        compressor = self.compressor_default

      comprRatio, _, _ = self.getComprProps(t, quant=quant, compressor=compressor)
      return comprRatio

    def getSparsity(self, t, quant=None):
      """get the share of 0-valued entries"""
    
      if quant == None:
        quant = self.quantMethod_default
        
      return t.contiguous().view(-1).eq(0).long().sum().item()/t.numel()

    def getTotalComprProps(self, outputs, quant=None, compressor=None):
        
      comprProps = [self.getComprProps(outp, quant=quant, compressor=compressor) 
                    for outp in outputs]
      totalLen = np.array([l for _, l, _ in comprProps]).sum()
      uncomprLen = np.array([l for _, _, l in comprProps]).sum()
      return uncomprLen/totalLen, totalLen, uncomprLen