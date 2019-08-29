# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

import math
import numpy as np
from bpcUtils import valuesToBinary

def CSC(values, maxDist=16, wordwidth=None):
  """CSC: encode each non-zero value as (rel. pos + value)"""
  #only transfer non-zero values, but with incremental coordinate
  if maxDist == None:
    maxDist = len(values)-1
  distBits = math.ceil(math.log2(maxDist))
  zeroVal = np.zeros((1,), dtype=values.dtype)[0]
  debug = False
  
  vals = [v for v in values if v != 0]
  idxs = [-1] + [i for i, v in enumerate(values) if v != 0]
  idxsRel = [i-iPrev for i, iPrev in zip(idxs[1:], idxs[:-1])]
  
  codedStream = ''
  for ir, v in zip(idxsRel, vals):
    while ir > maxDist:
      ir -= maxDist
      codedStream += bin(0)[2:].zfill(distBits)
      if debug: codedStream += '|'
      codedStream += valuesToBinary(zeroVal, wordwidth=wordwidth)
    codedStream += bin(ir-1)[2:].zfill(distBits)
    if debug: codedStream += '|'
    codedStream += valuesToBinary(v, wordwidth=wordwidth)
    if debug:
      codedStream += '||'
  return codedStream


def zeroRLE(values, maxZeroBurstLen=64, wordwidth=None):
  """zeroRLE: values with prefix, or different prefix and number of zeros"""
  #only transfer non-zero values, but with incremental coordinate
  if maxZeroBurstLen == None:
    maxZeroBurstLen = len(values)-1
  maxZeroBurstLenBits = math.ceil(math.log2(maxZeroBurstLen))
  zeroVal = np.zeros((1,), dtype=values.dtype)[0]
  debug = False
  
  codedStream = ''
  zeroCnt = 0
  for v in values:
    if v == 0 and zeroCnt < maxZeroBurstLen:
      zeroCnt += 1
    else:
      if zeroCnt > 0:
        #emit symbol
        codedStream += '0' + bin(zeroCnt-1+1)[2:].zfill(maxZeroBurstLenBits)
        if debug: codedStream += '|'
      #encode/keep track of current symbol
      if v == 0:
        zeroCnt = 1
      else:
        codedStream += '1' + valuesToBinary(v, wordwidth=wordwidth)
        if debug: codedStream += '|'
        zeroCnt = 0
        
  if zeroCnt > 0:
    #emit last symbol
    codedStream += '0' + bin(zeroCnt-1+1)[2:].zfill(maxZeroBurstLenBits)
    if debug: codedStream += '|'
     
  return codedStream


def ZVC(values, wordwidth=None, debug=False):
  """ZVC: zero value coding -- bit-vector for zero or not"""
  symbols = ''.join(['0' if v == 0 else '1' + valuesToBinary(v, wordwidth=wordwidth) for v in values])
  if debug: 
    return '|'.join(symbols)
  else:
    return ''.join(symbols)


# DEFLATE
#https://docs.python.org/3.7/library/zlib.html
import zlib
def deflate(values, level=9, wordwidth=None):
  valuesBitStream = valuesToBinary(values, wordwidth=wordwidth)
  valuesByteArr = int(valuesBitStream, 2).to_bytes(len(valuesBitStream) // 8, byteorder='big')
  byteArrCompr = zlib.compress(valuesByteArr, level=level)
#   intArrCompr = int.from_bytes(byteArrCompr, byteorder='big')
  finalBitStream = 'x'*len(byteArrCompr)*8 #valuesToBinary(values, wordwidth=32)
  return finalBitStream





















