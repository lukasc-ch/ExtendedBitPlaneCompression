# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

import numpy as np
import math

def valuesToBinary(t, wordwidth=None):
  """Converts a numpy array using its datatype to a string of 1/0 values"""
  arr = t.byteswap().tobytes()
  wordwidthInput = t.dtype.itemsize*8# t[0].nbytes*8
  if wordwidth is None:
    wordwidth = wordwidthInput
  longbinarr = bin(int.from_bytes(arr, byteorder='big'))[2:].zfill(len(arr)*8)
  
  #shorten the subwords
  binarr = ''.join([longbinarr[i:i+wordwidthInput][-wordwidth:] 
                   for i in range(0, len(longbinarr), wordwidthInput)])
  return binarr


###############################
# Ingredients for BPC: delta compression, delta-binarization, and bit-place XOR-ing
###############################

def deltaCompr(values):
  base = values[0]
  diffs = values[1:] - values[0:-1]
  return base, diffs

def deltaBP(values, wordwidth=None):
  base, diffs = deltaCompr(values)
  binDiffs = [valuesToBinary(v, wordwidth=wordwidth+1) for v in diffs]
  DBPs = [''.join(dbp) for dbp in zip(*binDiffs)]
  return valuesToBinary(base, wordwidth=wordwidth), DBPs

def DBX(values, wordwidth=None):
  base, DBPs = deltaBP(values, wordwidth=wordwidth)

  def xor(a, b): 
    y = int(a, 2)^int(b, 2)
    return bin(y)[2:].zfill(len(a))

  it1, it2 = iter(DBPs[::-1]), iter(DBPs[::-1])
  baseDBP = next(it1)
  DBXs = [xor(dbp, dbpPrev) for dbpPrev, dbp in zip(it1, it2)][::-1]
  return base, DBPs, DBXs

###############################
# BPC IMPLEMENTATION
###############################

bpcPerfCounter_init = {'multi-all-0 DBX': 0, 'all-0 DBX': 0, 'all-1 DBX': 0, 
                       'all-0 DBP': 0, '2-consec 1s': 0, 'single-1': 0, 
                       'uncompressed': 0, 
                       'numBlocks': 0}
bpcPerfCounter = bpcPerfCounter_init
bpcPerfCounterEnable = True

def bpcPerfCounterEnable(enable=True, reset=True):
  bpcPerfCounterEnable = enable
  if reset: 
    import copy
    bpcPerfCounter = copy.deepcopy(bpcPerfCounter_init)
    
def bpcPerfCounterIncr(propname, increment=1):
  if bpcPerfCounterEnable:
    bpcPerfCounter[propname] += increment
    

def BPC_block(values, variant='baseline', wordwidth=None, prevValue=0):
  
  bpcPerfCounterIncr('numBlocks')
  
  if len(values) < 2:
    return '' # this is wrong, but just a minor boundary effect...
  
  #sanitize input
  if variant == 'ours-02' or variant == 'ours-03':
    values = [prevValue] + values
  base, DBPs, DBXs = DBX(values, wordwidth=wordwidth)
  baseDBP = DBPs[-1]
  
  #flags
  debug = False
  chunkSize = len(values)
  chunkSizeL2C = math.ceil(math.log2(chunkSize))
  wordwidthL2C = math.ceil(math.log2(wordwidth))
  
  #encode first value before differences
  codedStream = ''
  if variant == 'ours-02' or variant == 'ours-03':
    pass
  else:
    codedStream += base
  if debug: 
    codedStream += '!'
  DBPsymbols = DBXs + [baseDBP]
  
  # perform code mapping of individual DBP/DBX symbols
  def codeMapper(dbx, dbp):
    if dbx == '0'*len(dbx):
      bpcPerfCounterIncr('all-0 DBX')
      if variant == 'ours-03' or variant == 'ours-04':
        return '01'
      else:
        return '001'
    if dbx == '1'*len(dbx):
      bpcPerfCounterIncr('all-1 DBX')
      return '00000'
    if dbp == '0'*len(dbp): # and implicitly dbx != 0
      bpcPerfCounterIncr('all-0 DBP')
      return '00001'

    # find first 1-bit and count number of 1-bits
    numOnes = 0
    firstIdx = -1
    for idx, b in enumerate(dbx):
      if b == '1':
        numOnes += 1
        if numOnes == 1:
          firstIdx = idx

    if numOnes == 2 and dbx[firstIdx+1] == '1': #two consec 1s
      bpcPerfCounterIncr('2-consec 1s')
      return '00010' + bin(firstIdx)[2:].zfill(chunkSizeL2C)
    if numOnes == 1:
      bpcPerfCounterIncr('single-1')
      return '00011' + bin(firstIdx)[2:].zfill(chunkSizeL2C)

    bpcPerfCounterIncr('uncompressed')
    return '1' + dbx 

  def reencoder(codedDBXsymbols):
    #postprocessing to merge multiple consecutive DBX==all-0 cases (001 symbols) to (01+cnt)
    codedSymbols = []
    runLen = 0
    for idx, symb in enumerate(codedDBXsymbols):
      if variant == 'ours-04':
          if symb == '01':
            runLen += 1
            if idx == len(codedDBXsymbols)-1:
              if runLen == 1:
                codedSymbols.append('001')
              elif runLen > 1:
                bpcPerfCounterIncr('multi-all-0 DBX')
                bpcPerfCounterIncr('all-0 DBX', -runLen)
                codedSymbols.append('01' + bin(runLen-2)[2:].zfill(wordwidthL2C))
          else:
            if runLen == 1:
              codedSymbols.append('001')
            elif runLen > 1:
              bpcPerfCounterIncr('multi-all-0 DBX')
              bpcPerfCounterIncr('all-0 DBX', -runLen)
              codedSymbols.append('01' + bin(runLen-2)[2:].zfill(wordwidthL2C))#zfill(5) for max. 32 values per block
            runLen = 0
            codedSymbols.append(symb)
      elif variant == 'ours-03':
          if symb == '01':
            runLen += 1
            if idx == len(codedDBXsymbols)-1:
              if runLen == 1:
                codedSymbols.append('01')
              elif runLen > 1:
                bpcPerfCounterIncr('multi-all-0 DBX')
                bpcPerfCounterIncr('all-0 DBX', -runLen)
                codedSymbols.append('001' + bin(runLen-2)[2:].zfill(wordwidthL2C))
          else:
            if runLen == 1:
              codedSymbols.append('01')
            elif runLen > 1:
              bpcPerfCounterIncr('multi-all-0 DBX')
              bpcPerfCounterIncr('all-0 DBX', -runLen)
              codedSymbols.append('001' + bin(runLen-2)[2:].zfill(wordwidthL2C))#zfill(5) for max. 32 values per block
            runLen = 0
            codedSymbols.append(symb)
      else:
          if symb == '001':
            runLen += 1
            if idx == len(codedDBXsymbols) - 1:
              if runLen == 1:
                codedSymbols.append('001')
              elif runLen > 1:
                bpcPerfCounterIncr('multi-all-0 DBX')
                bpcPerfCounterIncr('all-0 DBX', -runLen)
                codedSymbols.append('01' + bin(runLen-2)[2:].zfill(wordwidthL2C))
          else:
            if runLen == 1:
              codedSymbols.append('001')
            elif runLen > 1:
              bpcPerfCounterIncr('multi-all-0 DBX')
              bpcPerfCounterIncr('all-0 DBX', -runLen)
              codedSymbols.append('01' + bin(runLen-2)[2:].zfill(wordwidthL2C))
            runLen = 0
            codedSymbols.append(symb)
    if debug: 
      codedStream = '|'.join(codedSymbols)
    else:
      codedStream = ''.join(codedSymbols)
    return codedStream
    
  codedDBXsymbols = [codeMapper(dbx, dbp) for dbx, dbp in zip(DBXs, DBPs[:-1])]
  codedDBXsymbols += [codeMapper(baseDBP, baseDBP)]
    
  codedStream += reencoder(codedDBXsymbols)
  return codedStream


def BPC(values, chunkSize=32, variant='baseline', wordwidth=None):
  
  if chunkSize == None:
    chunkSize = int(1e12)
  
  if variant == 'ours-02' or variant == 'ours-03':
    chunkSize -= 1

  def blocks(l, n):
      """Yield successive n-sized blocks from l."""
      for i in range(0, len(l), n):
          yield l[i:i+n]

  strm = ''
  prevBlock = [0]
  for w in blocks(values, chunkSize):
    strm += BPC_block(w, variant=variant, wordwidth=wordwidth, prevValue=prevBlock[0])
    prevBlock = w
  return strm




##############################
# Implementation BPCplus
##############################
def BPCplus(values, variant='baseline', bpcVariant='ours-02', 
            maxRLEBurstLen=16, chunkSize=16, wordwidth=None, debug=False):
#     return BPCplus_golden(values, variant=variant, bpcVariant=bpcVariant, 
#             maxRLEBurstLen=maxRLEBurstLen, chunkSize=chunkSize, wordwidth=wordwidth, debug=debug)
    return BPCplus_fastLength(values, variant=variant, bpcVariant=bpcVariant, 
            maxRLEBurstLen=maxRLEBurstLen, chunkSize=chunkSize, wordwidth=wordwidth, debug=debug)

def BPCplus_golden(values, variant='baseline', bpcVariant='ours-02', 
            maxRLEBurstLen=16, chunkSize=16, wordwidth=None, debug=False):
  
  if maxRLEBurstLen == None:
    maxRLEBurstLen = len(values)-1
  maxRLEBurstLenBits = math.ceil(math.log2(maxRLEBurstLen))
    
  def RLE(values):
    #only transfer non-zero values, but with incremental coordinate
    zeroVal = np.zeros((1,), dtype=values.dtype)[0]

    zeroBlockLens = []
    zeroCnt = 0
    for v in values:
      if v == 0 and zeroCnt < maxRLEBurstLen:
        zeroCnt += 1
      else:
        zeroBlockLens += [zeroCnt]
        zeroCnt = 1 if v == 0 else 0 #encode/keep track of current symbol
      
    RLEblocks = []
    obl = 0
    for zbl in zeroBlockLens:
      emitOB = False
      if zbl > 0:
        if obl > 0:
          emitOB = True
        RLEblocks += [(0,zbl)]
      else:
        obl += 1
        if obl >= chunkSize:
          emitOB = True
        
      if emitOB:
        RLEblocks += [(1,obl)]
        obl = 0

    return RLEblocks
  
  def encodeRLEsym(symbol, count):
    if symbol == 0:
      return '0' + bin(count-1)[2:].zfill(maxRLEBurstLenBits)
    elif symbol == 1:
      return '1' + bin(count-1)[2:].zfill(maxRLEBurstLenBits)
    else:
      assert(False)
      
  def encodeRLEBursts(RLEblocks, fast=True):
    encodedSyms = [encodeRLEsym(symbol, count) for symbol, count in RLEblocks]
    if debug:
      return '|'.join(encodedSyms)
    else:
      return ''.join(encodedSyms)
  
  global nonzeroValues
  nonzeroValues = np.array(list(filter(lambda x: x != 0, values)), 
                           dtype=values.dtype)

  if variant == 'ZVC':
    RLEBitStream = ''.join(['0' if v == 0 else '1' for v in values])
  else:
    RLEblocks = RLE(values)
    RLEBitStream = encodeRLEBursts(RLEblocks)
    
  nonzeroBitStream = BPC(nonzeroValues, chunkSize=chunkSize, 
                         variant=bpcVariant, wordwidth=wordwidth)
  
  if debug:
    codedStream = RLEBitStream + '||' + nonzeroBitStream
  else:
    codedStream = RLEBitStream + nonzeroBitStream
  return codedStream


def BPCplus_fastLength(values, variant='baseline', bpcVariant='ours-02', 
            maxRLEBurstLen=16, chunkSize=16, wordwidth=None, debug=False):
  
  if maxRLEBurstLen == None:
    maxRLEBurstLen = len(values)-1
  maxRLEBurstLenBits = math.ceil(math.log2(maxRLEBurstLen))
    
  nonzeroValues = np.array(values[values!=0], 
                           dtype=values.dtype)

  #fast bit counting
  assert(variant != 'ZVC' and debug == False)
  def symbolize(binData):
    t = binData
    sel = t[:-1] * (1-t[1:])
    u = t.cumsum()
    lens = (u[1:])[sel>0]
    lens = lens - np.concatenate(([0],lens[:-1]))
    symbsLen = ((lens-1)//maxRLEBurstLen + 1).sum()*(maxRLEBurstLenBits+1)
    return symbsLen
  
  zeroNonzeroStreamLen = symbolize(values == 0) + (values != 0).sum()#symbolize(values != 0)
  RLEBitStream = 'x'*zeroNonzeroStreamLen
    
  nonzeroBitStream = BPC(nonzeroValues, chunkSize=chunkSize, 
                         variant=bpcVariant, wordwidth=wordwidth)
  
  if debug:
    codedStream = RLEBitStream + '||' + nonzeroBitStream
  else:
    codedStream = RLEBitStream + nonzeroBitStream
  return codedStream

################################
# quantization function
################################

def quantize(x, quant, safetyFactor=1.0, normalize=False):
  """Quantizes the tensor 'x' using the method 'quant'. 
     @param x: tensor with data to quantize. 
     @param quant: quantization method to use (float32/16, fixed32/16/8, ufixed32/16/8)
     @param normalize: select whether to normalize the value range of x here. If not the data is assumed to be normalized already.
     @returns: 1) the quantized tensor (not done in-place), 
               2) the number of bits used, 
               3) the numpy data type to use to encode the data
     @comments: 1) fixed32 and ufixed32 are of limited use if input data is float32
                2) quantization is done based on the (symmetric; except for ufixed) 
                   maximum range covered by the data in the tensor
  """
  if quant[:5] == 'float':
    numBit = int(quant[5:])
    if x is not None:
      x = x.clone() # perform no quantization; comes with dtype conversion
    if numBit == 32:
      dtype = np.float32
    elif numBit == 16:
      dtype = np.float16
    else:
      assert(False)
      
  elif quant[:5] == 'fixed':
    numBit = int(quant[5:])
    if numBit > 16:
      assert(numBit <= 32)
      dtype = np.int32
    elif numBit > 8:
      dtype = np.int16
    else:
      dtype = np.int8
      
    if x is not None:
      if normalize:
        x = x.div(x.abs().max().item()/safetyFactor) # now op in [-1.0, 1.0]
      x = x.mul(2**(numBit-1)-1) # now quantized to use full range
      
  elif quant[:6] == 'ufixed':
    numBit = int(quant[6:])
    if numBit > 16:
      assert(numBit <= 32)
      dtype = np.uint32
    elif numBit > 8:
      dtype = np.uint16
    else:
      dtype = np.uint8
    assert(x.ge(0).all().item())
    if x is not None:
      if normalize:
        x = x.div(x.max().item()/safetyFactor) # now op in [0, 1.0]
      x = x.mul(2**(numBit)-1) # now quantized to use full range
      
  else:
    assert(False)
    
  return x, numBit, dtype

getWW = lambda qm: quantize(None, quant=qm)[1]
