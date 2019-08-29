# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

import csv
import os

def readTable(tableName='totalComprRate-alexnet-after ReLU'):
    filename = './results/%s.csv' % tableName
    with open(filename, 'r') as csvfile:
        cr = csv.reader(csvfile)
        tbl = [row for row in cr]
    return tbl

def parseTable(tbl):
    comprMethods = tbl[0][1:]
    quantMethods = [r[0] for r in tbl[1:]]
    data = [[float(v) for v in r[1:]] for r in tbl[1:]]
    return comprMethods, quantMethods, data

def writeTable(tableName='test', table=None, tableHeader=None, tableRowHeader=None):
    if table == None:
        print('No data to be written for table %s.' % tableName)
        return
    if tableRowHeader != None:
        table = [[rowheader] + row for rowheader, row in zip(tableRowHeader,table)]
    filename = './results/%s.csv' % tableName
    with open(filename, 'w', newline='') as csvfile:
        cw = csv.writer(csvfile)
        if tableHeader != None:
            cw.writerow(tableHeader)
        cw.writerows(table)