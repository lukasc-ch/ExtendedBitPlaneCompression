# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

def groupedBarPlot(data, groupNames, legend=None, xtickRot=None):
  import matplotlib.pyplot as plt
  barWidth = 1/(1+len(data[0]))
  xpos = list(range(len(data)))
  for s in range(len(data[0])):
    plt.bar([x + s*barWidth for x in xpos], 
            [data[i][s] for i in range(len(data))], width=barWidth)
  plt.xticks([x + (len(data[0])-1)*barWidth/2 for x in xpos], groupNames, rotation=xtickRot)
  if legend is not None:
    plt.legend(legend)

if __name__ == "__main__":
  data = [[1,2,3],[6,5,7],[-1,-2,3],[-6,-7,-1]]
  groupNames = ['small', 'medium', 'neutral', 'negative']
  legend = ['a', 'b', 'max']
  groupedBarPlot(data, groupNames, legend)