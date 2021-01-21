import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from operator import add    

## Variant-2
otherConv = [265913, 265913, 265913, 132956, 132956] 
pointConv = [9329664, 5327220, 5309432, 1068178, 3987903] 
depthConv = [4162560, 4968416, 5652608, 1407400, 3794168]
linear = [17384, 21480, 21480, 32890, 51546]
fusepointconv =  [18021376, 7264628, 7405192, 1360322, 5318975]
fusedepthconv =  [837760, 1010240, 869288, 224896, 661472]
linearfuse = [17384, 21480, 21480, 57764, 130032]

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase, update_from_first_child

def subtitle_handler_factory(inherit_from):
    """Class factory to subclass Handlers and add our custom functionality
    """
    class SubtitleHandler(inherit_from):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            handlebox.set_visible(False)
            return inherit_from.legend_artist(self, legend,
                                              orig_handle, fontsize,
                                              handlebox)
    
    #HandlerPatch class needs a special unpdate_func
    if inherit_from is matplotlib.legend_handler.HandlerPatch:
        return SubtitleHandler(update_func=update_from_first_child)
    return SubtitleHandler()

def subtitle_handler_map(subtitles):
    defaults_handler_map = Legend.get_default_handler_map()
    handler_map = {}
    
    for orig_handle in subtitles:
        handler = Legend.get_legend_handler(defaults_handler_map, orig_handle)
        
        #Subclass the Handler
        new_handler = subtitle_handler_factory(type(handler))
        handler_map[orig_handle] = new_handler
    return handler_map

### Not edited!
# sns.set_context("talk", font_scale=1.0)
matplotlib.rcParams['figure.dpi'] = 500 
fig, ax = plt.subplots()
otherConv = list(map(add, otherConv, linear))
otherConvfuse= list(map(add, otherConv, linearfuse))
total = list(map( add, list(map(add, otherConv, pointConv)), depthConv))
totalFriendly = list(map( add, list(map(add, fusedepthconv, fusepointconv)), otherConvfuse))

percentoc = [i / j * 100 for i,j in zip(otherConv, total)]
percentpc = [i / j * 100 for i,j in zip(pointConv, total)]
percentdc = [i / j * 100 for i,j in zip(depthConv, total)]

percentocf = [i / j * 100 for i,j in zip(otherConvfuse, totalFriendly)]
percentpcf = [i / j * 100 for i,j in zip(fusepointconv, totalFriendly)]
percentdcf = [i / j * 100 for i,j in zip(fusedepthconv, totalFriendly)]

print(percentdc, percentdcf) 
names = ['MobileNet-V1', 'MobileNet-V2', 'MnasNet-B1', 'MobileNet-V3\nSmall', 'MobileNet-V3\nLarge']
barWidth = 0.25

r1 = [0,1,2,3,4]
r2 = [x + barWidth for x in r1]
a = plt.bar(0, 0, label='Baseline')
plt.bar(r1, percentoc, color='#b5ffb9', edgecolor='white', width=barWidth, label='Standard')
plt.bar(r1, percentpc, bottom=percentoc, color='#f9bc86', edgecolor='white', width=barWidth, label='Point-wise')
plt.bar(r1, percentdc, bottom=[i+j for i,j in zip(percentoc, percentpc)], color='#a3acff', edgecolor='white', width=barWidth, label='Depthwise')
b = plt.bar(0, 0, label='FuSeConv')
plt.bar(r2, percentocf, color='lightgreen', edgecolor='white', width=barWidth, label='Standard')
plt.bar(r2, percentpcf, bottom=percentocf, color='orange', edgecolor='white', width=barWidth, label='Point-wise')
plt.bar(r2, percentdcf, bottom=[i+j for i,j in zip(percentocf, percentpcf)], color='blue', edgecolor='white', width=barWidth, label='FuSe')

subtitles = [a, b]
handler_map = subtitle_handler_map(subtitles)

plt.xticks(r1, names)
# plt.legend(ncol=2, fontsize=10)
plt.legend(fontsize=10, bbox_to_anchor=(1, 1), handler_map=handler_map)
plt.xlabel("Networks")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
plt.tight_layout()
plt.savefig('breakdown.png', transparent=True)
plt.show()

## Full Variant
mv2baslinepointwise = [21756, 49784, 8967, 19845, 11319, 19845, 3059, 6057, 3683, 6057, 3683, 6057, 1216, 4224, 1984, 4224, 1984, 4224, 1984, 4224, 3840, 7488, 5376, 7488, 5376, 7488, 2032, 4080, 3184, 4080, 3184, 4080, 5360, 8640]
mv2baslinedepthwise = [457856, 349632, 515088, 133776, 172992, 172992, 47232, 89088, 89088, 89088, 89088, 133632, 133632, 37440, 55680, 55680, 55680]
mv2fullvariantpw    = [28028, 49784, 13671, 19845, 18375, 19845, 4931, 6057, 6179, 6057, 6179, 6057, 1984, 4224, 3520, 4224, 3520, 4224, 3520, 4224, 6912, 7488, 9984, 7488, 9984, 7488, 3760, 4080, 6064, 4080, 6064, 4080, 10160, 8640]
mv2fullvariantfuse  =[7504, 7504, 11256, 11256, 8442, 8442, 4221, 4221, 5628, 5628, 5628, 5628, 2814, 2814, 5628, 5628, 5628, 5628, 5628, 5628, 5628, 5628, 8442, 8442, 8442, 8442, 4221, 4221, 7035, 7035, 7035, 7035, 7035, 7035]

mv2baselinepw = []
mv2fusepw = []
mv2fusedw = []
for i in range(0,34,2):
    mv2baselinepw.append(mv2baslinepointwise[i] + mv2baslinepointwise[i+1])
    mv2fusepw.append(mv2fullvariantpw[i] + mv2fullvariantpw[i+1])
    mv2fusedw.append(mv2fullvariantfuse[i]+mv2fullvariantfuse[i+1])
# print(len(mv2pointwiseConv), len(mv2FuseConv), len(mv2depthwiseConv))
from operator import add
baseline = list(map(add, mv2baselinepw, mv2baslinedepthwise))
friendly = list(map(add, mv2fusepw, mv2fusedw))
from operator import truediv 
speedup = list(map(truediv, baseline, friendly))
# print(speedup)

x = []
color = []
matplotlib.rcParams['figure.dpi'] = 500
fig, ax = plt.subplots(figsize=(4, 4))
for i in range(17):
    if i%2 == 0:
        x.append('MB'+str(i+1))
    else:
        x.append(' ')
    # x.append('MB'+str(i+1))
    if i < 6:
        color.append('blue')
    else:
        color.append('cornflowerblue')

xx = np.arange(len(x))
ax.barh(xx,speedup, color=color)
ax.set_yticks(xx)
ax.set_yticklabels(x)
plt.xlabel('Speedup')
plt.ylabel('Layers')
plt.tight_layout()
plt.savefig('layerwise.png', transparent=True)
# plt.xlim(2,10)
#plt.title('MobileNet-V2 layerwise Speedup')
# ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right', fontsize=10)
# print(plt.xticks())
# plt.xticks(np.array([0., 5.0, 10.0]))
plt.show()