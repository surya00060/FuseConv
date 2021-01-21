import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from operator import add
# !sudo apt-get install -y msttcorefonts 
# !rm ~/.cache/matplotlib -fr
sns.set()
# sns.set_context("talk", font_scale=1.3)

matplotlib.rcParams['figure.dpi'] = 500
# sns.set_context("talk", font_scale=1) 
x = ['MobileNet-V1', 'MobileNet-V2', 'MnasNet-B1', 'MobileNet-V3\nSmall', 'MobileNet-V3\nLarge']
baseline = [2536900, 2979156, 2748684, 700161, 2018189]
v2 = [375260, 411910, 384202, 167935, 370097]
v1 = [621300, 576770, 542818, 231735, 558553]
h2 = [1072300, 1375054, 1394346, 414921, 1102045]
h1 = [1151124, 1453926, 1458602, 437559, 1144825]
baselineSpeedup = [1, 1, 1, 1, 1]
v2Speedup = [6.760379470233971, 7.232541089072856, 7.154267807039005, 4.169238098073659, 5.4531352591347675]
v1Speedup = [4.08321261870272, 5.165240910588276, 5.063730384769849, 3.021386497507929, 3.613245296328191]
h2Speedup = [2.3658491093910285, 2.1665738218280883, 1.9713069783253223, 1.687456166354559, 1.8313126959425432]
h1Speedup = [2.2038459801029253, 2.049042385926106, 1.8844647134722152, 1.6001522080450865, 1.7628799161443889]
# labels = np.arange(len(x))  # the label locations
width = 0.33  # the width of the bars
r1 = [0, 2, 4, 6, 8]
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]
r4 = [x + width for x in r3]
r5 = [x + width for x in r4]
fig, ax = plt.subplots()
rects1 = ax.bar(r1,  baseline, width, label='Baseline')
rects2 = ax.bar(r2,  v1, width, label='Full FuSeConv')
rects3 = ax.bar(r3,  v2, width, label='Half FuSeConv')
rects4 = ax.bar(r4,  h1, width, label='50% Full FuSeConv')
rects5 = ax.bar(r5,  h2, width, label='50% Half FuSeConv')

def annotate(rects, speed):
    i = 0
    for rect in rects:
        if speed[i]==1:
            ax.annotate('1x',(rect.get_x() + rect.get_width() / 2., rect.get_height()), size=9, ha='center', va='bottom', rotation='vertical', weight='bold')
        else:
            ax.annotate('%1.2fx'%(speed[i]),(rect.get_x() + rect.get_width() / 2., rect.get_height()), size=9, ha='center', va='bottom', rotation='vertical', weight='bold')
        i+= 1

annotate(rects1, baselineSpeedup)
annotate(rects2, v1Speedup)
annotate(rects3, v2Speedup)
annotate(rects4, h1Speedup)
annotate(rects5, h2Speedup)
plt.xticks(r3, x)
ax.legend(ncol=2)
plt.xlabel('Networks')
plt.ylabel('Latency Cycles (Millions)')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9.5)
plt.tight_layout()
matplotlib.rcParams["savefig.edgecolor"] = 'grey'
plt.savefig('Speedup.png', transparent=True)
plt.show()

Speedup =[[2.023718468729478, 4.08321261870272, 6.777745410408133, 8.636542215565186], [3.350260103199137, 5.165240910588276, 6.83670060674575, 7.891341918938507], [3.31582778991125, 5.063730384769849, 6.456343997050969, 7.171005322794138], [2.3800222973367635, 3.021386497507929, 3.618686195106574, 3.9708378515144944], [2.480129194282618, 3.613245296328191, 4.819822954009899, 5.639453565269347]]
speedup = np.array(Speedup)
# print(speedup)
# sns.set_context("talk", font_scale=1.0)
matplotlib.rcParams['figure.dpi'] = 500
fig, ax = plt.subplots()
networks = ['MobileNet-V1', 'MobileNet-V2', 'MnasNet-B1', 'MobileNet-V3 Small', 'MobileNet-V3 Large']
x = ['32x32', '64x64', '128x128', '256x256']
r = [0, 1, 2, 3]
for i in range(len(networks)):
    sns.lineplot(r, speedup[i], marker="o", label=networks[i])
plt.xticks(r, x)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.legend(fontsize=10)
plt.xlabel("Systolic array dimensions")
plt.ylabel("Speedup")
# print(matplotlib.rcParams["figure.figsize"])
plt.tight_layout()
matplotlib.rcParams["savefig.edgecolor"] = 'grey'
plt.savefig('ScaleUp.png', transparent=True)
plt.show()
# print(matplotlib.rcParams["figure.figsize"])
####
speedup = [1, 5.1, 7.23, 2.0, 2.1]