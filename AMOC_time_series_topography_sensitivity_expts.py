import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os

##----------------------------------------------------------------------------------------------##
# figure configuration
gs_kw = dict(width_ratios=[0.6,0.4], height_ratios=[0.6,0.4])
fig,axs = plt.subplot_mosaic([['left', 'upper right'],
                       ['left', 'lower right']],
                       gridspec_kw=gs_kw, 
                       figsize=(18,6))
##----------------------------------------------------------------------------------------------##
# file path
work_path = os.getcwd()
# data_path = os.path.abspath(os.path.join(work_path,'..','..','Simulations','data','xpgx'))
data_path = r"C:\Users\nd20983\docs\Simulations\data\xpgx"
file_path = os.path.abspath(os.path.join(data_path,'AMOC_time_series.xls'))
if not os.path.exists(file_path):
    print("🚫 File not found!")
    print("Tried path:", file_path)
    print("Does directory exist?", os.path.isdir(data_path))
    print("Directory contents:", os.listdir(data_path) if os.path.isdir(data_path) else "N/A")
else:
    print("✅ File found:", file_path)
# read data
df_whole = pd.read_excel(file_path,skiprows=1,
                  sheet_name = ['xpgx','brad','CO2']
                )
df = df_whole['xpgx']
df_bradshaw = df_whole['brad']
# plot1
# time series for different experiments
ax0 = axs['left']
ax0.plot(df['xpgxa'][0:1000], c='red', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxa'][0:1000], np.ones(50)/50, mode='valid'), label='xpgxa', markersize=1, c='red', marker='o',)
ax0.plot(df['xpgxs'], c='tomato', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxs'], np.ones(50)/50, mode='valid'), label='xpgxs', markersize=1, c='tomato', marker='o',)
ax0.plot(df['xpgxx'][0:1000], c='maroon', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxx'][0:1000], np.ones(50)/50, mode='valid'), label='xpgxx', markersize=1, c='maroon', marker='o',)
ax0.plot(df['xpgxt'], c='violet', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxt'], np.ones(50)/50, mode='valid'), label='xpgxt', markersize=1, c='violet', marker='o',)

ax0.plot(df['xpgxb'][0:1000], c='orange', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxb'][0:1000], np.ones(50)/50, mode='valid'), label='xpgxb', markersize=1, c='orange', marker='o',)
ax0.plot(df['xpgxc'][0:1000], c='gold', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxc'][0:1000], np.ones(50)/50, mode='valid'), label='xpgxc', markersize=1, c='gold', marker='o',)
ax0.plot(df['xpgxr'], c='olive', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxr'], np.ones(50)/50, mode='valid'), label='xpgxr', markersize=1, c='olive', marker='o',)

ax0.plot(df['xpgxw'], c='blue', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxw'], np.ones(50)/50, mode='valid'), label='xpgxw', markersize=1, c='blue', marker='o',)
ax0.plot(df['xpgxh'], c='black', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpgxh'], np.ones(50)/50, mode='valid'), label='xpgxh', markersize=1, c='black', marker='o',)
ax0.plot(df['xpecy'], c='slategray', marker='x', alpha=0.1)
ax0.plot(np.convolve(df['xpecy'], np.ones(50)/50, mode='valid'), label='xpecy', markersize=1, c='slategray', marker='o',)

# fonts & labels config
font_legend = font_manager.FontProperties(#family='Times New Roman',
                                          weight='normal',
                                          style='normal', size=13)
font_label = {#'family': 'Times New Roman',
              'weight': 'normal',
              'size': 12
             }
font_title = {#'family': 'Times New Roman',
              'weight': 'normal',
              'size': 20
             }
ax0.set_xlabel('Model year (yr)', fontdict=font_label)
ax0.set_ylabel('AMOC strength (Sv)', fontdict=font_label)
yminorLocator = MultipleLocator(0.025)

# add annotations for each experiment
ax0.text(x=800, y=17, s='GET_AC')
ax0.text(x=800, y=14, s='GET_CTR')
ax0.text(x=865, y=12.5, s='GET_TP')
ax0.text(x=800, y=11, s='GET_AD')
ax0.text(x=900, y=6.0, s='GET_RC')
ax0.text(x=400, y=4.0, s='SCO_CTR')
ax0.text(x=925, y=3.2, s='ROB_CTR')
ax0.text(x=400, y=1.4, s='GET_LC')
ax0.text(x=600, y=1.0, s='GET_LC1x')
ax0.text(x=800, y=1.0, s='GET_HF')

# add PI control AMOC value
ax0.axhline(y=17.9, color='orange', linewidth=1)
# add rectangles for different groups
rects = [
    (180, 1.5, 200, 2),
    (180, 7.0, 200, 2),
    (180, 15., 200, 2)
    ]
for x, y, w, h in rects:
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue',
                             facecolor='none', alpha=0.7)
    ax0.add_patch(rect)
ax0.text(280, 2.5, 'Low Rockies +', ha='center', va='center', fontsize=12)
ax0.text(280, 8.0, 'Low Rockies', ha='center', va='center', fontsize=12)
ax0.text(280, 16.0, 'High Rockies', ha='center', va='center', fontsize=12)
# ax.yaxis.set_minor_locator(yminorLocator)
# ax.legend(prop=font_legend)
# ax0.set_title('AMOC time series', fontdict=font_title, y=1.05)
##----------------------------------------------------------------------------------------------##
ax1 = axs['upper right']
# read data
file_path_wf = os.path.abspath(os.path.join(data_path,'waterflux.xls'))
df_wf = pd.read_excel(file_path_wf, skiprows=2,
                      sheet_name= ['water_budget','linear_regression'])
df1 = df_wf['water_budget'].iloc[0:10]
# labels = list(df['expt'].items())
labels = ['GET_HF','GET_LC','GET_LC1x','ROB_CTR','SCO_CTR','GET_RC','GET_TP','GET_AD','GET_AC','GET_CTR']

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

rects1 = ax1.bar(x - width*1.5, df1['precip7'], width, color='green', label='precip')
rects2 = ax1.bar(x - width/2, df1['evap7'], width, color='orange', label='evap')
rects3 = ax1.bar(x + width/2, df1['runoff7'], width, color='black', label='runoff')
rects4 = ax1.bar(x + width*1.5, df1['waterflux7'], width, color='red', label='waterflux')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Waterflux (Sv)', fontdict=font_label)
# ax1.set_title('Waterflux and components distribution (N.Atl Annual mean)')
ax1.set_xticks(x, labels=labels)
ax1.set_xticklabels(labels, rotation=-30)
#ax.set_xticks(labels)
ax1.legend(loc='center right')

ax1.set_label(rects1)
ax1.set_label(rects2)
ax1.set_label(rects3)
ax1.set_label(rects4)

##----------------------------------------------------------------------------------------------##
ax2 = axs['lower right']
# read data
# data = read_csv('../Simulations/data/xpgx/linear_regression.csv')
data = df_wf['linear_regression'][0:10]
model = LinearRegression(fit_intercept=True)
model.fit(data[["waterflux_djf"]], data["AMOC"])

# calculate r2_score
x_fit = DataFrame(data["waterflux_djf"])
y_pred = model.predict(x_fit)
y_true = DataFrame(data['AMOC'])
r_square = r2_score(y_true, y_pred)

# prep for linear regression visualisation
x_fit_v = DataFrame([data['waterflux_djf'].min(), data['waterflux_djf'].max()])
y_pred_v = model.predict(x_fit_v)

data.plot.scatter("waterflux_djf", "AMOC", ax=ax2)
ax2.plot(x_fit_v[0], y_pred_v, linestyle=":")
ax2.set_xlabel('Net fresh waterflux (Sv)', fontdict=font_label)
ax2.set_ylabel('AMOC (Sv)', fontdict=font_label)
ax2.text(x=0.15, y=10, s="y = -103.57x + 18.25")
ax2.text(x=0.15, y=7, s=r'$R^2 = {:.2f}$'.format(r_square))
'''
# set the x-spine
ax.spines['left'].set_position('zero')

# turn off the right spine/ticks
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()

# set the y-spine
ax.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax.spines['top'].set_color('none')
ax.xaxis.tick_bottom()
'''
# get positions for each axes
pos0 = ax0.get_position()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
# modify the positions
pos0_1 = [pos0.x0, pos0.y0, pos0.x1+0.001, pos0.y1]
pos1_1 = [pos1.x0, pos1.y0, pos1.x1, pos1.y1]

plt.tight_layout()
plt.show()
# plt.savefig('../Simulations/data/figures/AMOC_time_series', dpi=100, bbox_inches='tight', facecolor='w')