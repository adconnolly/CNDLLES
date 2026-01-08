import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pickle import load
from brokenaxes import brokenaxes

files=["coarse4x1026_Re900.nc", "coarse4x2052_Re1800.nc" ]
loadPaths=['../baseline/trainedModels/baseline_noDataAug_ReExtrap_',
           '../baseline/trainedModels/baseline_dataAug_ReExtrap_',
           '../training_C4DNN/trainedModels/C4_ReExtrap_']
nmodel = len(loadPaths)

component_text = [r'$\tau_{11}$', r'$\tau_{12}$', r'$\tau_{13}$',
                  r'$\tau_{22}$', r'$\tau_{23}$', r'$\tau_{33}$']
ncomponent=len(component_text)

model_text = ['Baseline','Baseline + aug.','Equivariant']
colors = ['tab:gray','tab:olive','tab:cyan']
x=np.arange(0,(nmodel+1)*ncomponent,nmodel+1)
offset=[-1,0,1]
task_text=['Original','Rotated']
task_suffix = ['','_270']
fs=20
ls=12
width = 1  # the width of the bars
multiplier = 0

try:
    del fig,ax
    fig = plt.figure(figsize=(15,8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
except:
   fig = plt.figure(figsize=(15,8))
   gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

fig.subplots_adjust(bottom=0.175)

ax0 = fig.add_subplot(gs[0])
ax0.set_ylim(0, 1)
ax1 = brokenaxes(ylims=((-3.5, -0.35), (-0.35, 1.0)), subplot_spec=gs[1], height_ratios=[10, 1], hspace=0.05, d = 0.01)

ax = [ax0, ax1]

for itask in range(len(task_text)):
    for imodel in range(nmodel):
    
        if itask*imodel < 2:
            loadFile = loadPaths[imodel]
            for n in range(len(files)):
                isep = files[n].index('_')
                loadFile=loadFile+files[n][6:isep]+files[n][isep+1:-3]+'_'
            loadDict=load( open( loadFile[:-1]+'.pkl', "rb" ) )
            print('stats from '+loadFile)
            
            r2s = loadDict["Rsquared"+task_suffix[itask]]
            print(r2s)
            
        else: # C4 270 test are done elsewhere
            # will print R2 on unrotated data as well, to confirm match
            C4test270file = '../analysis/C4_ReExtrap_test270.pkl'
            loadDict=load( open( C4test270file, "rb" ) )
            print('stats from '+C4test270file)
            print("Compare to original unrotated (above) for FP accuracy:")
            print(loadDict["Rsquared"])
            print()
            r2s = loadDict["Rsquared"+task_suffix[itask]]
            print(r2s)

        r2 = np.mean(r2s, axis = 0) # Average of training runs
        yerr = np.std(r2s, axis = 0)
        rects = ax[itask].bar(x + offset[imodel], r2, width, label=model_text[imodel], color=colors[imodel])
        bars = ax[itask].errorbar(x + offset[imodel], r2, yerr=yerr, linestyle=' ',color='k', capsize=5, capthick=1)

for itask in [0,1]:
    ax[itask].set_title(task_text[itask],fontsize=fs)
    ax[itask].tick_params(axis='y',labelsize=ls)

ax[0].set_xticks(x, labels = component_text, fontsize=fs) 
ax[0].set_ylabel(r'R$^2$',fontsize=fs)

ax[1].axs[1].set_xticks(x,labels = component_text, fontsize=fs)
ax[1].axs[1].tick_params(labelbottom=True)
ax[1].big_ax.tick_params(labelbottom=False)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles,labels, loc='lower left', bbox_to_anchor=(0.1, 0.0),fontsize=32,ncol=3,frameon=False,framealpha=0,columnspacing=0.3)

fmt="png"
plt.savefig('C4vsBaseline_rotStats.'+fmt,format=fmt)#,transparent=True)