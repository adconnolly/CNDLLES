from cndlles.torch_util import *
from cndlles.preprocess import preprocess
from cndlles.plot_helpers import plot_losses
from cndlles.plot_helpers import plot_scatter

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import pickle
import torch
import torch.utils.data as Data

size=3  # Horizontal size of input planes, always 3 in vertical
Nhid = [32,16,8]  # Multiplicity of hidden layers
Ri_pct = 0.25  # Percent of first hidden layer features that are due to tranforming Ri input
Nruns = 5
BATCH_SIZE = 1024  # Number of sample in each batch
n_epochs = 500  # Max number of epochs
patience = 20
LR0 = 1e-3
LRsteps = 1

saveFile='C4_ReExtrap_'
plotLosses = True
plotQuickPlots = True

## Training datasets
files=["coarse4x1026_Re900.nc", "coarse4x2052_Re1800.nc" ]

# Scaling factors based on exogenous forcing
# Note, lower Re in file names uses a viscous length scale from the
# DNS while LES scaling factors use the BL depth as length scale
fileRes=[20000.,40000.] # Re = 20k, 40k, 60k for Re900, Re1800, Re2700
fileUgs=[0.025,0.05] # 0.025, 0.05, 0.075
filemaskpercents=[0.34,1] # s.t. same # of samples from each DNS

# Test (reported statistics) and Validation (early stopping) data
testFiles=["coarse4x3078_Re2700.nc"]
testfileUgs=[0.075]
testfileRes=[60000.]
testfilemaskpercents=[0.4] 

# Half testFiles' data, after per file masking above, used for early stopping
valMaskPercent = 0.5 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

plotPath='quickPlots/'
savePath='trainedModels/'
# sizeStr = str(size)
for n in range(len(files)):
    isep = files[n].index('_')
    saveFile=saveFile+files[n][6:isep]+files[n][isep+1:-3]+'_'
print(saveFile)

y_text = ["tau_11", "tau_12", "tau_13","tau_22", "tau_23", "tau_33"]

r=np.empty((Nruns,len(y_text))) 
r2=np.empty((Nruns,len(y_text)))
auxDataDict=dict()
for irun in range(Nruns):

    print("Train Files:")
    u_train, Ri_train, y_train, trainMask = preprocess(files, filemaskpercents, fileUgs, fileRes, size, irun)
    auxDataDict.update(trainMask)
   
    torch_dataset = Data.TensorDataset(torch.from_numpy(u_train).float(),torch.from_numpy(Ri_train).float(),torch.from_numpy(y_train).float())
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Test Files:")
    utest, Ritest, ytest, testMask = preprocess(testFiles, testfilemaskpercents, testfileUgs, testfileRes, size, irun)
    auxDataDict.update(testMask) 

    # Note, mask_[irun]_* uses original spatial indices, but valMask is by sample
    mask =  np.random.rand(ytest.shape[0]) > valMaskPercent
    auxDataDict["valMask_"+str(irun)]=mask
    u_test=utest[mask]
    Ri_test=Ritest[mask]
    u_val=utest[~mask]
    Ri_val=Ritest[~mask]
    del utest, Ritest
    y_test=ytest[mask]
    y_val=ytest[~mask]
    del ytest

    torch_dataset_test = Data.TensorDataset(torch.from_numpy(u_test).float(),torch.from_numpy(Ri_test).float(),torch.from_numpy(y_test).float())
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=BATCH_SIZE,shuffle=True)

    torch_dataset_val = Data.TensorDataset(torch.from_numpy(u_val).float(),torch.from_numpy(Ri_val).float(),torch.from_numpy(y_val).float())
    loader_val = Data.DataLoader(dataset=torch_dataset_val,batch_size=BATCH_SIZE,shuffle=True)

    # Lossweights to balance components
    numerator=1
    LossWeights = np.array([numerator/np.std(y_train[:,i]) for i in range(y_train.shape[1])])
    print('Lossweights:')
    print(LossWeights)
    weights=torch.from_numpy(LossWeights).to(device)
    
    model=CNDNN(Nhid, Ri_pct = Ri_pct,N=4,size=size).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR0) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,range(1,LRsteps+1), gamma=0.10)
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    criterion = torch.nn.L1Loss() # MSE loss function

    validation_loss = list()
    train_loss = list()
    test_loss = list()
    min_loss = np.inf

    for epoch in range(n_epochs):
        print('Epoch: '+str(epoch))
        print('LR: ',scheduler.get_last_lr())
        
        train_model(model, criterion, loader, optimizer, scheduler, weights, device)
        
        train_loss.append(test_model(model, criterion, loader, weights, device, 'train'))
        validation_loss.append(test_model(model, criterion, loader_val, weights, device))
        test_loss.append(test_model(model, criterion, loader_test, weights, device, 'test'))
        
        if validation_loss[-1] < min_loss:
            torch.save(model.state_dict(),savePath+saveFile+str(irun)+'.pt')
            min_loss = validation_loss[-1]
        if early_stopper.early_stop(validation_loss[-1]):
            print('ES epoch: '+str(epoch-patience))
            break

    # Reloading best model weights
    model.load_state_dict(torch.load(savePath+saveFile+str(irun)+'.pt'))

    auxDataDict["train_loss_"+str(irun)]=train_loss
    auxDataDict["validation_loss_"+str(irun)]=validation_loss
    auxDataDict["test_loss_"+str(irun)]=test_loss

    zoomEndEpochs = np.max((2 * early_stopper.patience, n_epochs))
    if plotLosses:
        plot_losses(
                [train_loss, validation_loss, test_loss],
                ["Train", "Validation", "Test"], 
                cutStartPct = 0., zoomEndEpochs = zoomEndEpochs 
                  )
        plt.savefig(plotPath+saveFile+str(irun)+'_lossCurve.png')
    if plotQuickPlots:
        plot_scatter(model, device,
                                u_val, Ri_val, y_val,
                                # weights = LossWeights,
                                text = "Validation data",
                   )
        plt.savefig(plotPath+saveFile+str(irun)+'_valStats.png')
    
        plot_scatter(model, device,
                                u_train, Ri_train, y_train,
                                # weights = LossWeights,
                                text = "Train data",
                   )
        plt.savefig(plotPath+saveFile+str(irun)+'_trainStats.png')
    
        y_pred = plot_scatter(model, device,
                                u_test, Ri_test, y_test,
                                # weights = LossWeights,
                                text = "Test data",
                                return_predictions = True
                   )
        plt.savefig(plotPath+saveFile+str(irun)+'_TestStats.png')
    
    # else: # Not implemented b/c always want to plot and return predictions then
    #     yp = get_predictions( <Test Data> ) # to save on inference

    r2[irun]=[r2_score(y_test[:,i], y_pred[:,i]) for i in range(y_pred.shape[1])]
    r[irun]=[np.corrcoef(y_test[:,i], y_pred[:,i])[0, 1] for i in range(y_pred.shape[1])]

    torch.cuda.empty_cache()
    del model, optimizer, scheduler

print(r)
print(r2)
auxDataDict["r"] = r
auxDataDict["Rsquared"] = r2
pickle.dump(auxDataDict,open(savePath+saveFile[:-1]+'.pkl', "wb" ))

for v in range(r2.shape[1]):
        print(y_text[v]+r' avg. R$^2$ is '+str(np.mean(r2[:,v]))+' +/- '+str(np.std(r2[:,v])))
print(r'Overall avg. R$^2$ is '+str(np.mean(r2))+' +/- '+str(np.std(np.mean(r2,axis=1))))