import matplotlib.pyplot as plt
import numpy as np
from torch import from_numpy


def plot_losses(losses, labels=["Train"], cutStartPct=0.25, zoomEndEpochs=10):
    nlosses = len(losses)
    nepochs = len(losses[0])
    epochs = range(nepochs)
    plotEpochs0 = slice(int(cutStartPct * nepochs), nepochs)
    plotEpochs1 = slice(int(nepochs - zoomEndEpochs), nepochs)

    fig, ax = plt.subplots(2, nlosses + 1, figsize=(18, 8))
    for il in range(nlosses):
        loss = losses[il]
        ax[0, il].plot(epochs[plotEpochs0], loss[plotEpochs0], label=labels[il])
        ax[0, il].legend()
        ax[0, il].set_xlabel('Epoch')
        ax[0, il].set_ylabel('Loss')
        ax[0, il].set_title(labels[il])
        
        ax[1, il].plot(epochs[plotEpochs1], loss[plotEpochs1], label=labels[il])
        ax[1, il].legend()
        ax[1, il].set_xlabel('Final Epochs '+labels[il])
        
        ax[0, -1].plot(epochs, loss, label=labels[il])

        ax[1, -1].plot(epochs[plotEpochs1], loss[plotEpochs1], label=labels[il])

    ax[0, -1].legend()
    ax[0, -1].set_xlabel('Epoch')
    ax[0, -1].set_title('All epochs, all data')
    
    ax[1, -1].legend()
    ax[1, -1].set_xlabel('Final Epochs, all data')

    return

def plot_scatter(model, device, u, Ri, y, weights=None, scales=None, text = "Predictions vs DNS",
                y_text = [r"$\tau_{11}$", r"$\tau_{12}$", r"$\tau_{13}$",
                          r"$\tau_{22}$", r"$\tau_{23}$", r"$\tau_{33}$"], 
                 return_predictions = False):
    
    model.eval()
    yp = model(from_numpy(u).float().to(device), from_numpy(Ri).float().to(device)).cpu().detach().numpy().squeeze()
    
    noutvars = yp.shape[1] 
    assert len(y_text) == noutvars
    
    try:
        assert weights==None
        weights = np.ones(noutvars)
    except:
        assert len(weights) == noutvars
        
    fig, ax = plt.subplots(1, noutvars, figsize = (20, 6))
    for i in range(noutvars):

        ax[i].scatter(y[:,i] * weights[i], yp[:,i] * weights[i])
        xmin, xmax=ax[i].get_xlim()
        # ymin, ymax=ax[i].get_ylim()
        ax[i].plot([xmin,xmax],[xmin,xmax])
        ax[i].set_xlim([xmin,xmax])
        ax[i].set_xlabel('True')
        ax[i].set_ylabel('Predicted')
        ax[i].set_title(y_text[i])
    fig.suptitle(text)

    if return_predictions:
        return yp
    else:
        return
    


