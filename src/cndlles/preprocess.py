import xarray as xr
import numpy as np

def preprocess(files, filemaskpercents, fileUgs, fileRes, fileB0s, size=3, irun='', reshape=True):
    # Any of the below could be changed to inputs of the function
    path="/glade/u/home/adac/work/DNStoLES/coarseData/"
    width = size
    size=int((width-1)/2)  # Recentering on origin useful here, but size as extent more natural to input
    vsize=1 ; depth = 3 # Similar to above vsize = 1 in each direction from center, so 3 planes total

    maskDict = {}
    for ifile, file in enumerate(files):
        ds=xr.open_dataset(path+file,decode_times=0)
        print(file)

        # Getting input variables, removing sponge layers, wrapping periodic varaibles as reordering to x,y,z,t 
        b=np.transpose(np.pad(cut_sponge(ds['b'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        u=np.transpose(np.pad(cut_sponge(ds['u'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap') , [2, 1, 0,3])
        v=np.transpose(np.pad(cut_sponge(ds['v'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        w=np.transpose(np.pad(cut_sponge(ds['w'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        inputFields = np.array([u, v, w, b])
        
        # Getting output fields
        tau_12 = np.transpose(cut_sponge(ds['tau12'].values), [2, 1, 0,3])
        tau_13 = np.transpose(cut_sponge(ds['tau13'].values), [2, 1, 0,3])
        tau_23 = np.transpose(cut_sponge(ds['tau23'].values), [2, 1, 0,3])
        tau_11 = np.transpose(cut_sponge(ds['tau11'].values), [2, 1, 0,3])
        tau_22 = np.transpose(cut_sponge(ds['tau22'].values), [2, 1, 0,3])
        tau_33 = np.transpose(cut_sponge(ds['tau33'].values), [2, 1, 0,3])
        outputFields = np.array([tau_11,tau_12, tau_13, tau_22, tau_23, tau_33])

        # Precompute scaling factors
        Ug=fileUgs[ifile]
        Re=fileRes[ifile]
        b0=fileB0s[ifile]
        hvelScale = Ug/np.sqrt(Re)
        vvelScale = Ug/np.sqrt(Re)
        bScale = -b0  # bTop - b0 = 0 - b0
        tijScale=Ug**2/Re
        # thhScale=tijScale
        # ti3Scale=tijScale
        

        
        # Random mask is per file for data balancing
        nx=outputFields.shape[1] # sizes based on outputs b/c no padding from periodic BC
        ny=outputFields.shape[2]
        nz=outputFields.shape[3]
        nt=outputFields.shape[4]
        mask = np.random.rand(nx,ny,nz,nt) < filemaskpercents[ifile]
        mask[:,:,0,:] = False # Can't use bottom layer but useful for mask size
        mask[:,:,-1,:] = False # Can't use top " "
        maskDict["mask_"+file+'_'+str(irun)]=mask
        
        nsamples = np.sum(mask)
        x3d = np.empty((nsamples, inputFields.shape[0], width, width, depth))
        y = np.empty((nsamples, outputFields.shape[0]))
        
        s = 0 # Sample index as a counter
        for i in range(size, inputFields.shape[1] - size):
            for j in range(size, inputFields.shape[2] - size):
                for k in range(vsize, inputFields.shape[3] - vsize):
                    for it in range(inputFields.shape[4]):
                        if mask[i-size,j-size,k,it]:                            
                            
                            # Call to scale to subtract input-box mean
                            scaledInput=[scale(inputFields[0, i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it], sd=hvelScale),
                                         scale(inputFields[1, i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it], sd=hvelScale),
                                         scale(inputFields[2, i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it], sd=vvelScale),
                                         scale(inputFields[3, i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it], sd=bScale)]
                            x3d[s] = scaledInput
                            
                            # No subtracting mean from outputs
                            y[s] = outputFields[:,i-size,j-size,k,it]/tijScale
                            # y[s] = [outputFields[0,i-size,j-size,k,it]/thhScale, outputFields[1,i-size,j-size,k,it]/thhScale, outputFields[2,i-size,j-size,k,it]/ti3Scale, 
                            #         outputFields[3,i-size,j-size,k,it]/thhScale, outputFields[4,i-size,j-size,k,it]/ti3Scale, outputFields[5,i-size,j-size,k,it]/ti3Scale]
                            
                            s += 1

    print("output shape is " + str(y.shape))
    print("input shape was " + str(x3d.shape))
    if reshape:
        size=int(2*size+1)
        x=myreshape(x3d,size=size)                    
        print("input shape to do 3rd dimension as channel in R2Conv is "+str(x.shape))
    else:
        x=x3d

    return x, y, maskDict

def cut_sponge(x):
    # Cuts the input 75% along the z-axis to remove sponge layer
    return x[:int(0.75*x.shape[0])]

def scale(x,mean=None,sd=None):

    if mean==None:
        mean=np.mean(x)
        
    if sd==None:

        sd=np.std(x)
    
    return (x - mean) / sd

def myreshape(xtest,nvars=4,size=3):
    # Resulting order is, with each entry being a 3*3 horiz. plane:
    # u(k=0),v(k=0), u(k=1),v(k=1), u(k=2),v(k=2), w(k=0),w(k=1),w(k=2), b(k=0),b(k=1),b(k=2)
    xnew=np.reshape(xtest.copy(),(xtest.shape[0],nvars*3,size,size))
    # Have to make this loop explicit because grouping u,v as 
    #2d vector is inherent to using irrep(1) representation type
    for v in range(2):
        for k in range(xtest.shape[-1]):
            xnew[:,2*k+v,:,:]=xtest[:,v,:,:,k]
    if nvars>2:   
        for v in range(2,nvars):
            for k in range(xtest.shape[-1]):
                xnew[:,3*v+k,:,:]=xtest[:,v,:,:,k]
        
    return xnew