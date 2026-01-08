import xarray as xr
import numpy as np

def preprocess(files, filemaskpercents, fileUgs, fileRes, size=3, irun='', reshape=True, dataAug = False, maskdict = None):
    # Any of the below could be changed to inputs of the function
    path="/glade/u/home/adac/work/DNStoLES/coarseData/"
    Ri_char = 5 # Characteristic Richardson number for scaling
    width = size
    size=int((width-1)/2)  # Recentering on origin useful here, but size as extent more natural to input
    vsize=1; depth = 3 # Similar to above vsize = 1 in each direction from center, so 3 planes total

    maskDict = {}
    us = []
    Ris = []
    ys = []
    for ifile, file in enumerate(files):
        ds=xr.open_dataset(path+file,decode_times=0)
        print(file)

        # Getting input variables, removing sponge layers, wrapping periodic varaibles as reordering to x,y,z,t 
        b=np.transpose(np.pad(cut_sponge(ds['b'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        u=np.transpose(np.pad(cut_sponge(ds['u'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap') , [2, 1, 0,3])
        v=np.transpose(np.pad(cut_sponge(ds['v'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])
        w=np.transpose(np.pad(cut_sponge(ds['w'].values),((0,0),(size,size),(size,size),(0,0)),mode='wrap'), [2, 1, 0,3])

        dx=np.mean(np.diff(ds['x'].values))
        dy=np.mean(np.diff(ds['y'].values))
        dz=np.mean(np.diff(ds['z'].values))

        d11=(u[2:,1:-1,:,:]-u[:-2,1:-1,:,:])/(2*dx)
        d12=(u[1:-1,2:,:,:]-u[1:-1,:-2,:,:])/(2*dy)
        d13=np.gradient(u[1:-1,1:-1,:,:],dz,axis=2)

        d21=(v[2:,1:-1,:,:]-v[:-2,1:-1,:,:])/(2*dx)
        d22=(v[1:-1,2:,:,:]-v[1:-1,:-2,:,:])/(2*dy)
        d23=np.gradient(v[1:-1,1:-1,:,:],dz,axis=2)

        d31=(w[2:,1:-1,:,:]-w[:-2,1:-1,:,:])/(2*dx)
        d32=(w[1:-1,2:,:,:]-w[1:-1,:-2,:,:])/(2*dy)
        d33=np.gradient(w[1:-1,1:-1,:,:],dz,axis=2)
        S2 = 2*(d11**2 + d22**2 + d33**2) + 2*(d12*d21 + d13*d31 + d23*d32) + d12**2 + d21**2 + d13**2 + d31**2 + d23**2 + d32**2

        N2=np.gradient(b,dz,axis=2)
        S2 = np.pad(S2,((size,size),(size,size),(0,0),(0,0)),mode='wrap')
        
        inputFields = np.array([u, v, w, (N2 - Ri_char*S2) / (N2 + Ri_char*S2)])        
        
        # Getting output fields
        tau_12 = np.transpose(cut_sponge(ds['tau12'].values), [2, 1, 0,3])
        tau_13 = np.transpose(cut_sponge(ds['tau13'].values), [2, 1, 0,3])
        tau_23 = np.transpose(cut_sponge(ds['tau23'].values), [2, 1, 0,3])
        tau_11 = np.transpose(cut_sponge(ds['tau11'].values), [2, 1, 0,3])
        tau_22 = np.transpose(cut_sponge(ds['tau22'].values), [2, 1, 0,3])
        tau_33 = np.transpose(cut_sponge(ds['tau33'].values), [2, 1, 0,3])
        # outputFields = np.array([tau_11,tau_12, tau_13, tau_22, tau_23, tau_33])
        third_trace = (tau_11 + tau_22 + tau_33) / 3.0
        outputFields = np.array([tau_11 - third_trace, tau_12, tau_13, tau_22 - third_trace, tau_23, tau_33 - third_trace])

        # Precompute scaling factors
        Ug=fileUgs[ifile]
        Re=fileRes[ifile]
        hvelScale = Ug / np.sqrt(Re)
        vvelScale = Ug /  np.sqrt(Re)
        thhScale = Ug**2 / Re
        th3Scale = Ug**2 / Re
        t33Scale = Ug**2 / Re
      
        # Random mask is per file for data balancing
        nx=outputFields.shape[1] # sizes based on outputs b/c no padding from periodic BC
        ny=outputFields.shape[2]
        nz=outputFields.shape[3]
        nt=outputFields.shape[4]
        if maskdict==None:
            mask = np.random.rand(nx,ny,nz,nt) < filemaskpercents[ifile]
            mask[:,:,0,:] = False # Can't use bottom layer but useful for mask size
            mask[:,:,-1,:] = False # Can't use top " "
        else:
            mask = maskdict["mask_"+file+'_'+str(irun)]
        maskDict["mask_"+file+'_'+str(irun)]=mask
        
        if dataAug:
            try:
                krots = maskdict["krots_"+file+'_'+str(irun)]
                print('Using loaded krots')
            except:                
                krots = np.random.randint(0, 4, size = (nx,ny,nz,nt))
        else:
            krots = np.zeros( (nx,ny,nz,nt), dtype = int)
        maskDict["krots_"+file+'_'+str(irun)]=krots
        
        u3d, Ri, y = normalize_and_rotate(inputFields, outputFields, mask, hvelScale, vvelScale, 
                                          thhScale, th3Scale, t33Scale, width, krots)

        us.append(u3d)
        Ris.append(Ri)
        ys.append(y)
    
    u3d = np.concatenate(us)
    Ri = np.concatenate(Ris)
    y = np.concatenate(ys)
    
    print("output shape is " + str(y.shape))
    print("input shape was " + str(u3d.shape)+ " + " +str(Ri.shape))
    if reshape:
        size=int(2*size+1)
        u=my_reshape(u3d,size=size)    
        print("input shape to do 3rd dimension as channel in R2Conv is "+str(u.shape) + " + " +str(Ri.shape))
    else:
        u=u3d
        print("input shape is "+str(u.shape) + " + " +str(Ri.shape))
    
    return u, Ri, y, maskDict

def normalize_and_rotate(inputFields, outputFields, mask, hvelScale, vvelScale, thhScale, th3Scale, t33Scale, width, krots):
    size=int((width-1)/2)  # Recentering on origin useful here, but size as extent more natural to input
    vsize=1; depth = 3 
    
    nsamples = np.sum(mask)
    print("Number of samples is " + str(nsamples))
    u3d = np.empty((nsamples, 3, width, width, depth))
    Ri = np.empty((nsamples, 1, 1, 1))
    y = np.empty((nsamples, outputFields.shape[0]))
    
    s = 0 # Sample index as a counter
    for i in range(size, inputFields.shape[1] - size):
        for j in range(size, inputFields.shape[2] - size):
            for k in range(vsize, inputFields.shape[3] - vsize):
                for it in range(inputFields.shape[4]):
                    if mask[i-size,j-size,k,it]:                            
                        
                        # Call to scale to subtract input-box mean
                        scaledInput=[scale(inputFields[0][ i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it], sd=hvelScale),
                                     scale(inputFields[1][ i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it], sd=hvelScale),
                                     scale(inputFields[2][ i - size: i + size + 1, j - size: j + size + 1, k - vsize: k + vsize + 1, it], sd=vvelScale)]
                        u3d[s] = scaledInput
                      
                        Ri[s] = inputFields[3][ i , j , k , it]
                        # No subtracting mean from outputs
                        # y[s] = outputFields[:,i-size,j-size,k,it]/tijScale
                        y[s] = [outputFields[0,i-size,j-size,k,it]/thhScale,
                                outputFields[1,i-size,j-size,k,it]/thhScale,
                                outputFields[2,i-size,j-size,k,it]/th3Scale, 
                                outputFields[3,i-size,j-size,k,it]/thhScale,
                                outputFields[4,i-size,j-size,k,it]/th3Scale,
                                outputFields[5,i-size,j-size,k,it]/t33Scale]

                        krot=krots[i-size,j-size,k,it]
                        if krot != 0: # rotate_sample works for krot = 0, but no need
                            u3d[s], y[s] = rotate_sample(u3d[s], y[s], krot)
                        
                        s += 1

    return u3d, Ri, y

def cut_sponge(x):
    # Cuts the input 75% along the z-axis to remove sponge layer
    return x[:int(0.75*x.shape[0])]

def scale(x,mean=None,sd=None):

    if mean==None:
        mean=np.mean(x)
        
    if sd==None:

        sd=np.std(x)
    
    return (x - mean) / sd

def my_reshape(xtest,nvars=3,size=3):
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

def rotate_sample(inputFields_in, outputFields_in, krot):
    inputFields_out=np.empty(inputFields_in.shape)
    outputFields_out=np.empty(outputFields_in.shape)
    
    # Rotate size x size fields, including vertical velocity
    for v in range(inputFields_out.shape[0]):
        inputFields_out[v]=np.rot90(inputFields_in[v].copy(),krot) 
    theta=krot*np.pi/2.0

    # Apply 2x2 rotation matrix to only horizontal, [u,v], velocity
    R=np.rint([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) 
    for i in range(inputFields_out.shape[1]):
        for j in range(inputFields_out.shape[2]):
            for k in range(inputFields_out.shape[3]):
                inputFields_out[0:2,i,j,k]=R@inputFields_out[0:2,i,j,k] 

    # Apply 3x3 rotation matrix to 3x3 stress tensor 
    R=np.rint([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0,0,1]])
    T=np.array([[outputFields_in[0],outputFields_in[1], outputFields_in[2]],
                [outputFields_in[1],outputFields_in[3], outputFields_in[4]],
                [outputFields_in[2],outputFields_in[4], outputFields_in[5]]])
    Tprime=R@T@R.T

    # Recover only unique components of symmetric tensor
    outputFields_out[0]=Tprime[0,0]
    outputFields_out[1]=Tprime[0,1]
    outputFields_out[2]=Tprime[0,2]
    outputFields_out[3]=Tprime[1,1]
    outputFields_out[4]=Tprime[1,2]
    outputFields_out[5]=Tprime[2,2]

    return inputFields_out, outputFields_out