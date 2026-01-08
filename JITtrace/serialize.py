import torch
from cndlles.torch_util import CNDNN

savePath='./'
loadPath ='../Reynolds_oldEnv/trainedModels/'

size=3
Nhid = [32,16,8]
Ri_pct = 0.25

#loadFile='C4_ReInterp_'
#trainFiles=["coarse4x1026_Re900.nc","coarse4x3078_Re2700.nc"]
#irun=0
#for n in range(len(trainFiles)):
#    isep = trainFiles[n].index('_')
#    loadFile=loadFile+trainFiles[n][6:isep]+trainFiles[n][isep+1:-3]+'_'
#loadFile=loadFile+str(irun)
loadFile='C4_ReExtrap_4x1026Re900_4x2052Re1800_1'
print(loadFile)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

model=CNDNN(Nhid, Ri_pct = 0.25,N=4,size=size).float().to(device)
model.load_state_dict(torch.load(loadPath+loadFile+'.pt', map_location=device))

Ntests = 1
for n in range(Ntests):
    u_eg = torch.randn(10, 9, size, size).float().to(device)
    Ri_eg = torch.randn(10, 1, 1, 1).float().to(device)
    
    traced_model=torch.jit.trace(model, [u_eg, Ri_eg])
    
    print(model(u_eg, Ri_eg))
    print("vs")
    print(traced_model(u_eg, Ri_eg))

traced_model.save(savePath+"traced_"+loadFile+".pt")