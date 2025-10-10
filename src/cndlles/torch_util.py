import torch
from escnn import nn
from escnn import gspaces

class CNDNN(torch.nn.Module):
    # C(N)-equivariant Deep Neural Network
    def __init__(self, Nhid, N, size, device):
        super().__init__()
        
        r2_act = gspaces.rot2dOnR2(N = N)

        self.feat_type_in = nn.FieldType(r2_act, 3*[r2_act.irrep(1)] # 3 planes of [u,v] each a single irrep(1)
                                         + 2*3*[r2_act.trivial_repr]) # 2 variables, w and b, each with 3 planes and irrep(0)
        self.feat_type_hid1 = nn.FieldType(r2_act, Nhid[0]*[r2_act.regular_repr])
        self.feat_type_hid2 = nn.FieldType(r2_act, Nhid[1]*[r2_act.regular_repr])
        self.feat_type_hid3 = nn.FieldType(r2_act, Nhid[2]*[r2_act.regular_repr])
        self.feat_type_hid4 = nn.FieldType(r2_act, Nhid[3]*[r2_act.regular_repr])
        if N==4:
            self.feat_type_out = nn.FieldType(r2_act, 2*[r2_act.irrep(0)]+[r2_act.irrep(1)]+2*[r2_act.irrep(2)])
        else:
            self.feat_type_out = nn.FieldType(r2_act, 2*[r2_act.irrep(0)]+[r2_act.irrep(1)]+[r2_act.irrep(2)])

        
        self.input_layer = nn.R2Conv(self.feat_type_in, self.feat_type_hid1, kernel_size=size,bias=False)
        self.relu1 = nn.ReLU(self.feat_type_hid1)
        self.hid1 = nn.R2Conv(self.feat_type_hid1, self.feat_type_hid2, kernel_size=1,bias=False)
        self.relu2 = nn.ReLU(self.feat_type_hid2)
        self.hid2 = nn.R2Conv(self.feat_type_hid2, self.feat_type_hid3, kernel_size=1,bias=False)
        self.relu3 = nn.ReLU(self.feat_type_hid3)
        self.hid3 = nn.R2Conv(self.feat_type_hid3, self.feat_type_hid4, kernel_size=1,bias=False)
        self.relu4 = nn.ReLU(self.feat_type_hid4)
        self.hid4 = nn.R2Conv(self.feat_type_hid4, self.feat_type_out, kernel_size=1,bias=False)
   
        self.Pinv=torch.tensor([[1/2, 0, 0, 0, -1/4, 1/4],
                               [0, 0, 0, 0, 1/4, 1/4],
                               [0, 0, 1, 0, 0, 0],
                               [1/2, 0, 0, 0, 1/4, -1/4],
                               [0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 0]]).float().to(device)
        self.change_basis=torch.nn.Linear(6,6, bias=False)
        self.change_basis.weight=torch.nn.Parameter(self.Pinv, requires_grad=False)

    def forward(self,x):
        
        x = nn.GeometricTensor(x, self.feat_type_in)

        x= self.input_layer(x)
        x = self.relu1(x)
        x = self.hid1(x)
        x = self.relu2(x)
        x = self.hid2(x)
        x = self.relu3(x)
        x = self.hid3(x)
        x = self.relu4(x)
        x = self.hid4(x)
        x = x.tensor.squeeze(-1).squeeze(-1)
        x = self.change_basis(x)
        
        return x

def train_model(model, criterion, loader, optimizer, scheduler, weights, device):
    model.train()
    for step, (x_batch, y_batch) in enumerate(loader):  # for each training step
        x_batch=x_batch.to(device)
        y_batch=y_batch.to(device)
                
        prediction = model(x_batch)

        loss = criterion(prediction*weights, y_batch*weights).cpu()
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients to update weights
        
    scheduler.step()

def test_model(model, criterion, trainloader, weights, device, text = 'validation'):
    model.eval() # Evaluation mode (important when having dropout layers)
    test_loss = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(trainloader):  # for each training step
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device)

            prediction = model(x_batch)

            loss = criterion(prediction*weights, y_batch*weights).cpu()
            test_loss = test_loss + loss.data.numpy() # Keep track of the loss 
        test_loss /= len(trainloader) # dividing by the number of batches
        print(text + ' loss:',test_loss)
    return test_loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False