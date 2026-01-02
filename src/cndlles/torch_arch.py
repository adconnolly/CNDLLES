import torch
from escnn import nn
from escnn import gspaces
import itertools

def make_layer(
    in_feat_type,
    out_feat_type,
    activation=nn.ReLU,
    kernel_size=1,
    bias=False,
) -> list:
    """
    Packs equivariant layer and optionally activation
    """
    layer = [nn.R2Conv(in_feat_type, out_feat_type, kernel_size=kernel_size, bias=bias)]
    if activation is not None:
        layer.append(activation(out_feat_type))
    return layer

class CNDNN(torch.nn.Module):
    # C(N)-equivariant Deep Neural Network
    def __init__(self, Nhid,  N=4, Ri_pct = 0.25, size = 3, bias = False):
        super().__init__()
        self.nRi = int(Ri_pct*Nhid[0])
        
        # Define the C(N) group of interest
        self.r2_act = gspaces.rot2dOnR2(N = N)

        # Input feature types
                                    # 3 planes of [u,v] each a size x size irrep(1) field
        self.feat_type_u = nn.FieldType(self.r2_act, 3*[self.r2_act.irrep(1)] 
                                    # 3 planes of w velocity each a size x size irrep(0) field
                                        + 3*[self.r2_act.trivial_repr])
                                    # 1 Richardson number as an irrep(0) (1x1 'field')
        self.feat_type_Ri = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr]) # single Richardson number
        feat_type = [self.feat_type_u, self.feat_type_Ri ] 
        
        # Hidden layer feature types
        self.feat_type_uConv = nn.FieldType(self.r2_act, (Nhid[0]-self.nRi)*[self.r2_act.regular_repr])
        self.feat_type_RiConv = nn.FieldType(self.r2_act, self.nRi*[self.r2_act.regular_repr])
        feat_type.extend( [self.feat_type_uConv, self.feat_type_RiConv] )

        feat_type.extend( [nn.FieldType(self.r2_act, N*[self.r2_act.regular_repr]) for N in Nhid] )
        
        # Output layer feature types
        if N==4:
            feat_type.append( nn.FieldType(self.r2_act, 
                                    2*[self.r2_act.irrep(0)]+[self.r2_act.irrep(1)]+2*[self.r2_act.irrep(2)]))
        else:
            feat_type.append( nn.FieldType(self.r2_act,
                                    2*[self.r2_act.irrep(0)]+[self.r2_act.irrep(1)] + [self.r2_act.irrep(2)]))

        self.feat_type = feat_type
        
        # Defining the operations sequentially
        self.u_in = make_layer(self.feat_type_u, self.feat_type_uConv, kernel_size=size, bias = bias)[0] # size !=1 at input conv
        self.Ri_in = make_layer(self.feat_type_Ri, self.feat_type_RiConv, kernel_size=1, bias = True)[0] # size !=1 at input conv

        # Defining the post-directsum GeometricTensor operations sequentially
        ops = []         
        for i in range(4,len(feat_type) - 2):
            ops.extend( make_layer(feat_type[i], feat_type[i+1], bias = bias ))
        
        ops.extend( make_layer(feat_type[-2], feat_type[-1],
                               activation = None, bias = bias)) # no activation at last layer
        
        self.ops = nn.SequentialModule(*ops)

        # Change of basis layer with fixed weights
        Pinv=torch.tensor([[1/2, 0, 0, 0, -1/4, 1/4],
                            [0,  0, 0, 0, 1/4, 1/4],
                            [0,  0, 1, 0, 0,   0],
                            [1/2,0, 0, 0, 1/4, -1/4],
                            [0,  0, 0, 1, 0,   0],
                            [0,  1, 0, 0, 0,   0]])
        self.change_basis=torch.nn.Linear(6, 6, bias=False)
        self.change_basis.weight=torch.nn.Parameter(Pinv, requires_grad=False)

    def forward(self, u, Ri):

        u = nn.GeometricTensor(u, self.feat_type_u)
        u = self.u_in(u)
        
        Ri = nn.GeometricTensor(Ri, self.feat_type_Ri)
        Ri = self.Ri_in(Ri)
                
        x = nn.tensor_directsum([u, Ri])
        x = self.ops(x)
        x = x.tensor.squeeze(-1).squeeze(-1)
        
        return self.change_basis(x)

def make_baseline_layer(
    in_features: int,
    out_features: int,
    activation=torch.nn.ReLU,
    bias=False,
) -> list:
    """
    Packs ANN layer and optionally activation
    """
    layer = [torch.nn.Linear(in_features, out_features, bias=bias)]
    if activation is not None:
        layer.append(activation())
    return layer

class baselineDNN(torch.nn.Module):
    def __init__(self, Nhid, input_shape, Ri_pct = 0.25, bias=False):
        super().__init__()
        self.nRi = int(Ri_pct*Nhid[0])
        

        ## Input layer
        self.u_in = torch.nn.Conv2d(input_shape[0], Nhid[0] - self.nRi, (input_shape[1], input_shape[2]), padding='valid', bias=bias)
        self.Ri_in = torch.nn.Linear(1, self.nRi, bias=True)
        
     
        # Following recipe for itertools.pairwise
        ops = []
        Nin, Nout = itertools.tee(Nhid)
        next(Nout, None)
        n_iter = zip(Nin, Nout)
        for n_in, n_out in itertools.islice(n_iter, len(Nhid) - 1):
            ops.extend(
                make_baseline_layer(
                    in_features=n_in,
                    out_features=n_out,
                    # activation=activation,
                    # batch_norm=False,
                    bias=False,
                )
            )
        ops.extend(
            make_baseline_layer(
                in_features=n_out,
                out_features=6,
                activation=None,
                # batch_norm=False,
                bias=False,
            )
        )
        # Bundle into Sequential for forward pass
        self.ops = torch.nn.Sequential(*ops)

        Pinv=torch.tensor([[1/2, 0, 0, 0, -1/4, 1/4],
                               [0, 0, 0, 0, 1/4, 1/4],
                               [0, 0, 1, 0, 0, 0],
                               [1/2, 0, 0, 0, 1/4, -1/4],
                               [0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 0]]).float()
        self.change_basis=torch.nn.Linear(6,6, bias=False)
        self.change_basis.weight=torch.nn.Parameter(Pinv, requires_grad=False)
                                                                    
    def forward(self, u, Ri):
        
        u = self.u_in(u).squeeze(dim=(2,3))
        Ri = self.Ri_in(Ri.squeeze(dim=(2,3)))
        
        x = torch.cat( [u, Ri], dim = 1 )
        x = self.ops(x)
        
        return self.change_basis(x)