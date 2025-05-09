C<sub>N</sub>-equivariant Deep Learning for Large-Eddy Simulation (C<sub>N</sub>DLLES) host scripts to develop a deep neural network (DNN) which predicts the subfilter-scale (SFS) stress needed for large-eddy simulation (LES) of atmospheric boundary layer (ABL) flows.

To address anisotropy arising from density stratification in the ABL, our DNN are designed to enforce equivariance to rotations in the horizontal plane, but not in the direction of gravity. 
On discretized grid meshes used in LES, these rotations correspond to the symmetry of the C<sub>N</sub> group.
Enforcing this equivariance through the [e2cnn](https://github.com/QUVA-Lab/e2cnn) machine learning library requires a change of basis of the SFS stress, a rank-2 tensor, which is incorporated into the DNN.
Additional attention is needed to associate the components of this changed basis, as well as the input and hidden features of the DNN, are associated with the appropriate irreducible representations.
See the paper for more details about C<sub>N</sub>DLLES (link forthcoming).
