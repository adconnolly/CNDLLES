# C<sub>N</sub>DLLES

C<sub>N</sub>-equivariant Deep Learning for Large-Eddy Simulation (C<sub>N</sub>DLLES) hosts software to develop a deep neural network (DNN) which predicts the subfilter-scale (SFS) stress needed for large-eddy simulation (LES) of atmospheric boundary layer (ABL) flows.

To address anisotropy arising from density stratification in the ABL, our DNN are designed to enforce equivariance to rotations in the horizontal plane, but not in the direction of gravity. 
On discretized grid meshes used in LES, these rotations correspond to the symmetry of the C<sub>N</sub> group.
Enforcing this equivariance through the [escnn](https://github.com/QUVA-Lab/escnn) machine learning library requires associating input, outout, and hidden layer features with the appropriate irreducible representations. 
For example, to output the unique components of the SFS stress, a rank-2 tensor, a change of basis is incorporated into the DNN. 
See the paper for more details about C<sub>N</sub>DLLES (link forthcoming).

## Editable install

In an environment with the dependencies and from root directory run:

```bash
pip install -e .
```
