# Contact-GraspNet Pytorch
Pytorch implementation of Contact-GraspNet. This repo is based heavily on 
https://github.com/alinasarmiento/pytorch_contactnet. Original Tensorflow implementation can be found at: https://github.com/NVlabs/contact_graspnet


## Installation Instructions
`pip install cgn-pytorch`

## Usage:
`import cgn_pytorch`

`cgn_model, optimizer, config_dict  = cgn_pytorch.from_pretrained(cpu=False)`


## Run The Demo
### Clone this Repo
`git clone  https://github.com/sebbyjp/cgn_pytorch.git`

### Install Dependencies
`pip3 install -r requirements.txt`

### Vizualization
We're doing our visualizations in MeshCat. In a separate tab, start a meshcat server with `meshcat-server`

From here you can run `python3 eval.py`

To visualize different confidence threshold masks on the grasps, use the `--threshold` argument such as `--threshold=0.8`

To visualize different cluttered scenes (8-12 tabletop objects rendered in pyrender from ACRONYM), use the argument `--scene` and manually feed in a file name such as `--scene=002330.npz`.Your possible files are:
- 002330.npz
- 004086.npz
- 005274.npz

## Predicting grasps on your own pointclouds
The model should work on any pointcloud of shape (Nx3). For most consistent results, please make sure to put the pointcloud in the world frame and center it by subtracting the mean. Do not normalize the pointcloud to a unit sphere or unit box, as "graspability" naturally changes depending on the size of the objects (so we don't want to lose that information about the scene by scaling it).

