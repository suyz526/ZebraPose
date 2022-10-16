# Generate Ground Truth for ZebraPose from Scratch

## Step 1: build the mesh generator

### Requirments:
- PCL 1.8
- Opencv 3.4.5

### build the executable:

- cd to `Binary_Code_GT_Generator/Generate_Mesh_with_GT_Color`

- `mkdir build && cd build`

- `cmake ..`

- `make`

<details>
  <summary>details about this executable</summary>
We collapse this section because we also have a python API for this.

The executable `Generate_Mesh_with_GT_Label` takes the following args as input:

1. How many parts will the mesh be divided in each itration
2. How many itrations will be performed
3. The path to the orginal mesh in obj file
4. The output path to txt file, which stores the correspondence point position and its class id
5. The output for the mesh file

Example:
If we want to divide the mesh into 2^32 parts, we can use
`./Generate_Mesh_with_GT_Color 2 32 PATH_TO_MODEL/obj_000001.obj ..PATH_FOR_OUTPUT/correspondence_id.txt PATH_FOR_OUTPUT/obj_000001.ply`

or,

`./Generate_Mesh_with_GT_Color 4 16 PATH_TO_MODEL/obj_000001.obj PATH_FOR_OUTPUT/correspondence_id.txt PATH_FOR_OUTPUT/obj_000001.ply`

Note:

The number of faces in the mesh should be larger than 2^32. If not, we can generate more faces without changing the object shape using Meshlab.
  
</details>


## Step 2: build the renderer
We write a easy renderer ourself because we need to set
`glfwWindowHint(GLFW_SAMPLES, 0);`

### Requirments:
- OpenGL
- Eigen3
- glfw3
- assimp (4.1)
- OpenCV
- download the source of [`glad`](https://glad.dav1d.de/) (Profile: compatibility) into `Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/render_related_source/glad


### 1. build the C++ executable:

- cd to `Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img`
- adjust the path in `Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/render_related_source/opengl_render.cpp` line 39!!!!!! (Somehow a relativ path will be not valid in the python wrapper)
- `mkdir build && cd build`
- `cmake ..`
- `make`


### 2. build the python API:
- cd to `Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/Render_Python_API`
- `mkdir build`
- adjust the lib path in `setup.py`
- `python3 setup.py build`
- add the path (adjust the path, note that the name `lib.linux-x86_64-3.6` may different due to your machine or python version):  
`export PYTHONPATH=$PYTHONPATH:/path/to/Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/Render_Python_API/build/lib.linux-x86_64-3.6/`  
and  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/build`


## Step 3: install the bop toolkit
https://github.com/thodan/bop_toolkit
put it also under ZebraPose folder


## Step 4: Finally, generate ground truth for bop dataset:
### 1. Upsample the mesh 

Upsample the mesh until it has more than 2^16 vertices. For reconstructed mesh,  use the 'subdivision surface: mid point' function in Meshlab. For CAD mesh, use the 'Uniform Mesh Resampling Function' in Meshlab. The resulted number of vertex will not influence the CNN inference time, so bit more is also ok. Save the upsampled mesh with the `.obj` format in the folder `models`, e.g. the upsampled mesh of `obj_000001.ply` should be save as `obj_000001.obj`.

Or directly use the `models` from this [`link`](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW).

### 2. Generate a mesh with GT color

- cd to `/Binary_Code_GT_Generator`
- use `generate_mesh_with_GT_color_for_BOP.py`, e.g.  
`python3 generate_mesh_with_GT_color_for_BOP.py /home/ysu/data/data_object_pose/BOP_dataset/ lm 2 16 /home/ysu/project/Coarse_to_Fine_6DoF_Pose/Binary_Code_GT_Generator/Generate_Mesh_with_GT_Color/build/Generate_Mesh_with_GT_Color`.   
The 5 args should be:  
1\) path to bop folder  
2\) the dataset name, e.g. `lmo`  
3\) How many parts will the mesh be divided in each itration, we used 2  
4\) How many itrations will be performed, we used 16
5\) the path of the executable, which is created in step1.  
This scipt will create a folder "models_GT_color" under the dataset folder, and save all the mesh with required color and the correspondence information

### 3. Generate GT for Binary Code
<strong>WARNING</strong>: I'm not sure, if the rendering process might make some people uncomfortable (The one who has photosensitive epileptic). It is a bit similar to [`https://en.wikipedia.org/wiki/Denn%C5%8D_Senshi_Porygon`](https://en.wikipedia.org/wiki/Denn%C5%8D_Senshi_Porygon).

All the preparations are done, this is the only thing that we need for the training.  

use `generate_training_labels_for_BOP.py`, e.g. `python3 generate_training_labels_for_BOP.py --bop_path /home/ysu/data/data_object_pose/BOP_dataset/ --dataset_name lmo --force_rewrite True --is_training_data True --data_folder train_real --start_obj_id 0 --end_obj_id 3`

To be able to use symmetry aware training, use use `generate_training_labels_for_BOP_v2.py` to generate the required ground truth.

The args:
- `bop_path`: bop root path
- `dataset_name`: like lm, lmo...
- `force_rewrite`: if rewrite the existing images. if the code breaks, set this as False to avoid rendering some image again.
- `data_folder`: the target folder, e.g. train_real, train_pbr or test
- `is_training_data`: if it is training data
- `start_obj_id` and `end_obj_id`: To accelerate the rendering process, I usually run multiple processes parallelly, each of them taking care of partial objects. For example, `lmo` has 8 objects, we run 2 `generate_training_labels_for_BOP.py` independently. In the first process, the `start_obj_id` and `end_obj_id` are `0` and `3`. In the second process, the `start_obj_id` and `end_obj_id` are `4` and `8`.

The script will create a folder `train_GT_images` to save the ground truth data. The ground truth file is named in the same logic as the mask
files. 

Repeat this for all training image folders and test folders, then we are done with this part.




