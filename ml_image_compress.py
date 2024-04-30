'''
import numpy as np
import torch
import bump_data_process as bdp
import bump_ml_compress as bmc

generate_bump_sample_rays = True
debug_draw_sample_ray = False
retrain_ml_model = True

depth_image = bdp.loadBumpImage('tile1.tga', False)

height_map = np.array(depth_image) / 255.0

sample_ray_cache_file_name = 'tile1_data.npy'
trained_model_name = "tile1_bumpmap_compress_model.pth"

sample_data = None

if generate_bump_sample_rays:
    sample_data = bdp.generateBumpSampleRays(height_map, sample_ray_cache_file_name, True)
else:
    sample_data = np.load(sample_ray_cache_file_name)

print(sample_data.shape)
print(sample_data.dtype)

w = sample_data.shape[0]
h = sample_data.shape[1]

total_sample_count = w * h * bdp.SAMPLE_RAY_COUNT * bdp.ANGLE_SAMPLE_COUNT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phi = 30.0
theta = 5.0

if debug_draw_sample_ray:
    image_data = bdp.debugBumpRayBuffer(w, h, phi, theta, sample_data, depth_image)
    
if retrain_ml_model:
    inputs, targets = bmc.createTensorBuffer(total_sample_count, w, h, sample_data, device)
    bmc.trainMLModel(inputs, targets, total_sample_count, trained_model_name, True, device)
else:
    # Load the entire model
    model = bmc.BumpmapInfoNet().to(device)
    state_dict = torch.load(trained_model_name)
    model.load_state_dict(state_dict)

# Make sure to call model.eval() if you're in inference mode
model.eval()

for name, param in model.named_parameters():
    print("Parameter name:", name)
    print("Parameter data:")
    print(param.data)
    print("Parameter shape:", param.data.shape)
    print()

ml_result = bmc.testMLModel(model, w, h, phi, theta, device)

bmc.debugBumpMLResultBuffer(w, h, phi, ml_result, depth_image)
'''
import numpy as np
import torch
import bump_data_process as bdp
import bump_ml_per_pixel_compress as bmc

generate_bump_sample_rays = False
debug_draw_sample_ray = False
retrain_ml_model = True

depth_image = bdp.loadBumpImage('tile1.tga', False)

height_map = np.array(depth_image) / 255.0

sample_ray_cache_file_name = 'tile1_data.npy'
trained_model_name = "tile1_bumpmap_compress_model.pth"

sample_data = None

if generate_bump_sample_rays:
    sample_data = bdp.generateBumpSampleRays(height_map, sample_ray_cache_file_name, True)
else:
    sample_data = np.load(sample_ray_cache_file_name)

print(sample_data.shape)
print(sample_data.dtype)

w = sample_data.shape[0]
h = sample_data.shape[1]

total_sample_count = bdp.SAMPLE_RAY_COUNT * bdp.ANGLE_SAMPLE_COUNT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phi = 30.0
theta = 5.0

if debug_draw_sample_ray:
    image_data = bdp.debugBumpRayBuffer(w, h, phi, theta, sample_data, depth_image)

ml_result = np.empty(w * h, dtype = np.float32)  

for i in range(w):
    for j in range(h):
        print("process pixel", i, j)
        model = None
        if retrain_ml_model:
            inputs, targets = bmc.createTensorBuffer(total_sample_count, i, j, sample_data, device)
            model = bmc.trainMLModel(inputs, targets, total_sample_count, trained_model_name, False, device)
        else:
            # Load the entire model
            model = bmc.BumpmapInfoNet().to(device)
            state_dict = torch.load(trained_model_name)
            model.load_state_dict(state_dict)

        # Make sure to call model.eval() if you're in inference mode
        model.eval()

        value = bmc.testMLModel(model, phi, theta, device)
        ml_result[i * h + j] = value[0]

bdp.debugBumpMLResultBuffer(w, h, phi, ml_result, depth_image)