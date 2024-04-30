from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

ANGLE_SAMPLE_COUNT = 16
SAMPLE_RAY_COUNT = 256

def loadBumpImage(file_name, debug_draw):
    # Load the image
    image_path = file_name  # Update this to the path of your TGA image
    image = Image.open(image_path)

    depth_image = None

    print(image.mode)
    # Split the image into channels
    if image.mode == 'RGBA':
        dumb, dumb, dumb, depth_image = image.split()
    elif image.mode == 'RGB':
        dumb, dumb, depth_image = image.split()
    elif image.mode == 'L':
        depth_image = image

    # Display the alpha channel
    if debug_draw:
        plt.imshow(depth_image, cmap='gray')
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()
    
    return depth_image

def generateBumpSampleRays(height_map, file_name, write_out):
    w = height_map.shape[0]
    h = height_map.shape[1]

    sample_data = np.empty((w, h, SAMPLE_RAY_COUNT, ANGLE_SAMPLE_COUNT), dtype = np.float32)

    print(sample_data.shape)

    sample_dist_array = np.empty(w, dtype = np.float32)
    sample_angle_array = np.empty(w, dtype = np.float32)
    reference_angle_sample = np.empty(ANGLE_SAMPLE_COUNT, dtype = np.float32)

    for i in range(ANGLE_SAMPLE_COUNT) :
        reference_angle_sample[i] = (i + 0.5) / float(ANGLE_SAMPLE_COUNT) * np.pi / 2.0

    for i in range(w) :
        print("processing line ", i, " of ", w)
        for j in range(h) :
            x = (i + 0.5) / float(w) * 2.0 - 1.0
            y = (j + 0.5) / float(h) * 2.0 - 1.0

            for k in range(SAMPLE_RAY_COUNT) :
                for s in range(ANGLE_SAMPLE_COUNT) :
                    sample_data[i][j][k][s] = 0.0

            if height_map[i][j] != 0:
                for k in range(SAMPLE_RAY_COUNT) :
                    omega = (k + 0.5) / float(SAMPLE_RAY_COUNT) * 2.0 * np.pi
                    dir = (np.cos(omega), np.sin(omega))
                    
                    edge_x = np.sign(dir[0])
                    edge_y = np.sign(dir[1])

                    t_x = (edge_x - x) / dir[0]
                    t_y = (edge_y - y) / dir[1]

                    t = np.min((t_x, t_y))
                    num_sample_x = int(t * np.abs(dir[0]) * w / 2)
                    num_sample_y = int(t * np.abs(dir[1]) * h / 2)

                    num_sample = np.max((num_sample_x, num_sample_y))

                    valid_num_sample = 0
                    for s in range(num_sample):
                        ratio = (float(s) + 0.5) / float(num_sample)
                        s_i = i + int(dir[0] * t * ratio * w / 2)
                        s_j = j + int(dir[1] * t * ratio * h / 2)
                        s_x = (s_i - i) / float(w)
                        s_y = (s_j - j) / float(h)
                        sample_dist = np.sqrt(s_x * s_x + s_y * s_y)
                        sample_angle = np.pi / 2.0
                        if height_map[s_i][s_j] > 0:
                            sample_angle = np.arctan(sample_dist / height_map[s_i][s_j])
                        if s == 0 or sample_angle > sample_angle_array[valid_num_sample-1] :
                            sample_dist_array[valid_num_sample] = sample_dist
                            sample_angle_array[valid_num_sample] = sample_angle
                            valid_num_sample = valid_num_sample + 1

                    sample_idx = 0
                    for s in range(ANGLE_SAMPLE_COUNT):
                        if valid_num_sample > 0:
                            while sample_idx < valid_num_sample and reference_angle_sample[s] > sample_angle_array[sample_idx]:
                                sample_idx = sample_idx + 1

                            if sample_idx == 0: # first valid item
                                if sample_angle_array[sample_idx] == 0.0:
                                    sample_data[i][j][k][s] = 0
                                else:
                                    angle_ratio = reference_angle_sample[s] / sample_angle_array[sample_idx]
                                    sample_data[i][j][k][s] = sample_dist_array[valid_num_sample] * angle_ratio
                            elif sample_idx == valid_num_sample: # last item, out range
                                sample_data[i][j][k][s] = t / 2
                            else:
                                angle_ratio = (reference_angle_sample[s] - sample_angle_array[sample_idx-1]) / (sample_angle_array[sample_idx] - sample_angle_array[sample_idx-1])
                                sample_data[i][j][k][s] = sample_dist_array[sample_idx-1] + (sample_dist_array[sample_idx]-sample_dist_array[sample_idx-1]) * angle_ratio
                        else:
                            sample_data[i][j][k][s] = t / 2
                        
    #                    if sample_data[i][j][k][s] > 0 :
    #                        print(sample_data[i][j][k][s])

    if write_out:
        np.save(file_name, sample_data)
    
    return sample_data

def debugBumpRayBuffer(w, h, phi, theta, sample_data, src_image):
    image_data = np.empty((w, h), dtype = np.uint8)
    view_angle = phi / 360.0 * np.pi * 2.0
    view_angle_sample_idx =  view_angle / (np.pi * 2.0) * SAMPLE_RAY_COUNT
    view_down_angle_idx = theta / 90.0 * ANGLE_SAMPLE_COUNT

    for i in range(w):
        for j in range(h):
            pixel_offset = sample_data[i][j][int(view_angle_sample_idx)][int(view_down_angle_idx)]
            #print(pixel_offset)

            n_i = i + int(np.cos(view_angle) * pixel_offset * w)
            n_j = j + int(np.sin(view_angle) * pixel_offset * h)

            image_data[i][j] = src_image.getpixel((n_i,n_j))

    image = Image.fromarray(image_data, 'L')
    image.show()

    return image_data

def debugBumpMLResultBuffer(w, h, phi, sample_data, depth_image):
    view_angle = phi / 360.0 * np.pi * 2.0
    image_data = np.empty((w, h), dtype = np.uint8)
    for i in range(w):
        for j in range(h):
            pixel_offset = sample_data[i * h + j]
            #print(pixel_offset)

            n_i = i + int(np.cos(view_angle) * pixel_offset * w)
            n_j = j + int(np.sin(view_angle) * pixel_offset * h)

            if n_i < 0:
                n_i = 0
            if n_i >= w:
                n_i = w - 1
            if n_j < 0:
                n_j = 0
            if n_j >= h:
                n_j = h - 1

            gray_level = depth_image.getpixel((n_i,n_j))
            image_data[i][j] = gray_level

    image = Image.fromarray(image_data, 'L')
    image.show()

    return image_data
