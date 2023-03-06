import cv2
import jsonpickle as jsonpickle
import numpy as np
from Segmentation.A2D2Object import ImgInfo,A2D2_Object
from Segmentation.Objectclass import class_list
from Segmentation.helper import keep_single_color,hex_to_hsv_cv
import os
import glob
import shutil
import jsonpickle.ext.numpy as json_pickle_numpy


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

output_folder_path = 'result'
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)

os.makedirs(output_folder_path)

input_folder_path = "camera_lidar_semantic"
scene_list = list()
scene_id_list = list()
for root, dirs, files in os.walk(input_folder_path):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if dir_path.count(os.sep) - input_folder_path.count(os.sep) == 1:
            scene = ImgInfo()
            scene.scene_path = dir
            scene_id_list.append(scene.scene_path)
            scene.image_path = os.path.join(dir_path,"camera/cam_front_center")
            files = glob.glob(os.path.join(scene.image_path, "*"))
            for file in files:
                if is_image_file(file):
                    scene.image_path = file
            scene.mask_path = os.path.join(dir_path, "label/cam_front_center")
            files = glob.glob(os.path.join(scene.mask_path, "*"))
            for file in files:
                if is_image_file(file):
                    scene.mask_path = file
            # add scene to the list
            scene_list.append(scene)
save_json = os.path.join(output_folder_path,'scene_id_list.json')
with open(save_json, 'w') as f:
    f.write(jsonpickle.encode(scene_id_list,indent=4))

for scene in scene_list:
    os.makedirs(os.path.join(output_folder_path, scene.scene_path))
    out_scene = os.path.join(output_folder_path, scene.scene_path)

    image_origin = cv2.imread(scene.image_path)
    _, image_name = os.path.split(scene.image_path)
    cv2.imwrite(os.path.join(out_scene,image_name),image_origin)
    mask_origin = cv2.imread(scene.mask_path)
    _, mask_name = os.path.split(scene.mask_path)
    cv2.imwrite(os.path.join(out_scene, mask_name), mask_origin)

    image = cv2.imread(scene.mask_path)
    os.makedirs(os.path.join(out_scene, "masks"))
    save_img = os.path.join(out_scene, "masks")
    classname = "0"
    count = 0
    for hex_color in class_list.keys():
        hsv_color = hex_to_hsv_cv(hex_color)
        new_image,mask = keep_single_color(image, hsv_color)
        cv2.imwrite(os.path.join(save_img, f"{class_list[hex_color]}.png"),new_image)
        #Closing
        kernel = np.ones((8, 8), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        img_closed = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(save_img, f"{class_list[hex_color]}_closed.png"), img_closed)


        # Find contours of color blocks
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assign only pixels of specified color to new images

        if class_list[hex_color] != classname:
            classname = class_list[hex_color]
            count = 0

        for contour in contours:
            new_image = np.zeros_like(image)
            cv2.drawContours(new_image, [contour], 0, (255, 255, 255), -1)
            new_image = cv2.bitwise_and(image, new_image)
            # If you want to save each piece of the block, use this code
            # cv2.imwrite(f'result_{count}.png', new_image)

            # Count number of pixels in color block
            pixel_count = cv2.countNonZero(cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY))

            # Areas smaller than 40 pixels are ignored
            if pixel_count <= 40:
                continue
            count += 1
            obj = A2D2_Object()
            obj.color_hex = hex_color
            obj.color_hsv = hsv_color
            obj.object_id = count
            obj.classname = class_list[obj.color_hex]
            obj.pixelcount = pixel_count
            obj.percentage = obj.pixelcount / (1920 * 1208)
            obj.contour = contour
            print(f'Object {obj.classname} block {count} has {pixel_count} pixels, which occupies {obj.percentage} of the image')
            scene.objectlist.append(obj)


    save_json = os.path.join(out_scene,f'{scene.scene_path}.json')
    with open(save_json, 'w') as f:
        f.write(jsonpickle.encode(scene,indent=4))




