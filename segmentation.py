import cv2
import jsonpickle as jsonpickle
import numpy as np
from Segmentation.A2D2Object import ImgInfo, A2D2_Object
from Segmentation.Objectclass import class_list
from Segmentation.helper import keep_single_color, hex_to_hsv_cv, is_image_file, min_contour_distance
import os
import glob
import shutil
from shapely.geometry import LineString

#Create output folder
output_folder_path = 'result'
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)

os.makedirs(output_folder_path)

#Scan scene
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
            scene.image_path = os.path.join(dir_path, "camera/cam_front_center")
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
save_json = os.path.join(output_folder_path, 'scene_id_list.json')
with open(save_json, 'w') as f:
    f.write(jsonpickle.encode(scene_id_list, indent=4))

for scene in scene_list:
    os.makedirs(os.path.join(output_folder_path, scene.scene_path))
    out_scene = os.path.join(output_folder_path, scene.scene_path)

    image_origin = cv2.imread(scene.image_path)
    _, image_name = os.path.split(scene.image_path)
    cv2.imwrite(os.path.join(out_scene, image_name), image_origin)
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
        new_image, mask = keep_single_color(image, hsv_color)
        cv2.imwrite(os.path.join(save_img, f"{class_list[hex_color]}_{hex_color}.png"), new_image)

        os.makedirs(os.path.join(save_img, f"{class_list[hex_color]}"), exist_ok=True)
        obj_path = os.path.join(save_img, f"{class_list[hex_color]}")

        # Find contours of color blocks
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assign only pixels of specified color to new images

        if class_list[hex_color] != classname:
            classname = class_list[hex_color]
            count = 0

        for contour in contours:
            new_image = np.zeros_like(image)
            cv2.drawContours(new_image, [contour], 0, (255, 255, 255), -1)
            new_image = cv2.bitwise_and(image, new_image)

            # Count number of pixels in color block
            pixel_count = cv2.countNonZero(cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY))

            # Areas smaller than 40 pixels are ignored
            if pixel_count <= 55:
                continue
            count += 1
            # If you want to save each piece of the block, use this code
            #cv2.imwrite(os.path.join(obj_path, f"{class_list[hex_color]}_{count}.png"), new_image)
            obj = A2D2_Object()
            obj.color_hex = hex_color
            obj.color_hsv = hsv_color
            obj.object_id = count
            obj.classname = class_list[obj.color_hex]
            obj.pixelcount = pixel_count
            obj.percentage = obj.pixelcount / (1920 * 1208)
            obj.contours.append(contour)
            obj.block_path = os.path.join(obj_path, f"{class_list[hex_color]}_{count}.png")
            gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=10, minLineLength=3, maxLineGap=25)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1)) * 180 / np.pi
                    if 80 < angle < 100 or 260 < angle < 280:
                        cv2.line(new_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        obj.lines.append([x1, y1, x2, y2])
                cv2.imwrite(os.path.join(obj_path, f"{class_list[hex_color]}_{count}_withline.png"), new_image)
            print(
                f'Object {obj.classname} block {count} has {pixel_count} pixels, which occupies {obj.percentage} of the image')
            flag = False
            for pre_obj in scene.objectlist:
                if pre_obj.color_hex != obj.color_hex or len(obj.lines) == 0 or len(pre_obj.lines) == 0:
                    continue
                for pre_contour in pre_obj.contours:
                    if min_contour_distance(pre_contour,contour) < 32:
                        print(pre_obj.lines, obj.lines)
                        dist_matrix = np.zeros((len(pre_obj.lines), len(obj.lines)))
                        for i in range(len(pre_obj.lines)):
                            pre_line = LineString([(pre_obj.lines[i][0], pre_obj.lines[i][1]),
                                                   (pre_obj.lines[i][2], pre_obj.lines[i][3])])
                            for j in range(len(obj.lines)):
                                obj_line = LineString([(obj.lines[j][0], obj.lines[j][1]),
                                                       (obj.lines[j][2], obj.lines[j][3])])
                                dist_matrix[i][j] = pre_line.distance(obj_line)

                        if np.min(dist_matrix) < 32:
                            flag = True
                            break
                if flag:
                    pre_obj.contours = pre_obj.contours + obj.contours
                    pre_obj.lines = pre_obj.lines + obj.lines
                    pre_obj.pixelcount += obj.pixelcount
                    pre_obj.percentage += obj.percentage
                    break
            if not flag:
                scene.objectlist.append(obj)

    for obj in scene.objectlist:
        new_image = np.zeros_like(image)
        for contour in obj.contours:
            cv2.drawContours(new_image, [contour], 0, (255, 255, 255), -1)
        obj_path = os.path.join(save_img, f"{class_list[obj.color_hex]}")
        cv2.imwrite(os.path.join(obj_path, f"{class_list[obj.color_hex]}_{obj.object_id}.png"), new_image)





    save_json = os.path.join(out_scene, f'{scene.scene_path}.json')
    with open(save_json, 'w') as f:
        f.write(jsonpickle.encode(scene, indent=4))
