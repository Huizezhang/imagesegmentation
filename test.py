import collections
import os
import jsonpickle as jsonpickle
import cv2
import numpy as np
from Segmentation.helper import min_contour_distance
from shapely.geometry import LineString
a=LineString([(1,1),(3,5)])
b=LineString([(2,2),(5,2)])
print(a.distance(b))

output_folder_path = 'result'
save_json = os.path.join(output_folder_path, 'scene_id_list.json')
with open(save_json, 'r') as in_file:
    scene_id_list = jsonpickle.decode(in_file.read())

save_json = os.path.join(os.path.join(output_folder_path, scene_id_list[0]), f'{scene_id_list[0]}.json')
with open(save_json, 'r') as in_file:
    scene = jsonpickle.decode(in_file.read())
image_origin = cv2.imread(scene.image_path)
car = list()
for obj in scene.objectlist:
    if obj.classname != "Car":
        continue

    img = cv2.imread(f"result/{scene_id_list[0]}/masks/Car/Car_{obj.object_id}.png")
    canvas = np.zeros_like(img, np.uint8)
    cv2.drawContours(canvas, [obj.contour], 0, 255, -1)
    cv2.imwrite(f"xxx.png", canvas)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=10, minLineLength=5, maxLineGap=25)

    # 绘制直线并显示原始图像
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1)) * 180 / np.pi
            if 80 < angle < 100 or 260 < angle < 280:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.imwrite(f"Car_{obj.object_id}.png", img)







