import collections
import os
import jsonpickle as jsonpickle
import cv2
import matplotlib.pyplot as plt
from Segmentation.helper import hsv_to_rgb_cv

output_folder_path = 'result'
save_json = os.path.join(output_folder_path,'scene_id_list.json')
with open(save_json, 'r') as in_file:
    scene_id_list = jsonpickle.decode(in_file.read())

for scene_id in scene_id_list:

    save_json = os.path.join(os.path.join(output_folder_path,scene_id),f'{scene_id}.json')
    with open(save_json, 'r') as in_file:
        scene = jsonpickle.decode(in_file.read())
    image_origin = cv2.imread(scene.image_path)
    prev_text_pos_list = list()
    class_dict = collections.defaultdict(float)
    class_now = "0"


    for obj in scene.objectlist:
        class_dict[obj.classname] += obj.percentage * 100
        if len(list(class_dict.keys())) % 4 == 0 and obj.classname != class_now:
            cv2.imwrite(os.path.join(os.path.join(output_folder_path, scene_id),
                                     f"{scene_id}_{len(list(class_dict.keys())) // 4}_label.png"), image_origin)
            image_origin = cv2.imread(scene.image_path)
            class_now = obj.classname
            prev_text_pos_list = list()

        text_pos = (tuple(obj.contour[0][0])[0], tuple(obj.contour[0][0])[1]+50)
        overlap = True
        while overlap:
            overlap = False
            for pre in prev_text_pos_list:
                dis_x = abs(pre[0] - text_pos[0])
                dis_y = abs(pre[1] - text_pos[1])
                if dis_x < 60 and dis_y < 15:
                    text_pos = (text_pos[0] - 5, text_pos[1] - 5)
                    overlap = True
                    break

        color = hsv_to_rgb_cv(obj.color_hsv)
        color = tuple(color.tolist())
        cv2.drawContours(image_origin, [obj.contour], 0, color, 2)
        cv2.putText(image_origin, obj.classname+str(obj.object_id), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        prev_text_pos_list.append(text_pos)


    cv2.imwrite(os.path.join(os.path.join(output_folder_path, scene_id),
                             f"{scene_id}_{(len(list(class_dict.keys())) // 4) + 1}_label.png"), image_origin)


    labels = list(class_dict.keys())
    values = list(class_dict.values())
    plt.figure(figsize=(24, 18))
    wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.3f%%', textprops=dict(color="w", fontsize=10))

    # Add legend with percentages
    legend_labels = ['{} - {:.3f}%'.format(label, value) for label, value in zip(labels, values)]
    plt.legend(wedges, legend_labels, title='Categories', loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
               prop={'size': 15})

    # Add title
    plt.title('Pie chart with categories and percentages')
    plt.savefig(os.path.join(os.path.join(output_folder_path,scene_id),f"{scene_id}_piechart.png"))


