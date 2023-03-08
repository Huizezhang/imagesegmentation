import collections
import os
import jsonpickle as jsonpickle
import cv2
import streamlit as st
import plotly.graph_objs as go
from Segmentation.helper import hsv_to_rgb_cv, merge_small_parts

output_folder_path = 'result'
save_json = os.path.join(output_folder_path, 'scene_id_list.json')
with open(save_json, 'r') as in_file:
    scene_id_list = jsonpickle.decode(in_file.read())

selected_scene = st.selectbox('Select Scene', scene_id_list)
scene_index = scene_id_list.index(selected_scene)
class_count = {"Car": 0, "Bicycle": 0, "Pedestrian": 0, "Truck": 0, "Small vehicles": 0,
               "Traffic signal": 0, "Traffic sign": 0, "Animals": 0, "Utility vehicle": 0}

for scene_id in scene_id_list:
    if scene_id != selected_scene:
        continue

    save_json = os.path.join(os.path.join(output_folder_path, scene_id), f'{scene_id}.json')
    with open(save_json, 'r') as in_file:
        scene = jsonpickle.decode(in_file.read())
    image_origin = cv2.imread(scene.image_path)
    prev_text_pos_list = list()
    class_dict = collections.defaultdict(float)
    class_now = "0"

    for obj in scene.objectlist:
        if class_count.get(obj.classname) is not None:
            class_count[obj.classname] += 1
        class_dict[obj.classname] += obj.percentage * 100
        if len(list(class_dict.keys())) % 4 == 0 and obj.classname != class_now:
            image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
            st.image(image_origin, caption='Image for {}'.format(scene.image_path), use_column_width=True)
            image_origin = cv2.imread(scene.image_path)
            class_now = obj.classname
            prev_text_pos_list = list()

        text_pos = (tuple(obj.contour[0][0])[0], tuple(obj.contour[0][0])[1] + 50)
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
        cv2.putText(image_origin, obj.classname + str(obj.object_id), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        prev_text_pos_list.append(text_pos)

    cv2.imwrite(os.path.join(os.path.join(output_folder_path, scene_id),
                             f"{scene_id}_{(len(list(class_dict.keys())) // 4) + 1}_label.png"), image_origin)
    st.image(image_origin, caption='Image for {}'.format(scene.image_path), use_column_width=True)

    labels = list(class_dict.keys())
    values = list(class_dict.values())
    new_sizes, new_labels = merge_small_parts(values, labels, 1)
    total = sum(new_sizes)
    normalized = [x / total for x in new_sizes]
    normalized_sum = sum(normalized)
    factor = 100.0 / normalized_sum
    normalized_values = [x * factor for x in normalized]
    fig = go.Figure(data=[go.Pie(labels=labels, values=normalized_values)])
    fig.update_layout(title='Piechart with categories and percentages')
    st.plotly_chart(fig)

    class_count = {key: value for key, value in class_count.items() if value != 0}
    print(class_count)
    data = [go.Bar(x=list(class_count.keys()), y=list(class_count.values()))]

    # 设置布局
    layout = go.Layout(title='柱状图', xaxis=dict(title='类别'), yaxis=dict(title='数据'))

    st.plotly_chart(data)
