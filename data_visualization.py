import collections
import os
import jsonpickle as jsonpickle
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from Segmentation.helper import hsv_to_rgb_cv, merge_small_parts
from Segmentation.Objectclass import class_list
from PIL import Image

st.set_page_config(page_title="Image Segmentation A2D2 Dataset", layout="wide")

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Image Segmentation for A2D2 Dataset')
with row0_2:
    st.text("")
    st.subheader('Streamlit App by Huize Zhang. View source code [here](https://github.com/Huizezhang/imagesegmentation).')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown(
        "This APP is based on the preview dataset of [A2D2 dataset](https://www.a2d2.audi/a2d2/en/dataset.html) from Audi. The dataset provides photos taken by on-board cameras and corresponding semantic segmentation information. The following functions are provided:")

    st.markdown(
        "1. Classify and label objects on photos.")
    st.markdown(
        "2. A pie chart showing the proportion of different objects occupying the picture.")
    st.markdown(
        "3. A histogram representing the number of important objects in the picture.")
    st.markdown(
        "3. A histogram representing the number of important objects in the picture.")



output_folder_path = './result'
save_json = os.path.join(output_folder_path, 'scene_id_list.json')
with open(save_json, 'r') as in_file:
    scene_id_list = jsonpickle.decode(in_file.read())

selected_scene = st.selectbox('Select Scene', scene_id_list)
class_name = set(list(class_list.values()))
selected_class = st.selectbox('Select Class', list(class_name))

scene_index = scene_id_list.index(selected_scene)
class_count = {"Car": 0, "Bicycle": 0, "Pedestrian": 0, "Truck": 0, "Small vehicles": 0,
               "Traffic signal": 0, "Traffic sign": 0, "Animals": 0, "Utility vehicle": 0}

for scene_id in scene_id_list:
    if scene_id != selected_scene:
        continue

    save_json = os.path.join(os.path.join(output_folder_path, scene_id), f'{scene_id}.json')
    with open(save_json, 'r') as in_file:
        scene = jsonpickle.decode(in_file.read())


    pil_image = Image.open(scene.image_path)
    image_origin = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    class_dict = collections.defaultdict(float)

    for obj in scene.objectlist:
        if class_count.get(obj.classname) is not None:
            class_count[obj.classname] += 1
        class_dict[obj.classname] += obj.percentage * 100
        if selected_class == obj.classname:
            color = hsv_to_rgb_cv(obj.color_hsv)
            color = tuple(color.tolist())
            merged_contour = []
            merged_contour = np.concatenate(obj.contours)
            for contour in obj.contours:
                cv2.drawContours(image_origin, [contour], 0, color, 2)
            x, y, w, h = cv2.boundingRect(merged_contour)
            text = f'{obj.classname}{obj.object_id}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            color = (0, 0, 255)
            text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
            text_x = int((x + x+w) / 2 - text_size[0] / 2)
            text_y = y - 10  # 将文本的起始坐标设置在矩形的上方
            cv2.putText(image_origin, text, (text_x, text_y), font, fontScale, color, thickness, cv2.LINE_AA)

            cv2.rectangle(image_origin, (x, y), (x + w, y + h), (255,255,255), 1)
    image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
    st.image(image_origin, caption='Image for {}'.format(scene.image_path), use_column_width=True)
    labels = list(class_dict.keys())
    values = list(class_dict.values())
    new_sizes, new_labels = merge_small_parts(values, labels, 1)
    total = sum(new_sizes)
    normalized = [x / total for x in new_sizes]
    normalized_sum = sum(normalized)
    factor = 100.0 / normalized_sum
    normalized_values = [x * factor for x in normalized]
    fig = go.Figure(data=[go.Pie(labels=new_labels, values=normalized_values)])
    fig.update_layout(title='Piechart with categories and percentages')
    st.plotly_chart(fig, use_column_width=True)

    class_count = {key: value for key, value in class_count.items() if value != 0}
    data = [go.Bar(x=list(class_count.keys()), y=list(class_count.values()))]

    # 设置布局
    layout = go.Layout(title='柱状图', xaxis=dict(title='类别'), yaxis=dict(title='数据'))

    st.plotly_chart(data, use_column_width=True)
