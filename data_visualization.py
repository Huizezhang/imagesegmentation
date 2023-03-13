import collections
import os
import jsonpickle as jsonpickle
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from Segmentation.helper import hsv_to_rgb_cv, merge_small_parts
from Segmentation.Objectclass import class_list



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
        "3. A histogram representing the number of important objects(e.g. cars or pedestrians) in the picture.")
    st.markdown(
        "4. A heat map showing how the objects(e.g. cars or pedestrians) in the image are distributed across all data.")
    st.markdown(
        "5. A histogram showing how objects(e.g. cars or pedestrians) are distributed in the data set.")



output_folder_path = './result'
save_json = os.path.join(output_folder_path, 'scene_id_list.json')
with open(save_json, 'r') as in_file:
    scene_id_list = jsonpickle.decode(in_file.read())

with row3_1:
    class_name = set(list(class_list.values()))
    selected_scene = st.selectbox('Select Scene', scene_id_list)
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
    scene.image_path = scene.image_path.replace("\\", "/")
    image_origin = cv2.imread(scene.image_path)
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
            text_y = y - 10
            #cv2.putText(image_origin, text, (text_x, text_y), font, fontScale, color, thickness, cv2.LINE_AA)

            cv2.rectangle(image_origin, (x, y), (x + w, y + h), (255,255,255), 1)

    image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
    with row3_1:
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
    with row3_1:
        st.plotly_chart(fig, use_column_width=True)

    class_count = {key: value for key, value in class_count.items() if value != 0}
    data = [go.Bar(x=list(class_count.keys()), y=list(class_count.values()))]


    layout = go.Layout(title='histogram', xaxis=dict(title='histogram'), yaxis=dict(title='histogram'))
    with row3_1:
        st.plotly_chart(data, use_column_width=True)

with row3_1:
    class_name = set(list(class_list.values()))
    selected_class2 = st.selectbox('Select Class for heatmap', list(class_name))
    selected_class3 = st.selectbox('Select Class for histogram', list(class_name))

heatmapdata = list()
histogramdata = collections.defaultdict(int)
for scene_id in scene_id_list:
    for hex_color in class_list:
        if class_list[hex_color] == selected_class2:
            save_masks = os.path.join(os.path.join(output_folder_path, scene_id), "masks")
            mask_path = os.path.join(save_masks,f"{selected_class2}_{hex_color}.png")
            mask_path = mask_path.replace("\\", "/")
            img = cv2.imread(mask_path)
            gaussian_blur = cv2.GaussianBlur(img, (11,11), 5)
            img_g = cv2.cvtColor(cv2.rotate(gaussian_blur, cv2.ROTATE_180), cv2.COLOR_BGR2GRAY)
            heatmapdata.append(img_g)
    save_json = os.path.join(os.path.join(output_folder_path, scene_id), f'{scene_id}.json')
    with open(save_json, 'r') as in_file:
        scene = jsonpickle.decode(in_file.read())
    count = 0
    for obj in scene.objectlist:
        if obj.classname == selected_class3:
            count += 1


    histogramdata[count] += 1



heatmapdata = np.sum(heatmapdata,axis=0)

heatmap = go.Figure(data=go.Heatmap(z=heatmapdata, colorscale="Jet"))

with row3_1:
    st.plotly_chart(heatmap)
    data = [go.Bar(x=list(histogramdata.keys()), y=list(histogramdata.values()))]
    st.plotly_chart(data, use_column_width=True)

