import streamlit as st
import plotly.graph_objs as go
import pandas as pd

class_count = {"Car": 0, "Bicycle": 0, "Pedestrian": 0, "Truck": 0, "Small vehicles": 0,
               "Traffic signal": 0, "Traffic sign": 0, "Animals": 0, "Utility vehicle": 0}

class_count = {key: value for key, value in class_count.items() if value != 0}


print(class_count)