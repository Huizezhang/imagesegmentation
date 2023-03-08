import subprocess

command = 'python -m streamlit run data_visualization.py'

subprocess.call(command, shell=True)