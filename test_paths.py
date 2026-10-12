import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(BASE_DIR, 'templates')
print(f"BASE_DIR: {BASE_DIR}")
print(f"template_folder: {template_path}")
print(f"templates exists: {os.path.exists(template_path)}")
print(f"predict.html exists: {os.path.exists(os.path.join(template_path, 'predict.html'))}")
print(f"Files in templates: {os.listdir(template_path) if os.path.exists(template_path) else 'N/A'}")
