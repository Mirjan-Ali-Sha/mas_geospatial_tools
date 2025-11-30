
import os
import re

plugin_dir = r'c:/Users/acer/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/mas_geospatial_tools'

def check_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if 'QgsProcessingParameterEnum' in content:
        if 'QgsProcessingParameterEnum' not in re.findall(r'from qgis.core import .*', content, re.DOTALL)[0]:
            # Check if it's imported individually or in a multi-line import
            if 'QgsProcessingParameterEnum' not in content.split('from qgis.core import')[1].split(')')[0]:
                 return False
    return True

def check_imports():
    missing = []
    for root, dirs, files in os.walk(plugin_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'QgsProcessingParameterEnum(' in content:
                        # Check if imported
                        if 'QgsProcessingParameterEnum' not in content.split('class ')[0]:
                            missing.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    
    return missing

if __name__ == '__main__':
    missing_files = check_imports()
    if missing_files:
        print("Files missing QgsProcessingParameterEnum import:")
        for f in missing_files:
            print(f)
    else:
        print("No missing imports found.")
