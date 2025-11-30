
import os
import zipfile
import sys

def package_plugin():
    # Plugin directory is the current directory
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_name = os.path.basename(plugin_dir)
    
    # Output zip file
    zip_filename = f"{plugin_name}.zip"
    zip_path = os.path.join(plugin_dir, zip_filename)
    
    print(f"Packaging plugin '{plugin_name}' into '{zip_filename}'...")
    
    # Files and directories to exclude
    excludes = [
        '__pycache__',
        '.git',
        '.idea',
        '.vscode',
        '.gitignore',
        zip_filename,
        'package_plugin.py',
        'plugin_structure.txt',
        'verify_all_tools.py',
        'debug_core_imports.py',
        'check_imports.py',
        'test_algorithms.py',
        'test_imports.py',
        'qgis_diagnostic.py',
        'generate_readme_table.py',
        'LOG_INSTRUCTIONS.txt',
        'function_list.txt',
        'mas_algorithm.py', # Obsolete
        'mas_utils.py',     # Obsolete
        'hydrology_struct_old', # Old version folder
        'tests'
    ]
    
    # Extensions to exclude
    exclude_exts = ['.pyc', '.pyo', '.pyd', '.zip']
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(plugin_dir):
                # Remove excluded directories
                dirs[:] = [d for d in dirs if d not in excludes]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, plugin_dir)
                    
                    # Check exclusions
                    if any(rel_path.startswith(ex) or rel_path == ex for ex in excludes):
                        continue
                        
                    _, ext = os.path.splitext(file)
                    if ext in exclude_exts:
                        continue
                        
                    # Add to zip
                    # The archive should contain the plugin folder at the root
                    arcname = os.path.join(plugin_name, rel_path)
                    print(f"  Adding: {rel_path}")
                    zipf.write(file_path, arcname)
                    
        print(f"\nSuccessfully created '{zip_filename}'")
        print(f"Location: {zip_path}")
        
    except Exception as e:
        print(f"Error creating zip file: {e}")

if __name__ == '__main__':
    package_plugin()
