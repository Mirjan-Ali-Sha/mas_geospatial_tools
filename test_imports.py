import os
import sys
import importlib.util
import traceback

# Add the parent directory of mas_geospatial_tools to sys.path
# Assuming this script is inside mas_geospatial_tools
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Scanning directory: {current_dir}")
print("-" * 60)

failed_imports = []

for root, dirs, files in os.walk(current_dir):
    for file in files:
        if file.endswith(".py") and file != "test_imports.py":
            file_path = os.path.join(root, file)
            
            # Construct module name
            rel_path = os.path.relpath(file_path, parent_dir)
            module_name = rel_path.replace(os.sep, ".").replace(".py", "")
            
            print(f"Testing import: {module_name}...", end=" ")
            
            try:
                # We use importlib to try and import the module
                if module_name in sys.modules:
                    del sys.modules[module_name] # Force reload
                
                importlib.import_module(module_name)
                print("OK")
            except Exception as e:
                print("FAILED!")
                print(f"  Error: {e}")
                failed_imports.append((module_name, str(e)))
                # traceback.print_exc() # Optional: print full traceback

print("-" * 60)
if failed_imports:
    print(f"Found {len(failed_imports)} failed imports:")
    for mod, err in failed_imports:
        print(f"  - {mod}: {err}")
else:
    print("All modules imported successfully!")
