
import sys
import os
import traceback

# Add plugin parent path
plugin_parent_path = r'c:/Users/acer/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins'
if plugin_parent_path not in sys.path:
    sys.path.append(plugin_parent_path)

print("Testing core imports...")

try:
    print("Importing core.dem_utils...")
    from mas_geospatial_tools.core import dem_utils
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

try:
    print("Importing core.flow_algorithms...")
    from mas_geospatial_tools.core import flow_algorithms
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

try:
    print("Importing core.morphometry...")
    from mas_geospatial_tools.core import morphometry
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
