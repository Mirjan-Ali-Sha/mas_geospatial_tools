
import sys
import os
import traceback

# Add plugin path to sys.path
plugin_path = r'c:/Users/acer/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/mas_geospatial_tools'
if plugin_path not in sys.path:
    sys.path.append(plugin_path)

# Mock QGIS environment
from qgis.core import QgsApplication, QgsProcessingAlgorithm
from qgis.analysis import QgsNativeAlgorithms

QgsApplication.setPrefixPath("C:/Program Files/QGIS 3.28/apps/qgis", True)
qgs = QgsApplication([], False)
qgs.initQgis()

from mas_geospatial_tools.mas_geospatial_tools_provider import MasGeospatialToolsProvider

def verify_algorithms():
    provider = MasGeospatialToolsProvider()
    provider.loadAlgorithms()
    
    print(f"Loaded {len(provider.algorithms())} algorithms.")
    
    failed = []
    for alg in provider.algorithms():
        try:
            print(f"Verifying {alg.name()}...")
            # Check if createInstance works
            instance = alg.createInstance()
            if not instance:
                raise Exception("createInstance returned None")
            
            # Check if initAlgorithm works
            instance.initAlgorithm()
            
            print(f"  OK: {alg.displayName()}")
        except Exception as e:
            print(f"  FAILED: {alg.name()} - {str(e)}")
            traceback.print_exc()
            failed.append(alg.name())
            
    if failed:
        print(f"\nFailed algorithms ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll algorithms verified successfully!")
        sys.exit(0)

if __name__ == '__main__':
    verify_algorithms()
