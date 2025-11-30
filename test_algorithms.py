# Test script to check if algorithms can be instantiated
# Run this in the QGIS Python Console

try:
    from mas_geospatial_tools.algorithms.geomorphometric.hillshade_algorithms import HillshadeAlgorithm
    print("✓ HillshadeAlgorithm imported")
    alg = HillshadeAlgorithm()
    print(f"✓ HillshadeAlgorithm instantiated: {alg.name()}")
except Exception as e:
    print(f"✗ HillshadeAlgorithm failed: {e}")

try:
    from mas_geospatial_tools.algorithms.geomorphometric.slope_algorithms import SlopeAlgorithm
    print("✓ SlopeAlgorithm imported")
    alg = SlopeAlgorithm()
    print(f"✓ SlopeAlgorithm instantiated: {alg.name()}")
except Exception as e:
    print(f"✗ SlopeAlgorithm failed: {e}")

try:
    from mas_geospatial_tools.algorithms.hydrological.flow_direction import D8FlowDirectionAlgorithm
    print("✓ D8FlowDirectionAlgorithm imported")
    alg = D8FlowDirectionAlgorithm()
    print(f"✓ D8FlowDirectionAlgorithm instantiated: {alg.name()}")
except Exception as e:
    print(f"✗ D8FlowDirectionAlgorithm failed: {e}")

try:
    from mas_geospatial_tools.mas_provider import MasGeospatialProvider
    print("✓ MasGeospatialProvider imported")
    provider = MasGeospatialProvider()
    print(f"✓ Provider instantiated, ID: {provider.id()}")
    print(f"  Algorithm count in __init__: {len(provider.algorithms)}")
except Exception as e:
    print(f"✗ MasGeospatialProvider failed: {e}")
    import traceback
    traceback.print_exc()
