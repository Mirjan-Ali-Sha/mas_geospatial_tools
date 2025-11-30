# Copy and paste this ENTIRE block into the QGIS Python Console
# (Plugins → Python Console)

print("=== MAS Geospatial Tools Diagnostic ===")
print()

# Test 1: Check if provider is registered
from qgis.core import QgsApplication
registry = QgsApplication.processingRegistry()
providers = registry.providers()
print(f"Total providers registered: {len(providers)}")
mas_provider = None
for p in providers:
    if 'mas' in p.id().lower():
        print(f"✓ Found MAS provider: {p.id()} - {p.name()}")
        mas_provider = p
        break
if not mas_provider:
    print("✗ MAS Geospatial Tools provider NOT found in registry!")
print()

# Test 2: Try to get algorithms from the provider
if mas_provider:
    try:
        algs = mas_provider.algorithms()
        print(f"Algorithms from provider: {len(algs)}")
        if len(algs) == 0:
            print("✗ Provider has ZERO algorithms!")
            print("  This means loadAlgorithms() might not be working")
        else:
            print("✓ Provider has algorithms:")
            for alg in algs[:5]:  # Show first 5
                print(f"  - {alg.id()}: {alg.displayName()}")
            if len(algs) > 5:
                print(f"  ... and {len(algs) - 5} more")
    except Exception as e:
        print(f"✗ Error getting algorithms: {e}")
        import traceback
        traceback.print_exc()
print()

# Test 3: Try to manually instantiate the provider
print("Testing manual provider instantiation...")
try:
    from mas_geospatial_tools.mas_provider import MasGeospatialProvider
    test_provider = MasGeospatialProvider()
    print(f"✓ Provider created manually")
    print(f"  Provider ID: {test_provider.id()}")
    print(f"  Provider name: {test_provider.name()}")
    print(f"  Algorithms in __init__: {len(test_provider.algorithms)}")
    
    # Try to load algorithms
    test_provider.loadAlgorithms()
    algs = list(test_provider.algorithms())
    print(f"  Algorithms after loadAlgorithms(): {len(algs)}")
    
except Exception as e:
    print(f"✗ Failed to create provider manually: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== Diagnostic Complete ===")
