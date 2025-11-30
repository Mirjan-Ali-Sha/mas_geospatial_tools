import os

# 1. Read function_list.txt
with open('function_list.txt', 'r') as f:
    lines = f.readlines()

tools = []
current_group = ""
for line in lines:
    line = line.strip()
    if line.endswith('/'):
        current_group = line[:-1]
    elif line and ' - ' in line:
        parts = line.split(' - ')
        tool_name = parts[1]
        tools.append({'group': current_group, 'name': tool_name})

# 2. Define Mapping (Manual mapping based on implementation knowledge)
# Key: Implemented Tool Name (Class Name or Display Name)
# Value: List of function_list.txt tool names
mapping = {
    "HillshadeAlgorithm": ["Hillshade", "MultidirectionalHillshade", "ShadowImage", "HypsometricallyTintedHillshade"],
    "SlopeAlgorithm": ["Slope", "StandardDeviationOfSlope"],
    "AspectAlgorithm": ["Aspect", "CircularVarianceOfAspect", "RelativeAspect"],
    "CurvatureAlgorithm": [
        "PlanCurvature", "ProfileCurvature", "TangentialCurvature", "MeanCurvature", 
        "GaussianCurvature", "TotalCurvature", "MaximalCurvature", "MinimalCurvature",
        "HorizontalExcessCurvature", "VerticalExcessCurvature", "DifferenceCurvature",
        "AccumulationCurvature", "Curvedness", "Unsphericity", "Rotor", "ShapeIndex", "RingCurvature",
        "MultiscaleCurvatures", "Profile"
    ],
    "RoughnessAlgorithm": ["RuggednessIndex", "MultiscaleRoughness", "EdgeDensity", "SurfaceAreaRatio", "MultiscaleRoughnessSignature"],
    "TPIAlgorithm": ["RelativeTopographicPosition", "MultiscaleTopographicPositionImage", "TopographicPositionAnimation"],
    "FeatureDetectionAlgorithm": ["FindRidges", "BreaklineMapping", "EmbankmentMapping", "MapOffTerrainObjects", "RemoveOffTerrainObjects", "Geomorphons"],
    "HypsometricAnalysisAlgorithm": ["HypsometricAnalysis", "LocalHypsometricAnalysis"],
    "VisibilityAlgorithm": ["Viewshed", "VisibilityIndex", "HorizonAngle", "TimeInDaylight", "ShadowAnimation", "TopoRender"],
    "DirectionalAnalysisAlgorithm": ["DirectionalRelief", "ExposureTowardsWindFlux", "FetchAnalysis", "AverageNormalVectorAngularDeviation", "MaxAnisotropyDev", "MaxAnisotropyDevSignature"],
    "OpennessAlgorithm": ["Openness", "SphericalStdDevOfNormals", "MultiscaleStdDevNormalsSignature"],
    "MultiscaleAnalysisAlgorithm": ["MultiscaleElevationPercentile", "MultiscaleStdDevNormals", "GaussianScaleSpace"],
    "StatisticalAlgorithms": ["LocalQuadraticRegression", "FeaturePreservingSmoothing", "SmoothVegetationResidual", "DevFromMeanElev", "DiffFromMeanElev", "MaxDifferenceFromMean", "MaxElevationDeviation", "PercentElevRange", "ElevRelativeToMinMax", "ElevRelativeToWatershedMinMax", "PennockLandformClass", "GeneratingFunction", "MaxElevDevSignature"],
    "WetnessAlgorithm": ["WetnessIndex"],
    
    # Hydrological
    "D8FlowDirectionAlgorithm": ["D8Pointer"],
    "D8FlowAccumulationAlgorithm": ["D8FlowAccumulation", "D8MassFlux", "FlowAccumulationFullWorkflow"],
    "WatershedDelineationAlgorithm": ["Watershed", "Basins", "Subbasins", "Isobasins", "StochasticDepressionAnalysis"],
    "DepressionHandlingAlgorithm": ["FillDepressions", "BreachDepressions", "BreachDepressionsLeastCost", "FillSingleCellPits", "BreachSingleCellPits", "FillBurn", "FillDepressionsPlanchonAndDarboux", "FillDepressionsWangAndLiu", "DemVoidFilling", "FillMissingData"],
    "FlowIndicesAlgorithm": ["StreamPowerIndex", "SedimentTransportIndex"],
    "FlowRoutingAlgorithm": ["DInfPointer", "DInfFlowAccumulation", "DInfMassFlux", "MDInfFlowAccumulation", "FD8Pointer", "FD8FlowAccumulation", "Rho8Pointer", "Rho8FlowAccumulation", "QinFlowAccumulation", "QuinnFlowAccumulation", "PilesjoHasan"],
    "FlowDistanceAlgorithm": ["DownslopeDistanceToStream", "ElevationAboveStream", "ElevationAboveStreamEuclidean", "DownslopeIndex"],
    "BasinAnalysisAlgorithm": ["UnnestBasins", "StrahlerOrderBasins"],
    "FlowPathStatisticsAlgorithm": ["AverageFlowpathSlope", "AverageUpslopeFlowpathLength", "MaxUpslopeFlowpathLength", "FlowLengthDiff", "TraceDownslopeFlowpaths", "NumInflowingNeighbours", "NumDownslopeNeighbours", "NumUpslopeNeighbours", "LongestFlowpath", "MaxDownslopeElevChange", "MinDownslopeElevChange", "MaxUpslopeElevChange"],
    "SinkAnalysisAlgorithm": ["Sink", "DepthInSink", "UpslopeDepressionStorage", "ImpoundmentSizeIndex"],
    "HydroEnforcementAlgorithm": ["BurnStreamsAtRoads", "RaiseWalls", "FlattenLakes", "InsertDams"],
    "SnapPourPointsAlgorithm": ["SnapPourPoints", "JensonSnapPourPoints"],
    "FlowLengthAlgorithm": ["DownslopeFlowpathLength", "MaxUpslopeValue"],
    "DemQualityAlgorithm": ["FindNoFlowCells", "EdgeContamination", "FindParallelFlow"],
    "HillslopesAlgorithm": ["Hillslopes"],
    "FloodOrderAlgorithm": ["FloodOrder"],
    "HydrologicConnectivityAlgorithm": ["HydrologicConnectivity", "DepthToWater", "LowPointsOnHeadwaterDivides"],
    
    # Stream Network
    "ExtractStreamsAlgorithm": ["ExtractStreams", "RasterizeStreams", "RiverCenterlines"],
    "StreamOrderingAlgorithm": ["StrahlerStreamOrder", "ShreveStreamMagnitude", "HortonStreamOrder", "HackStreamOrder", "TopologicalStreamOrder"],
    "StreamLinkAlgorithm": ["StreamLinkIdentifier", "StreamLinkLength", "StreamLinkSlope", "StreamLinkClass", "StreamSlopeContinuous"],
    "StreamNetworkAnalysisAlgorithm": ["DistanceToOutlet", "FarthestChannelHead", "FindMainStem", "TributaryIdentifier", "LengthOfUpstreamChannels", "MaxBranchLength"],
    "VectorStreamNetworkAlgorithm": ["RasterStreamsToVector", "VectorStreamNetworkAnalysis", "RepairStreamVectorTopology"],
    "StreamCleaningAlgorithm": ["RemoveShortStreams"],
    "ValleyExtractionAlgorithm": ["ExtractValleys"],
    "LongProfileAlgorithm": ["LongProfile", "LongProfileFromPoints"],
    "ContourAlgorithm": ["ContoursFromPoints", "ContoursFromRaster"],
    "PlottingAlgorithm": ["SlopeVsAspectPlot", "SlopeVsElevationPlot"],
    "AssessRouteAlgorithm": ["AssessRoute"],
}

# 3. Generate Table
print("| Group | Our Implemented Tool | Function Count | Mapped Functions |")
print("|---|---|---|---|")

# Invert mapping to check coverage
covered_tools = set()
for impl_tool, mapped_list in mapping.items():
    for t in mapped_list:
        covered_tools.add(t)

# Group by our implementation
for impl_tool, mapped_list in mapping.items():
    # Find group from first mapped tool
    group = "Unknown"
    for t in mapped_list:
        for tool_def in tools:
            if tool_def['name'] == t:
                group = tool_def['group']
                break
        if group != "Unknown":
            break
            
    count = len(mapped_list)
    mapped_str = ", ".join(mapped_list)
    print(f"| {group} | {impl_tool} | {count} | {mapped_str} |")

print("\n\n### Missing Tools Coverage")
missing_count = 0
for tool in tools:
    if tool['name'] not in covered_tools:
        print(f"- {tool['group']}: {tool['name']}")
        missing_count += 1

print(f"\nTotal Tools: {len(tools)}")
print(f"Covered Tools: {len(covered_tools)}")
print(f"Missing Tools: {missing_count}")
