# -*- coding: utf-8 -*-
"""
algorithms/hydrology_struct/structured_tools_old.py
Structured Hydrology tools mimicking standard workflow (Old Version).
"""

from qgis.core import QgsProcessingAlgorithm, QgsProcessingParameterEnum
from ..hydrological.basin_analysis import BasinAnalysisAlgorithm
from ..hydrological.depression_algorithms import DepressionHandlingAlgorithm
from ..hydrological.flow_accumulation import FlowAccumulationAlgorithm
from ..hydrological.flow_direction import FlowDirectionAlgorithm
from ..hydrological.flow_distance import FlowDistanceAlgorithm
from ..hydrological.flow_length import FlowLengthAlgorithm
from ..hydrological.sink_analysis import SinkAnalysisAlgorithm
from ..hydrological.snap_pour_points import SnapPourPointsAlgorithm
from ..hydrological.watershed_delineation import WatershedDelineationAlgorithm
from ..stream_network.stream_link_analysis import StreamLinkAlgorithm
from ..stream_network.stream_ordering import StreamOrderingAlgorithm
from ..stream_network.vector_stream_network import VectorStreamNetworkAlgorithm

class StructBasinAlgorithm(BasinAnalysisAlgorithm):
    def name(self): return 'struct_basin_old'
    def displayName(self): return 'Basin (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructBasinAlgorithm()

class StructFillAlgorithm(DepressionHandlingAlgorithm):
    def name(self): return 'struct_fill_old'
    def displayName(self): return 'Fill (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructFillAlgorithm()
    
    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)
        # Default to Fill (Method 0)
        if self.parameterDefinition(self.METHOD):
            self.parameterDefinition(self.METHOD).setDefaultValue(0)

class StructFlowAccumulationAlgorithm(FlowAccumulationAlgorithm):
    def name(self): return 'struct_flow_accumulation_old'
    def displayName(self): return 'Flow Accumulation (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructFlowAccumulationAlgorithm()
    
    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)
        # Default to D8 (Method 0)
        if self.parameterDefinition(self.METHOD):
            self.parameterDefinition(self.METHOD).setDefaultValue(0)

class StructFlowDirectionAlgorithm(FlowDirectionAlgorithm):
    def name(self): return 'struct_flow_direction_old'
    def displayName(self): return 'Flow Direction (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructFlowDirectionAlgorithm()
    
    def initAlgorithm(self, config=None):
        super().initAlgorithm(config)
        # Default to D8 (Method 0)
        if self.parameterDefinition(self.METHOD):
            self.parameterDefinition(self.METHOD).setDefaultValue(0)

class StructFlowDistanceAlgorithm(FlowDistanceAlgorithm):
    def name(self): return 'struct_flow_distance_old'
    def displayName(self): return 'Flow Distance (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructFlowDistanceAlgorithm()

class StructFlowLengthAlgorithm(FlowLengthAlgorithm):
    def name(self): return 'struct_flow_length_old'
    def displayName(self): return 'Flow Length (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructFlowLengthAlgorithm()

class StructSinkAlgorithm(SinkAnalysisAlgorithm):
    def name(self): return 'struct_sink_old'
    def displayName(self): return 'Sink (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructSinkAlgorithm()

class StructSnapPourPointAlgorithm(SnapPourPointsAlgorithm):
    def name(self): return 'struct_snap_pour_point_old'
    def displayName(self): return 'Snap Pour Point (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructSnapPourPointAlgorithm()

class StructStreamLinkAlgorithm(StreamLinkAlgorithm):
    def name(self): return 'struct_stream_link_old'
    def displayName(self): return 'Stream Link (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructStreamLinkAlgorithm()

class StructStreamOrderAlgorithm(StreamOrderingAlgorithm):
    def name(self): return 'struct_stream_order_old'
    def displayName(self): return 'Stream Order (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructStreamOrderAlgorithm()

class StructStreamToFeatureAlgorithm(VectorStreamNetworkAlgorithm):
    def name(self): return 'struct_stream_to_feature_old'
    def displayName(self): return 'Stream to Feature (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructStreamToFeatureAlgorithm()

class StructWatershedAlgorithm(WatershedDelineationAlgorithm):
    def name(self): return 'struct_watershed_old'
    def displayName(self): return 'Watershed (Old)'
    def group(self): return 'Hydrological Analysis - Struct Old'
    def groupId(self): return 'hydrology_struct_old'
    def createInstance(self): return StructWatershedAlgorithm()
