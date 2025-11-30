# -*- coding: utf-8 -*-
"""
algorithms/stream_network/stream_link_analysis.py
Unified stream link analysis tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
import numpy as np

class StreamLinkAlgorithm(QgsProcessingAlgorithm):
    """Unified stream link analysis tool."""
    
    INPUT_STREAMS = 'INPUT_STREAMS'
    INPUT_FLOW_DIR = 'INPUT_FLOW_DIR'
    INPUT_DEM = 'INPUT_DEM'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Stream Link Identifier',
        'Stream Link Length',
        'Stream Link Slope',
        'Stream Link Class',
        'Stream Slope Continuous'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return StreamLinkAlgorithm()
    
    def name(self):
        return 'stream_link_analysis'
    
    def displayName(self):
        return 'Stream Link Analysis'
    
    def group(self):
        return 'Stream Network Analysis'
    
    def groupId(self):
        return 'stream_network'
    
    def shortHelpString(self):
        return """
        Analyze stream links.
        
        - Identifier: Unique ID for each link.
        - Length: Length of each link.
        - Slope: Average slope of each link.
        - Class: Classification (Placeholder).
        - Slope Continuous: Local slope along stream.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Input Stream Raster'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_DIR,
                'Input Flow Direction'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                'Input DEM (Required for Slope)',
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                'Analysis Method',
                options=self.METHOD_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output Raster'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            streams_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            flow_dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_FLOW_DIR, context)
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if streams_layer is None or flow_dir_layer is None:
                raise QgsProcessingException('Invalid inputs')
            
            if method_idx == 2 and dem_layer is None:
                raise QgsProcessingException('DEM is required for Stream Link Slope')
            
            feedback.pushInfo('Loading data...')
            # Load DEM if available, else use streams for dimensions
            if dem_layer:
                provider = dem_layer.dataProvider()
                dem_array = provider.block(0, dem_layer.extent(), dem_layer.width(), dem_layer.height()).data()
                nodata = provider.sourceNoDataValue(1)
                router = FlowRouter(dem_array, dem_layer.rasterUnitsPerPixelX(), dem_layer.rasterUnitsPerPixelY(), nodata)
            else:
                # Use streams as dummy DEM
                provider = streams_layer.dataProvider()
                dem_array = provider.block(0, streams_layer.extent(), streams_layer.width(), streams_layer.height()).data()
                nodata = provider.sourceNoDataValue(1)
                router = FlowRouter(dem_array, streams_layer.rasterUnitsPerPixelX(), streams_layer.rasterUnitsPerPixelY(), nodata)
            
            # Load Flow Dir and Streams
            provider_dir = flow_dir_layer.dataProvider()
            flow_dir = provider_dir.block(0, flow_dir_layer.extent(), flow_dir_layer.width(), flow_dir_layer.height()).data()
            
            provider_streams = streams_layer.dataProvider()
            streams = provider_streams.block(0, streams_layer.extent(), streams_layer.width(), streams_layer.height()).data()
            
            method_map = {
                0: 'id',
                1: 'length',
                2: 'slope',
                3: 'class',
                4: 'slope_continuous'
            }
            
            stat_type = method_map.get(method_idx, 'id')
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(30)
            
            result = router.calculate_stream_link_statistics(stat_type, flow_dir, streams)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            router.save_raster(output_path, result, nodata=0)
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
