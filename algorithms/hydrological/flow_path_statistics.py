# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_path_statistics.py
Unified flow path statistics tool
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException
)
from ...core.flow_algorithms import FlowRouter

class FlowPathStatisticsAlgorithm(QgsProcessingAlgorithm):
    """Unified flow path statistics tool."""
    
    INPUT_DEM = 'INPUT_DEM'
    INPUT_DIR = 'INPUT_DIR'
    INPUT_STREAMS = 'INPUT_STREAMS'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'Downslope Distance to Stream',
        'Elevation Above Stream (HAND)',
        'Max Upslope Flowpath Length',
        'Downslope Flowpath Length',
        'Flow Length Difference',
        'Longest Flowpath',
        'Number of Inflowing Neighbours',
        'Number of Upslope Neighbours',
        'Number of Downslope Neighbours',
        'Average Flowpath Slope (Placeholder)',
        'Max Downslope Elev Change (Placeholder)',
        'Min Downslope Elev Change (Placeholder)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowPathStatisticsAlgorithm()
    
    def name(self):
        return 'flow_path_statistics'
    
    def displayName(self):
        return 'Flow Path Statistics'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate flow path statistics relative to a stream network.
        
        - Downslope Distance to Stream: Distance along flow path to nearest stream cell.
        - Elevation Above Stream (HAND): Height difference between cell and nearest downslope stream cell.
        - Max Upslope Flowpath Length: Distance to ridge.
        - Downslope Flowpath Length: Distance to outlet.
        - Flow Length Difference: Upslope - Downslope.
        - Longest Flowpath: Upslope + Downslope.
        - Number of Inflowing Neighbours: Count of cells flowing into this cell.
        - Number of Upslope Neighbours: Count of all cells upslope (Flow Accumulation - 1).
        - Number of Downslope Neighbours: Count of cells receiving flow (1 for D8).
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DIR,
                'D8 Flow Direction'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STREAMS,
                'Stream Raster',
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                'Input DEM (Required for HAND)',
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                'Statistic',
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
            dir_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DIR, context)
            stream_layer = self.parameterAsRasterLayer(parameters, self.INPUT_STREAMS, context)
            dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if dir_layer is None:
                raise QgsProcessingException('Flow Direction is required')
            
            # Check requirements
            if method_idx in [0, 1] and stream_layer is None:
                 raise QgsProcessingException('Stream Raster is required for this method')
                 
            if method_idx == 1 and dem_layer is None:
                raise QgsProcessingException('DEM is required for Elevation Above Stream (HAND)')
            
            feedback.pushInfo('Loading data...')
            # Initialize with DEM if available, else just Flow Dir source for dimensions
            # But FlowRouter expects DEM array.
            # If DEM not provided, we might need to load Flow Dir as DEM?
            # No, FlowRouter needs DEM for some calcs, but for others just dimensions.
            # Let's load Flow Dir as "dem" if DEM is missing, but be careful.
            
            if dem_layer:
                router = FlowRouter(dem_layer.source().dataProvider().block(0, dem_layer.extent(), dem_layer.width(), dem_layer.height()).data(), 
                                  dem_layer.rasterUnitsPerPixelX(), dem_layer.rasterUnitsPerPixelY())
                # Re-load proper DEM array
                # Wait, FlowRouter takes array.
                # Let's use the standard way:
                # We need a way to init FlowRouter without DEM if we only have FlowDir?
                # For now, let's assume DEM is passed or we use FlowDir as dummy DEM
                pass
            
            # We need to handle the initialization properly.
            # If DEM is not provided, we can't init FlowRouter correctly if it expects DEM.
            # Let's require DEM for all for simplicity, or handle it.
            # Actually, FlowRouter.__init__ takes dem_array.
            
            if dem_layer:
                provider = dem_layer.dataProvider()
                dem_array = provider.block(0, dem_layer.extent(), dem_layer.width(), dem_layer.height()).data()
                # Handle nodata
                nodata = provider.sourceNoDataValue(1)
                router = FlowRouter(dem_array, dem_layer.rasterUnitsPerPixelX(), dem_layer.rasterUnitsPerPixelY(), nodata)
            else:
                # Use Flow Dir as dummy DEM
                provider = dir_layer.dataProvider()
                dem_array = provider.block(0, dir_layer.extent(), dir_layer.width(), dir_layer.height()).data()
                nodata = provider.sourceNoDataValue(1)
                router = FlowRouter(dem_array, dir_layer.rasterUnitsPerPixelX(), dir_layer.rasterUnitsPerPixelY(), nodata)
            
            # Load Flow Dir
            provider_dir = dir_layer.dataProvider()
            flow_dir = provider_dir.block(0, dir_layer.extent(), dir_layer.width(), dir_layer.height()).data()
            
            streams = None
            if stream_layer:
                provider_streams = stream_layer.dataProvider()
                streams = provider_streams.block(0, stream_layer.extent(), stream_layer.width(), stream_layer.height()).data()
            
            method_map = {
                0: 'downslope_distance',
                1: 'hand',
                2: 'max_upslope_length',
                3: 'downslope_length',
                4: 'flow_length_diff',
                5: 'longest_flowpath',
                6: 'num_inflowing',
                7: 'num_upslope',
                8: 'num_downslope',
                9: 'avg_flowpath_slope',
                10: 'max_downslope_elev_change',
                11: 'min_downslope_elev_change'
            }
            
            stat_type = method_map.get(method_idx, 'downslope_distance')
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            feedback.setProgress(20)
            
            result = router.calculate_flow_path_statistics(stat_type, flow_dir, streams)
            
            feedback.pushInfo('Saving output...')
            feedback.setProgress(80)
            
            router.save_raster(output_path, result, nodata=-9999.0)
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error: {str(e)}')
