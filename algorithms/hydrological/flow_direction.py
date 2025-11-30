# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_direction.py
Native D8 Flow Direction implementation
"""

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingException
)
from osgeo import gdal
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter

class FlowDirectionAlgorithm(QgsProcessingAlgorithm):
    """Unified Flow Direction tool (D8, D-Inf, Rho8)."""
    
    INPUT = 'INPUT'
    METHOD = 'METHOD'
    OUTPUT = 'OUTPUT'
    
    METHOD_OPTIONS = [
        'D8 (Deterministic 8-neighbor)', 
        'D-Infinity (Tarboton 1997)', 
        'Rho8 (Stochastic 8-neighbor)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowDirectionAlgorithm()
    
    def name(self):
        return 'flow_direction'
    
    def displayName(self):
        return 'Flow Direction'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate flow direction from DEM.
        
        Methods:
        - D8: Routes flow to steepest downslope neighbor (1-128 codes).
        - D-Infinity: Routes flow to single angle (radians).
        - Rho8: Stochastic D8 to break parallel lines.
        """
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                'Input DEM'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                'Method',
                options=self.METHOD_OPTIONS,
                defaultValue=0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                'Output flow direction'
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        try:
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            
            feedback.setProgress(20)
            
            # Initialize router
            router = FlowRouter(processor.array, processor.cellsize_x)
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            
            if method_idx == 0: # D8
                flow_dir = router.d8_flow_direction()
                dtype = gdal.GDT_Int32
            elif method_idx == 1: # D-Inf
                flow_dir = router.dinf_flow_direction()
                dtype = gdal.GDT_Float32
            elif method_idx == 2: # Rho8
                flow_dir = router.rho8_flow_direction()
                dtype = gdal.GDT_Int32
            
            feedback.setProgress(80)
            
            feedback.pushInfo('Saving output...')
            processor.save_raster(output_path, flow_dir, dtype=dtype)
            processor.close()
            
            feedback.setProgress(100)
            
            return {self.OUTPUT: output_path}
            
        except Exception as e:
            raise QgsProcessingException(f'Error processing flow direction: {str(e)}')
