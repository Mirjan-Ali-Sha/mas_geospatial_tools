# -*- coding: utf-8 -*-
"""
algorithms/hydrological/flow_accumulation.py
Native D8 Flow Accumulation implementation
"""

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingException
)
from ...core.dem_utils import DEMProcessor
from ...core.flow_algorithms import FlowRouter
import numpy as np


class FlowAccumulationAlgorithm(QgsProcessingAlgorithm):
    """Unified Flow Accumulation tool (D8, D-Inf, FD8)."""
    
    INPUT_DEM = 'INPUT_DEM'
    INPUT_WEIGHT = 'INPUT_WEIGHT'
    METHOD = 'METHOD'
    OUTPUT_FLOW_ACC = 'OUTPUT_FLOW_ACC'
    OUTPUT_FLOW_DIR = 'OUTPUT_FLOW_DIR'
    FILL_DEPRESSIONS = 'FILL_DEPRESSIONS'
    LOG_TRANSFORM = 'LOG_TRANSFORM'
    
    METHOD_OPTIONS = [
        'D8 (Deterministic 8-neighbor)', 
        'D-Infinity (Tarboton 1997)', 
        'FD8 (Freeman 1991)'
    ]
    
    def __init__(self):
        super().__init__()
    
    def createInstance(self):
        return FlowAccumulationAlgorithm()
    
    def name(self):
        return 'flow_accumulation'
    
    def displayName(self):
        return 'Flow Accumulation'
    
    def group(self):
        return 'Hydrological Analysis'
    
    def groupId(self):
        return 'hydrological'
    
    def shortHelpString(self):
        return """
        Calculate flow accumulation from DEM.
        
        Methods:
        - D8: Routes flow to steepest downslope neighbor.
        - D-Infinity: Routes flow to single angle (continuous).
        - FD8: Distributes flow to multiple downslope neighbors.
        """
    
    def initAlgorithm(self, config=None):
        # Input DEM
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
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
        
        # Optional weight raster
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_WEIGHT,
                'Weight raster (optional)',
                optional=True
            )
        )
        
        # Fill depressions option
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FILL_DEPRESSIONS,
                'Fill depressions before routing',
                defaultValue=True
            )
        )
        
        # Log transform option
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOG_TRANSFORM,
                'Apply log transform to output',
                defaultValue=False
            )
        )
        
        # Output flow accumulation
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_FLOW_ACC,
                'Output flow accumulation'
            )
        )
        
        # Output flow direction
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_FLOW_DIR,
                'Output flow direction (D8/D-Inf only)',
                optional=True
            )
        )
    
    def processAlgorithm(self, parameters, context, feedback):
        """Execute algorithm."""
        try:
            # Get parameters
            input_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
            weight_layer = self.parameterAsRasterLayer(parameters, self.INPUT_WEIGHT, context)
            method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
            fill_deps = self.parameterAsBool(parameters, self.FILL_DEPRESSIONS, context)
            log_transform = self.parameterAsBool(parameters, self.LOG_TRANSFORM, context)
            output_acc_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_FLOW_ACC, context)
            output_dir_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_FLOW_DIR, context)
            
            if input_layer is None:
                raise QgsProcessingException('Invalid input DEM')
            
            feedback.pushInfo('Loading DEM...')
            processor = DEMProcessor(input_layer.source())
            
            # Load weights if provided
            weights = None
            if weight_layer:
                feedback.pushInfo('Loading weight raster...')
                weight_processor = DEMProcessor(weight_layer.source())
                weights = weight_processor.array
                weight_processor.close()
            
            feedback.setProgress(10)
            
            # Initialize flow router
            router = FlowRouter(processor.array, processor.cellsize_x)
            
            # Fill depressions if requested
            if fill_deps:
                feedback.pushInfo('Filling depressions...')
                filled_dem = router.fill_depressions()
                router.dem = filled_dem
            
            feedback.setProgress(30)
            
            flow_dir = None
            
            feedback.pushInfo(f'Calculating {self.METHOD_OPTIONS[method_idx]}...')
            
            if method_idx == 0: # D8
                feedback.pushInfo('Calculating flow direction...')
                flow_dir = router.d8_flow_direction()
                feedback.pushInfo('Calculating flow accumulation...')
                flow_acc = router.d8_flow_accumulation(flow_dir, weights)
                
            elif method_idx == 1: # D-Inf
                feedback.pushInfo('Calculating flow direction...')
                flow_dir = router.dinf_flow_direction()
                feedback.pushInfo('Calculating flow accumulation...')
                flow_acc = router.dinf_flow_accumulation(flow_dir, weights)
                
            elif method_idx == 2: # FD8
                feedback.pushInfo('Calculating FD8 flow accumulation...')
                # FD8 doesn't produce a single flow direction raster
                flow_acc = router.fd8_flow_accumulation(weights)
            
            feedback.setProgress(70)
            
            # Apply log transform if requested
            if log_transform:
                feedback.pushInfo('Applying log transform...')
                flow_acc = np.log10(flow_acc + 1)
            
            # Save flow accumulation
            feedback.pushInfo('Saving flow accumulation...')
            processor.save_raster(output_acc_path, flow_acc)
            
            feedback.setProgress(85)
            
            # Save flow direction if requested and available
            if output_dir_path:
                if flow_dir is not None:
                    feedback.pushInfo('Saving flow direction...')
                    dtype = gdal.GDT_Int32 if method_idx == 0 else gdal.GDT_Float32
                    processor.save_raster(output_dir_path, flow_dir, dtype=dtype)
                else:
                    feedback.pushInfo('Flow direction not available for this method.')
            
            processor.close()
            
            feedback.pushInfo('Flow accumulation complete!')
            feedback.setProgress(100)
            
            results = {self.OUTPUT_FLOW_ACC: output_acc_path}
            if output_dir_path and flow_dir is not None:
                results[self.OUTPUT_FLOW_DIR] = output_dir_path
            
            return results
            
        except Exception as e:
            raise QgsProcessingException(f'Error in flow accumulation: {str(e)}')
