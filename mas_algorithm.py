# -*- coding: utf-8 -*-

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterString,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterVectorDestination,
    QgsProcessingException
)

import os
from .mas_utils import executeWbt, validateRasterInput, ensureOutputDirectory


class MasBaseAlgorithm(QgsProcessingAlgorithm):
    """Base class for all MAS Geospatial algorithms."""
    
    # Override these in subclasses
    WBT_TOOL_NAME = ''
    ALGORITHM_NAME = ''
    DISPLAY_NAME = ''
    GROUP = ''
    GROUP_ID = ''
    SHORT_DESCRIPTION = ''
    
    def __init__(self):
        super().__init__()
        self.plugin_dir = os.path.dirname(__file__)
    
    def createInstance(self):
        """Create new instance of algorithm."""
        return self.__class__()
    
    def name(self):
        """Return algorithm name."""
        return self.ALGORITHM_NAME
    
    def displayName(self):
        """Return algorithm display name."""
        return self.DISPLAY_NAME
    
    def group(self):
        """Return algorithm group."""
        return self.GROUP
    
    def groupId(self):
        """Return algorithm group ID."""
        return self.GROUP_ID
    
    def shortHelpString(self):
        """Return algorithm help."""
        return self.SHORT_DESCRIPTION
    
    def icon(self):
        """Return algorithm icon."""
        icon_path = os.path.join(
            os.path.dirname(self.plugin_dir),
            'icons',
            'mas_icon.svg'
        )
        return QIcon(icon_path)
    
    def tr(self, string):
        """Get translation."""
        return QCoreApplication.translate(self.__class__.__name__, string)


# Example implementations for specific tool categories

class GeomorphometricAlgorithm(MasBaseAlgorithm):
    """Base for geomorphometric analysis algorithms."""
    
    GROUP = 'Geomorphometric Analysis'
    GROUP_ID = 'geomorphometric'


class HydrologicalAlgorithm(MasBaseAlgorithm):
    """Base for hydrological analysis algorithms."""
    
    GROUP = 'Hydrological Analysis'
    GROUP_ID = 'hydrological'


class StreamNetworkAlgorithm(MasBaseAlgorithm):
    """Base for stream network analysis algorithms."""
    
    GROUP = 'Stream Network Analysis'
    GROUP_ID = 'stream_network'
