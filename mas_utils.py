# -*- coding: utf-8 -*-

import os
from pathlib import Path
from qgis.core import (
    Qgis,
    QgsMessageLog,
    QgsProcessingFeedback,
    QgsProcessingException
)

def validateRasterInput(layer_path):
    """Validate raster input file.
    
    Args:
        layer_path (str): Path to raster file
        
    Returns:
        bool: True if valid
    """
    return Path(layer_path).exists() and Path(layer_path).is_file()


def ensureOutputDirectory(output_path):
    """Ensure output directory exists.
    
    Args:
        output_path (str): Output file path
        
    Returns:
        bool: True if directory exists or was created
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.exists()
