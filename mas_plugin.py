# -*- coding: utf-8 -*-

import os
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsApplication

from processing.core.ProcessingConfig import ProcessingConfig
from mas_geospatial_tools.mas_provider import MasGeospatialProvider


class MasGeospatialPlugin:
    """QGIS Plugin Implementation for MAS Geospatial Tools."""

    def __init__(self, iface):
        """Initialize the plugin.
        
        Args:
            iface: A QGIS interface instance
        """
        self.iface = iface
        self.provider = None
        self.plugin_dir = os.path.dirname(__file__)

    def initProcessing(self):
        """Initialize Processing provider."""
        self.provider = MasGeospatialProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        try:
            self.initProcessing()
        except Exception as e:
            from qgis.core import QgsMessageLog, Qgis
            QgsMessageLog.logMessage(f"MAS Plugin Init Error: {str(e)}", "MAS Geospatial Tools", Qgis.Critical)
            self.iface.messageBar().pushMessage("MAS Plugin", f"Init Error: {str(e)}", level=Qgis.Critical)
            raise e

    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        try:
            if self.provider:
                QgsApplication.processingRegistry().removeProvider(self.provider)
        except RuntimeError:
            # Provider might have been deleted already
            pass

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate('MasGeospatialPlugin', message)
