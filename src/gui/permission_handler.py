"""
Permission handler for managing camera access and privacy permissions.
"""

import os
import sys
import platform
from typing import Optional, Dict, Any
from PySide6.QtWidgets import QMessageBox, QApplication
from PySide6.QtCore import QObject, Signal

from ..utils.logger import logger


class PermissionHandler(QObject):
    """Handler for camera and privacy permissions."""
    
    permission_granted = Signal()
    permission_denied = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.camera_permission_granted = False
        self.privacy_permission_granted = False
        
        logger.info("Permission handler initialized")
    
    def check_camera_permission(self) -> bool:
        """Check if camera permission is granted."""
        try:
            # Check if camera is accessible
            import cv2
            
            # Try to open camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
                self.camera_permission_granted = True
                return True
            else:
                self.camera_permission_granted = False
                return False
                
        except Exception as e:
            logger.log_error_with_context(e, "camera_permission_check")
            self.camera_permission_granted = False
            return False
    
    def request_camera_permission(self, parent_widget=None) -> bool:
        """Request camera permission from user."""
        try:
            # Show permission request dialog
            msg_box = QMessageBox(parent_widget)
            msg_box.setWindowTitle("Camera Permission Required")
            msg_box.setText("This application needs access to your camera to monitor attentiveness.")
            msg_box.setInformativeText("Do you want to grant camera access?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            
            # Set icon
            msg_box.setIcon(QMessageBox.Question)
            
            # Show dialog
            result = msg_box.exec()
            
            if result == QMessageBox.Yes:
                # Test camera access
                if self.check_camera_permission():
                    self.camera_permission_granted = True
                    self.permission_granted.emit()
                    logger.info("Camera permission granted")
                    return True
                else:
                    self.camera_permission_granted = False
                    self.permission_denied.emit("Camera access failed")
                    logger.warning("Camera permission denied - camera not accessible")
                    return False
            else:
                self.camera_permission_granted = False
                self.permission_denied.emit("User denied camera permission")
                logger.warning("Camera permission denied by user")
                return False
                
        except Exception as e:
            logger.log_error_with_context(e, "camera_permission_request")
            return False
    
    def check_privacy_permission(self) -> bool:
        """Check if privacy permissions are properly configured."""
        try:
            # Check if privacy settings are enabled
            from ..utils.config import config
            
            # Check anonymization setting
            if not config.privacy.anonymize_data:
                logger.warning("Data anonymization is disabled")
                return False
            
            # Check data retention setting
            if config.privacy.data_retention_days <= 0:
                logger.warning("Data retention is disabled")
                return False
            
            self.privacy_permission_granted = True
            return True
            
        except Exception as e:
            logger.log_error_with_context(e, "privacy_permission_check")
            return False
    
    def request_privacy_permission(self, parent_widget=None) -> bool:
        """Request privacy permission from user."""
        try:
            # Show privacy policy dialog
            msg_box = QMessageBox(parent_widget)
            msg_box.setWindowTitle("Privacy Policy")
            msg_box.setText("This application collects anonymized data for attentiveness analysis.")
            msg_box.setInformativeText("""
Privacy Features:
• All data is anonymized and contains no personal information
• Data is automatically deleted after 7 days
• No facial images are stored permanently
• You can disable data collection at any time

Do you accept the privacy policy?
            """)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            
            # Set icon
            msg_box.setIcon(QMessageBox.Information)
            
            # Show dialog
            result = msg_box.exec()
            
            if result == QMessageBox.Yes:
                self.privacy_permission_granted = True
                self.permission_granted.emit()
                logger.info("Privacy permission granted")
                return True
            else:
                self.privacy_permission_granted = False
                self.permission_denied.emit("User denied privacy permission")
                logger.warning("Privacy permission denied by user")
                return False
                
        except Exception as e:
            logger.log_error_with_context(e, "privacy_permission_request")
            return False
    
    def check_system_permissions(self) -> Dict[str, bool]:
        """Check all system permissions."""
        try:
            permissions = {
                'camera': self.check_camera_permission(),
                'privacy': self.check_privacy_permission(),
                'storage': self._check_storage_permission(),
                'network': self._check_network_permission()
            }
            
            logger.info(f"System permissions: {permissions}")
            return permissions
            
        except Exception as e:
            logger.log_error_with_context(e, "system_permissions_check")
            return {
                'camera': False,
                'privacy': False,
                'storage': False,
                'network': False
            }
    
    def _check_storage_permission(self) -> bool:
        """Check if storage permission is available."""
        try:
            # Check if we can write to data directory
            data_dir = "data"
            test_file = os.path.join(data_dir, "test_permission.txt")
            
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Try to write test file
            with open(test_file, 'w') as f:
                f.write("test")
            
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
            
            return True
            
        except Exception as e:
            logger.log_error_with_context(e, "storage_permission_check")
            return False
    
    def _check_network_permission(self) -> bool:
        """Check if network permission is available (for future features)."""
        try:
            # For now, assume network is available
            # This could be expanded for cloud features
            return True
            
        except Exception as e:
            logger.log_error_with_context(e, "network_permission_check")
            return False
    
    def request_all_permissions(self, parent_widget=None) -> Dict[str, bool]:
        """Request all necessary permissions."""
        try:
            permissions = {}
            
            # Request camera permission
            permissions['camera'] = self.request_camera_permission(parent_widget)
            
            # Request privacy permission
            permissions['privacy'] = self.request_privacy_permission(parent_widget)
            
            # Check other permissions
            permissions['storage'] = self._check_storage_permission()
            permissions['network'] = self._check_network_permission()
            
            logger.info(f"All permissions requested: {permissions}")
            return permissions
            
        except Exception as e:
            logger.log_error_with_context(e, "all_permissions_request")
            return {
                'camera': False,
                'privacy': False,
                'storage': False,
                'network': False
            }
    
    def get_permission_status(self) -> Dict[str, Any]:
        """Get current permission status."""
        try:
            return {
                'camera_granted': self.camera_permission_granted,
                'privacy_granted': self.privacy_permission_granted,
                'system_permissions': self.check_system_permissions(),
                'platform': platform.system(),
                'python_version': sys.version
            }
            
        except Exception as e:
            logger.log_error_with_context(e, "permission_status")
            return {
                'camera_granted': False,
                'privacy_granted': False,
                'system_permissions': {},
                'platform': 'Unknown',
                'python_version': 'Unknown'
            }
    
    def show_permission_help(self, parent_widget=None):
        """Show help dialog for permission issues."""
        try:
            msg_box = QMessageBox(parent_widget)
            msg_box.setWindowTitle("Permission Help")
            msg_box.setText("Having trouble with permissions?")
            msg_box.setInformativeText("""
Camera Permission Issues:
• Ensure your camera is not being used by another application
• Check your system's camera privacy settings
• Try restarting the application

Privacy Settings:
• Data anonymization can be disabled in settings
• All data is stored locally by default
• No data is transmitted without explicit consent

For more help, check the application documentation.
            """)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.setIcon(QMessageBox.Information)
            
            msg_box.exec()
            
        except Exception as e:
            logger.log_error_with_context(e, "permission_help_display")
    
    def reset_permissions(self):
        """Reset all permission states."""
        try:
            self.camera_permission_granted = False
            self.privacy_permission_granted = False
            logger.info("Permissions reset")
            
        except Exception as e:
            logger.log_error_with_context(e, "permission_reset")


def check_platform_permissions() -> Dict[str, bool]:
    """Check platform-specific permissions."""
    try:
        platform_name = platform.system().lower()
        
        if platform_name == "windows":
            return _check_windows_permissions()
        elif platform_name == "darwin":  # macOS
            return _check_macos_permissions()
        elif platform_name == "linux":
            return _check_linux_permissions()
        else:
            logger.warning(f"Unknown platform: {platform_name}")
            return {
                'camera': True,  # Assume available
                'privacy': True,
                'storage': True,
                'network': True
            }
            
    except Exception as e:
        logger.log_error_with_context(e, "platform_permissions_check")
        return {
            'camera': False,
            'privacy': False,
            'storage': False,
            'network': False
        }


def _check_windows_permissions() -> Dict[str, bool]:
    """Check Windows-specific permissions."""
    try:
        # Windows typically doesn't require explicit camera permissions
        # but we should check if camera is accessible
        import cv2
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        cap.release()
        
        return {
            'camera': camera_available,
            'privacy': True,  # Assume privacy settings are configurable
            'storage': True,  # Assume storage is available
            'network': True   # Assume network is available
        }
        
    except Exception as e:
        logger.log_error_with_context(e, "windows_permissions_check")
        return {
            'camera': False,
            'privacy': False,
            'storage': False,
            'network': False
        }


def _check_macos_permissions() -> Dict[str, bool]:
    """Check macOS-specific permissions."""
    try:
        # macOS requires explicit camera permissions
        # This is a simplified check - in practice, you'd need to check system settings
        import cv2
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        cap.release()
        
        return {
            'camera': camera_available,
            'privacy': True,
            'storage': True,
            'network': True
        }
        
    except Exception as e:
        logger.log_error_with_context(e, "macos_permissions_check")
        return {
            'camera': False,
            'privacy': False,
            'storage': False,
            'network': False
        }


def _check_linux_permissions() -> Dict[str, bool]:
    """Check Linux-specific permissions."""
    try:
        # Linux typically doesn't require explicit camera permissions
        # but we should check if camera is accessible
        import cv2
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        cap.release()
        
        return {
            'camera': camera_available,
            'privacy': True,
            'storage': True,
            'network': True
        }
        
    except Exception as e:
        logger.log_error_with_context(e, "linux_permissions_check")
        return {
            'camera': False,
            'privacy': False,
            'storage': False,
            'network': False
        } 