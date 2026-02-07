
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer, QPoint, QPointF
from main import ProjectionMappingApp

def verify():
    app = QApplication(sys.argv)

    # Set offscreen for headless verification
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    main_win = ProjectionMappingApp()
    main_win.show()

    # 1. Verify Wizard Step 1
    main_win.grab().save("wizard_step1_bounds.png")
    print("Saved wizard_step1_bounds.png")

    # 2. Switch to Boundary Tab
    # Tabs: Setup Wizard (0), Stage (1), Media & Cues (2), Calibration (3), Boundary (4), System (5), Connectivity (6)
    main_win.tabs.setCurrentIndex(4)
    main_win.grab().save("boundary_tab.png")
    print("Saved boundary_tab.png")

    # 3. Test Manual Edit mode
    main_win.edit_bounds_btn.setChecked(True)
    # Simulate some points
    main_win.video_display.set_mask_points([QPoint(100, 100), QPoint(500, 100), QPoint(500, 400), QPoint(100, 400)])
    main_win.grab().save("boundary_edit_mode.png")
    print("Saved boundary_edit_mode.png")

    sys.exit(0)

if __name__ == "__main__":
    verify()
