import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from main import ProjectionMappingApp

def verify():
    app = QApplication(sys.argv)
    main_win = ProjectionMappingApp()
    main_win.show()

    # Advanced Wizard Step 1
    main_win.tabs.setCurrentIndex(0)

    def capture():
        main_win.grab().save("wizard_step1_pro.png")
        print("Professional Wizard Step 1 captured.")

        # Test fade state (internally)
        if hasattr(main_win.worker, 'mask_fade_levels'):
            print("Fade levels dict exists.")

        if hasattr(main_win.worker, 'rbf_x'):
            print("RBF mapping support exists.")

        app.quit()

    QTimer.singleShot(1000, capture)
    sys.exit(app.exec_())

if __name__ == "__main__":
    verify()
