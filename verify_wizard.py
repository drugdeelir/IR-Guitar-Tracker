import sys
import os
from PyQt5.QtWidgets import QApplication
from main import ProjectionMappingApp

def main():
    app = QApplication(sys.argv)
    window = ProjectionMappingApp()

    # Go to Setup Wizard
    window.tabs.setCurrentIndex(0)

    # Initial step
    window.grab().save("wizard_step1.png")

    # Advance to Step 4 (Guitar Mask)
    # 1 -> 2 (Markers) -> 3 (BG Mask) -> 4 (Guitar Mask)
    window.next_setup_step() # to 2
    window.next_setup_step() # to 3
    window.next_setup_step() # to 4

    window.grab().save("wizard_step4.png")

    # Step 5 (Finish)
    window.next_setup_step()
    window.grab().save("wizard_step5.png")

    # Workspace tab
    window.tabs.setCurrentIndex(1)
    window.grab().save("workspace_tab.png")

    print("Screenshots saved.")

if __name__ == "__main__":
    main()
