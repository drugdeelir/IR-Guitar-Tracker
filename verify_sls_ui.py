import sys
from PyQt5.QtWidgets import QApplication
from main import ProjectionMappingApp

def main():
    app = QApplication(sys.argv)
    window = ProjectionMappingApp()

    # Go to Setup Wizard
    window.tabs.setCurrentIndex(0)
    window.grab().save("wizard_step1_sls.png")

    # Check Background Mask color
    window.start_setup_bg_mask()
    window.grab().save("wizard_step2_bg_color.png")

    # Check Guitar Mask color
    window.setup_step = 2 # Jump to guitar step
    window.next_setup_step()
    window.start_setup_amp_mask()
    window.grab().save("wizard_step4_guitar_color.png")

    print("Screenshots saved.")

if __name__ == "__main__":
    main()
