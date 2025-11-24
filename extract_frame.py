
import cv2

def extract_frame():
    """Extracts the first frame from logo.mkv and saves it as logo.png."""
    cap = cv2.VideoCapture('logo.mkv')
    if not cap.isOpened():
        print("Error: Could not open logo.mkv")
        return

    ret, frame = cap.read()
    if ret:
        cv2.imwrite('logo.png', frame)
        print("Successfully extracted frame from logo.mkv and saved as logo.png")
    else:
        print("Error: Could not read frame from logo.mkv")

    cap.release()

if __name__ == '__main__':
    extract_frame()
