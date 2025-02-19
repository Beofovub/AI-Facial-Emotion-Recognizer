import cv2
from fer import FER

def recognize_emotions(image_path):
    """Detects emotions in the given image using the FER library."""
    img = cv2.imread(image_path)
    detector = FER(mtcnn=True)
    results = detector.detect_emotions(img)
    return results

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    emotions = recognize_emotions(image_path)
    print("Detected Emotions:", emotions)
