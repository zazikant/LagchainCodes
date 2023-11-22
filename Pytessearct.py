# python code
!pip install pytesseract
!apt install tesseract-ocr -y
!apt install libtesseract-dev

# python code
import cv2
import pytesseract

# Define the path
image_path = 'D:\colab_pro\AUTOGEN\groupchat\IvV2y.png'

# Read the image
img = cv2.imread(image_path)

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Tesseract to extract text
text = pytesseract.image_to_string(gray)

# Print the extracted text
print(text)