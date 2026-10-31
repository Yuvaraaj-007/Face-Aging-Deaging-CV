import cv2
import dlib
import math
import requests
import zipfile
import io
import os
from tkinter import Tk,filedialog
#print(os.path.exists("Models/haarcascade_frontalface_default.xml"))
#img = cv2.imread("E:/Myself/Computer Vision/Face_Aging/Images/test.jpg")
root=Tk()
root.withdraw()
image_path=filedialog.askopenfilename(title="Select a Frontal Face Image",filetypes=[("Image Files","*.jpg *.jpeg *.png")])
root.destroy()
if not image_path:
    print("No image selected!")
    exit()
img=cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Could not load input image. Check test.jpg path.")
face = cv2.CascadeClassifier(r"E:/Myself/Computer Vision/Face_Aging/Models/haarcascade_frontalface_default.xml")
#print(face.empty())
h,w = img.shape[0:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clean_img = img.copy()
faces = face.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=9)
if len(faces) == 0:
    raise ValueError("No face detected in the input image.")
predictor = dlib.shape_predictor(r"Models/shape_predictor_68_face_landmarks.dat")
print("Landmark model successfully loaded")
for x,y,w,h in faces:
    rectangle = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    rect = dlib.rectangle(x,y,x+w,y+h)
    landmarks = predictor(gray,rect)
for i, (x, y, w, h) in enumerate(faces, start=1):
    print("Face", i, ":", x, y, w, h)
    crop = clean_img[y:h+y,x:w+x]
print(landmarks.num_parts)
print(landmarks.part(0).x)
print(landmarks.part(0).y)
for j in range(landmarks.num_parts):
    x1 = landmarks.part(j).x
    y1 = landmarks.part(j).y
    cv2.circle(img, (x1,y1), 2, (0,255,0),-1)
c = 0
d = 0
for k in range(36,42):
    a = landmarks.part(k).x
    b = landmarks.part(k).y
    c+=a
    d+=b
c_x = c/6
c_y = d/6
e = 0
f = 0
for l in range(42,48):
    a1 = landmarks.part(l).x
    b1 = landmarks.part(l).y
    e+=a1
    f+=b1
c_x1 = e/6
c_y1 = f/6
d_x = c_x1-c_x
d_y = c_y1-c_y
an = math.atan2(d_y,d_x)
angle = math.degrees(an)
m = cv2.getRotationMatrix2D(((c_x+c_x1)/2,(c_y+c_y1)/2), angle, 1)
h1,w1 = img.shape[0:2]
img2 = cv2.warpAffine(clean_img,m,(w1,h1))
print("Total faces:", len(faces))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
clean_img2 = img2.copy()
face2 = cv2.CascadeClassifier(r"E:/Myself/Computer Vision/Face_Aging/Models/haarcascade_frontalface_default.xml")
#print(face.empty())
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces1 = face2.detectMultiScale(gray2, scaleFactor = 1.1, minNeighbors=9)
print("Detected faces after alignment:", len(faces1))
predictor = dlib.shape_predictor(r"Models/shape_predictor_68_face_landmarks.dat")
print("Landmark model successfully loaded")
for x3,y3,w3,h3 in faces1:
    rectangle = cv2.rectangle(img2, (x3,y3), (x3+w3, y3+h3), (0,255,0), 2)
    rect = dlib.rectangle(x3,y3,x3+w3,y3+h3)
    landmarks = predictor(gray2,rect)
for i1, (x3, y3, w3, h3) in enumerate(faces1, start=1):
    print("Face", i1, ":", x3, y3, w3, h3)
    crop1 = clean_img2[y3:h3+y3,x3:x3+w3]
    print(crop1.shape)
#print(x3, y3, w3, h3)
#print(img2.shape)
#print(landmarks.num_parts)
#print(landmarks.part(0).x)
#print(landmarks.part(0).y)
for j1 in range(landmarks.num_parts):
    x4 = landmarks.part(j1).x
    y4 = landmarks.part(j1).y
    cv2.circle(img2, (x4,y4), 2, (0,255,0),-1)
c1 = 0
d1 = 0
for k1 in range(36,42):
    a2 = landmarks.part(k1).x
    b2 = landmarks.part(k1).y
    c1+=a2
    d1+=b2
c_x2 = c1/6
c_y2 = d1/6
e1 = 0
f1 = 0
for l1 in range(42,48):
    a3 = landmarks.part(l1).x
    b3 = landmarks.part(l1).y
    e1+=a3
    f1+=b3
c_x3 = e1/6
c_y3 = f1/6
d_x1 = c_x3-c_x2
d_y1 = c_y3-c_y2
corrected_angle = math.degrees(math.atan2(d_y1, d_x1))
print("Angle = ", corrected_angle)
#cv2.imshow("Original Image",img)
cv2.imshow("Face detection", img2)
cv2.imshow("Cropped image", crop1)
cv2.imwrite("E:/Myself/Computer Vision/Face_Aging/Output/face.jpg",crop1)
#cv2.imshow("Grayscale Image",gray)
# ================= SAM API INTEGRATION =================

API_URL = "https://boney-valiant-remake.ngrok-free.dev/generate"

# Enter your newly regenerated Colab API key
API_KEY = "***REMOVED***"

face_path = r"E:/Myself/Computer Vision/Face_Aging/Output/face.jpg"

output_folder = r"E:/Myself/Computer Vision/Face_Aging/Output/Aged_Images"

os.makedirs(output_folder, exist_ok=True)

print("Sending aligned face to SAM API...")

try:
    with open(face_path, "rb") as image_file:
        files = {
            "image": ("face.jpg", image_file, "image/jpeg")
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}"
        }
        response = requests.post(
            API_URL,
            files=files,
            headers=headers,
            timeout=600
        )
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_data:
            zip_data.extractall(output_folder)
        print("All 11 aged images saved successfully!")
        print("Images saved at:", output_folder)
    else:
        print("API Error:", response.status_code)
        print("Server response:", response.text)
except requests.exceptions.RequestException as e:
    print("API connection error:", e)
except zipfile.BadZipFile:
    print("The server response was not a valid ZIP file.")

# =====================================================
cv2.waitKey(0)
cv2.destroyAllWindows()