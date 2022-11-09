import streamlit as st
import cv2
from PIL import Image
from datetime import datetime,date
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_default.xml')

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingData.yml")
def detect_faces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    # Draw rectangle around the faces
    name='Unknown'
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.rectangle(img, (x-50, y-50), (x + w+50, y + h+50), (225, 0, 0), 2)
        id, uncertainty = rec.predict(gray[y:y + h, x:x + w])
        print(id, uncertainty)

        if (uncertainty< 200):
            if (id == 1):
                name = "Rajesh"
                markAttendance(name)
                cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 255, 255),2)
            if(id==2):
                name = "Dhoni"
                markAttendance(name)
                cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 255, 255),2)
            if(id==3):
                name = "Kohli"
                markAttendance(name)
                cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 255, 255),2)
        else:
            cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 255, 255),2)


    return img

def markAttendance(name):
  with open('Attendance.csv','r+') as f:
    myDataList = f.readlines()
    nameList = []
    for line in myDataList:
      entry = line.split(',')
      nameList.append(entry[0])
    if name not in nameList:
      now = datetime.now()
      today = date.today()
# print("Today's date:", today)
      dtString = now.strftime('%H:%M:%S')
      # date = now.today('%d%b%Y%H%M%S')
      f.writelines(f'\n{name},{dtString},{today}')

def main():
    """Face Recognition App"""

    st.title("Streamlit Tutorial")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        result_img= detect_faces(our_image)
        st.image(result_img)


if __name__ == '__main__':
    main()