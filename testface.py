import face_recognition
import os





known_image = face_recognition.load_image_file(r"C:\Users\htc\Desktop\face-recognition-opencv\face-recognition-opencv\dataset\Sarah_Khan\5.jpeg")
#known_image = face_recognition.load_image_file(r"C:\Users\htc\PycharmProjects\first\Found\3.png")



#directory = (r"C:\Users\htc\PycharmProjects\first\Found\\")
directory = (r"C:\Users\htc\Desktop\face-recognition-opencv\face-recognition-opencv\dataset\Sarah_Khan\\")
for x in os.listdir(directory):
     print(x)
     unknown_image = face_recognition.load_image_file(directory + x)  
     unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
     biden_encoding = face_recognition.face_encodings(known_image)[0]
     results = face_recognition.compare_faces([biden_encoding], unknown_encoding, tolerance=0.3)
     print(results)




