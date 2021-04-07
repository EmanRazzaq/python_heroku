import face_recognition
import os


from selenium import webdriver
op = webdriver.ChromeOption()
op.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
op.add_argument("--headless")
op.add_argument("--no-sandbox")
op.add_argument("--disable-dev-sh-usage")

driver = webdriver.Chrome(executable_path = os.environ.get("CHROMMEDRIVER_PATH"), chrome_option=op)

driver.get("https://youtube.com")
print(driver.page_source)


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




