import cv2
import threading
import os
import pandas as pd
from deepface import DeepFace
from datetime import datetime
import openai
from secrets import api_key



openai.api_key = api_key
students = pd.read_csv("attendence.csv")
current_date = datetime.now().strftime("%Y-%m-%d")

class Recognition:
    
    def __init__(self):
        self.db = "./database"
        self.ref = self.load_references()
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.count = 0
        self.match = False
        self.match_name = ""
   

    def load_references(self):
        ref = []
        for file in os.listdir(self.db):
            if file.endswith(('.png', '.jpg', '.jpeg', 'webp')):
                img_path = os.path.join(self.db, file)
                person_name = os.path.splitext(file)[0]
                ref.append((cv2.imread(img_path), person_name))
            if not ref:
                raise Exception("Database is empty")
            
        return ref
    def cam_runner(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                raise Exception("Could not open video device")

            if self.count % 30 == 0:
                try:
                    threading.Thread(target=self.face_recognition, args=(frame,)).start()
                except ValueError:
                    raise Exception("Face not matched!")

            if self.match:
                add_attendence(self.match_name)

            self.count += 1
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cam.release()
        cv2.destroyAllWindows()

    def face_recognition(self, frame):
        try:
            for ref, name in self.ref:
                if DeepFace.verify(frame, ref)["verified"]:
                    self.match = True
                    self.match_name = name
                    return
                else:
                    self.match = False
                    self.match_name = ""

        except Exception as e:
            print(e)
            self.match = False
            self.match_name = ""

def add_attendence(student):
    students.loc[students["Name"] == student, current_date] = "Present"

def save_data():
    students.to_csv("attendence.csv", index=False)

def get_response(prompt):
    data = students.to_string(index=False)
    prompt = f"Here is the attendance data:\n{data}\n\n{prompt}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    choice = input("Enter '1' to recognize faces or '2' to get information from the CSV file: ")

    if choice == '1':
        recog = Recognition()
        recog.cam_runner()
        save_data()
    elif choice == '2':
        prompt = input("Enter your question: ")
        print(get_response(prompt))
    else:
        print("Invalid choice. Please enter '1' or '2'.")