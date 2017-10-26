from PIL import Image
import glob
import face_recognition

known_persons = []
test_encoding = face_recognition.face_encodings(face_recognition.load_image_file("test_images/test.jpg"))[0]
i = 0
for filename in glob.glob('test_images/kunal/*.jpg'): #assuming gif
    im = face_recognition.face_encodings(face_recognition.load_image_file(filename))[0]
    known_persons.append(im)
    print(i)

results = face_recognition.compare_faces(known_persons, test_encoding)

for result in results:
    print(result)

