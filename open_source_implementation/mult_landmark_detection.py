import face_recognition
from PIL import Image, ImageDraw

image = face_recognition.load_image_file("test_images/profile.jpg")
face_locations = face_recognition.face_locations(image)
print("There are {} face(s) in this photo.".format(len(face_locations)))
face_images = []

for face in face_locations:
    top, right, bottom, left = face

    face_image = image[top:bottom, left:right]

    face_landmarks_list = face_recognition.face_landmarks(face_image)

    feature_list = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]
    pil_image = Image.fromarray(face_image)

    for mark in face_landmarks_list:
        for feature in feature_list:
            img = ImageDraw.Draw(pil_image)
            img.line(mark[feature], width=1)
    face_images.append(face_image)

# for face_picture in faces:
    # face_picture.show()

face_encs = []
i = 0
for face_picture in face_images:
    face_encs.append(face_recognition.face_encodings(face_picture)[++i])

unknown_enc = face_recognition.face_encodings(face_recognition.load_image_file("test_images/test.jpg"))[0]

print(len(face_encs))

results = face_recognition.compare_faces(face_encs, unknown_enc)

for result in results:
    print(result)
