import face_recognition
from PIL import Image, ImageDraw
import numpy

image = face_recognition.load_image_file("test_images/profile.jpg")
face_locations = face_recognition.face_locations(image)
print("I found {} face(s) in this photograph.".format(len(face_locations)))
for face in face_locations:
    print()
    cimage = Image.fromarray(image)
    print("width:", cimage.size[0])
    print("height:", cimage.size[1])
    width = cimage.size[0]
    height = cimage.size[1]
    # cimage = cimage.crop()
    # print("width:", cimage.size[0])
    # print("height:", cimage.size[1])
    #cimage.show()

    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        facial_features = [
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

        for feature in facial_features:
            print("The {} in this face has the following points: {}".format(feature, face_landmarks[feature]))

            ed_image = Image.fromarray(image)
            x = ImageDraw.Draw(ed_image)

            for feature in facial_features:
                x.line(face_landmarks[feature], width=2)

ed_image.show()
