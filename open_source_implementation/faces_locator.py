import dlib

from skimage import io

file_name = "../test/IMG_1138.jpeg"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Load the image into an array
image = io.imread(file_name)

# Run the HOG face detector on the image data.

# The result will be the bounding boxes of the faces in our image.

detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))
# Open a window on the desktop showing the image
# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                             face_rect.right(), face_rect.bottom()))
    io.imshow(image[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()])
    io.show()
    # Draw a box around each face we found

# Wait until the user hits <enter> to close the window

dlib.hit_enter_to_continue()