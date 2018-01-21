# train.py
import cv2, sys, numpy, os

size = 4
fn_haar = 'Haarcascade/haarcascade_frontalface_default.xml'
fn_dir1 = 'done'
fn_dir = 'pics'
pin = 0
lest_walue = numpy.ndarray

try:
    fn_name = sys.argv[1]
except:
    print("You must provide a name")
    sys.exit(0)
path = os.path.join(fn_dir1, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(fn_haar)

# Part 1: Create fisherRecognizer
print('Training...')

(images, lables, names, id) = ([], [], {}, 0)

# Get the folders containing the training data
for (subdirs, dirs, files) in os.walk(fn_dir):

    # Loop through each folder named after the subject in the photos
    for subdir in dirs:
        names[id] = subdir

        subjectpath = os.path.join(fn_dir, subdir)

        # Loop through each photo in the folder
        for filename in os.listdir(subjectpath):

            # Skip non-image formates
            f_name, f_extension = os.path.splitext(filename)
            if (f_extension.lower() not in
                    ['.png', '.jpg', '.jpeg', '.gif', '.pgm']):
                print("Skipping " + filename + ", wrong file type")
                continue
            path = subjectpath + '/' + filename
            lable = id

            frame = cv2.imread(path)

            # Add to training data
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
            faces = haar_cascade.detectMultiScale(mini)

            if (numpy.array_equal(faces, lest_walue)):
                pin += 1
            else:

                lest_walue = faces
                faces = sorted(faces, key=lambda x: x[3])
                if faces:
                    face_i = faces[0]
                    (x, y, w, h) = [v * size for v in face_i]

                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (im_width, im_height))
                    print(pin)
                    cv2.imwrite('done/' + subdir + '/' + str(pin) + '.png', face_resize)
                    pin += 1
