import cv2
from matplotlib import pyplot as plt
import glob

source_path = "Object_and_Hand/"
from pathlib import Path
image_files = []
for path in Path(source_path).rglob('*.jpg'):
    image_files.append(str(path.name))

print("image_files[0]: ", image_files[0])

for curr_img in image_files:
    print("curr_img: ",source_path+curr_img)
    # Opening image
    img = cv2.imread(source_path+curr_img)

    # OpenCV opens images as BRG
    # but we want it as RGB We'll
    # also need a grayscale version
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use minSize because for not
    # bothering with extra-small
    # dots that would look like STOP signs
    curr_xml = curr_img[:-4]
    print("curr_xml: ", source_path + curr_xml + '.xml')

    #img_data = cv2.CascadeClassifier(cv2.data.haarcascades+source_path+curr_xml+'.xml')
    img_data = cv2.CascadeClassifier(source_path + curr_xml + '.xml')

    found = img_data.detectMultiScale(img_gray, minSize=(20, 20))

    # Don't do anything if there's
    # no sign
    amount_found = len(found)

    if amount_found != 0:

        # There may be more than one
        # sign in the image
        for (x, y, width, height) in found:
            # We draw a green rectangle around
            # every recognized sign
            cv2.rectangle(img_rgb, (x, y),
                          (x + height, y + width),
                          (0, 255, 0), 5)

        # Creates the environment of
    # the picture and shows it
    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()
