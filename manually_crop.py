import re
import screeninfo
import argparse
import cv2

# Initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    # Grab references to the global variables
    global refPt, cropping

    # If the left mouse button was clicked, record the starting (x, y)
    # coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # Draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow(window_name, image)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-d", "--output_dir", required=True, help="Path to the output directory")
ap.add_argument("-r", "--region_file", required=True, help="Path to a file to write region coordinates")
args = vars(ap.parse_args())
file_path = args["image"]
output_dir = args["output_dir"]
region_file = args["region_file"]
if output_dir[-1] != '/':
    output_dir += '/'

# Change filename to write as png and without "frame"
filename = re.sub(".*frame(.*)\.jpg", "\\1", file_path) + ".png"

# Set up fullscreen
window_name = "Image to Crop"
screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load the image, clone it, and set up the mouse callback function
image = cv2.imread(file_path)
clone = image.copy()
cv2.setMouseCallback(window_name, click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow(window_name, image)
    key = cv2.waitKey(1) & 0xFF

    # If the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # If the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# If there are two reference points, then crop the region of interest
# from the image, save the cropped image, and write the region coordinates.
if len(refPt) == 2:
    crop_width = 350
    crop_height = 350
    x1 = int(refPt[0][0])
    y1 = int(refPt[0][1])
    x2 = int(refPt[1][0])
    y2 = int(refPt[1][1])
    width, height = x2 - x1, y2 - y1
    w_diff, h_diff = crop_width - width, crop_height - height
    left_pad, top_pad = int(w_diff / 2), int(h_diff / 2)
    right_pad, bottom_pad = w_diff - left_pad, h_diff - top_pad
    x1 = x1 - left_pad
    x2 = x2 + right_pad
    y1 = y1 - top_pad
    y2 = y2 + bottom_pad
    roi = clone[y1:y2, x1:x2]
    cv2.imwrite(output_dir + filename, roi)
    with open(region_file, 'a') as f:
        f.write(f"{filename},{x1},{y1},{x2},{y2}\n")
    cv2.waitKey(0)

# Close all open windows
cv2.destroyAllWindows()
