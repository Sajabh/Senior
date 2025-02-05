import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import os

def draw_facebox(filename, result_list):
  # load the image
  data = plt.imread(filename)
  # plot the image
  plt.imshow(data)
  #ل
  # get the context for drawing boxes
  ax = plt.gca()
  # plot each box
  for result in result_list:
    # get coordinates
    x, y, width, height = result['box']
    # create the shape
    rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
    # draw the box
    ax.add_patch(rect)
    # show the plot
  plt.title(f'Detected faces in {os.path.basename(filename)}')
  plt.show()

# Create MTCNN detector
detector = MTCNN()

# Define the folder path and supported image extensions
folder_path = "familyMem"  # Current directory
supported_extensions = ('.jpg', '.jpeg', '.png')

# Process all images in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(supported_extensions):
        image_path = os.path.join(folder_path, filename)
        try:
            # Load and process image
            pixels = plt.imread(image_path)
            faces = detector.detect_faces(pixels)
            # Display faces on the original image
            draw_facebox(image_path, faces)
            print(f"Processed {filename}: Found {len(faces)} faces")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")