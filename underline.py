import sys
sys.path.append("./Mask_RCNN")
import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN

# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "underline_cfg"

    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'mask_rcnn_underline_cfg_0019.h5'
model.load_weights(model_path, by_name=True)


temp = cv2.imread("./text_detection/test/androidFlask.jpg")

yhat = model.detect([temp], verbose=0)[0]
print(yhat['rois'])
# plot first few images
pyplot.imshow(temp)
# plot all masks
# for j in range(len(yhat['rois'])):
#   pyplot.imshow(list(yhat['rois'][j]), cmap='gray', alpha=0.3)
ax = pyplot.gca()
# plot each box
# print(len(yhat['rois']))
for box in yhat['rois']:
    # get coordinates
    y1, x1, y2, x2 = box
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    # draw the box
    ax.add_patch(rect)
# show the figure
pyplot.show()