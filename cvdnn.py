import numpy as np
import argparse as ap
import time
import cv2

arp = ap.ArgumentParser()
arp.add_argument("-i", "--image", required=True,
	help="path to input image")
arp.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
arp.add_argument("-m", "--modle", required=True,
	help="path to Caffe pretrained model")
arp.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(arp.parse_args())

image = cv2.imread(args["image"])

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()

print("[INFO] classification took {:.5} seconds".format(end - start))

idxs = np.atgsort(preds[0])[::-1][:5]
for (i, idx) in enumerate(idxs):
	if i == 0:
		text = "Label: {}, {:.2f}%".format(i + 1, classes[idx], preds[0][idx])
cv2.imshow("Image", image)
cv2.waitKey(0)
