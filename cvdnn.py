import numpy as np
import argparse as ap
import time
import cv2

arp = ap.ArgumentParser()
arp.add_argument("-i", "--image", required=False,
	help="path to input image")
arp.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
arp.add_argument("-m", "--model", required=True,
	help="path to Caffe pretrained model")
arp.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(arp.parse_args())

cap = cv2.VideoCapture()
if (args["image"] is None) :
   cap.open(0)
else :
   cap.open(args["image"])

if (cap.isOpened() == False) :
   print("ERROR: Fail to capture")
   exit(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
   
while True:
   ret, image = cap.read()
   if (image is None) : break
   # cv2.imshow("Image", image)
   # cv2.waitKey(1)
   
   blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
   
   net.setInput(blob)
   start = time.time()
   preds = net.forward()
   end = time.time()

   print("[INFO] classification took {:.5} seconds".format(end - start))

   idxs = np.argsort(preds[0])[::-1][:3]
   for (i, idx) in enumerate(idxs):
	   if i == 0:
		   text = "Label: {}, {:.2f}".format(classes[idx], preds[0][idx])
		   cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			   0.7, (0, 0, 255), 2)
	   print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		   classes[idx], preds[0][idx]))

   cv2.imshow("Image", image)
   cv2.waitKey(1)
   
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
