import numpy as np
import cv2 
import imutils
from imutils import paths

bShowColor = False

TEST_PATH = "testImgs"

FRAME_WIDTH = 0
FRAME_HEIGHT = 0

vColor = (0, 255, 0)#vehicle bounding-rect and information color
pColor = (0, 0, 255)#plate bounding-rect and information color
rectThinkness = 2

items = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
         "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
         "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
         "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
         "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
         "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
         "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
         "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
         "<Zhejiang>", "<police>",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z"]
         
seq_ind = np.ones([88, 1], dtype=np.float32)
seq_ind[0, 0] = 0

#https://docs.openvinotoolkit.org/2019_R1/_license_plate_recognition_barrier_0001_description_license_plate_recognition_barrier_0001.html
lpr_net = cv2.dnn.Net_readFromModelOptimizer('./license-plate-recognition-barrier-0001/license-plate-recognition-barrier-0001.xml', './license-plate-recognition-barrier-0001/license-plate-recognition-barrier-0001.bin')

lpr_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
lpr_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]
#https://docs.openvinotoolkit.org/2019_R1/_vehicle_attributes_recognition_barrier_0039_description_vehicle_attributes_recognition_barrier_0039.html
attr_net = cv2.dnn.Net_readFromModelOptimizer('./vehicle-attributes-recognition-barrier-0039/vehicle-attributes-recognition-barrier-0039.xml', './vehicle-attributes-recognition-barrier-0039/vehicle-attributes-recognition-barrier-0039.bin')

attr_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
attr_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html
pd_net = cv2.dnn.readNet('./vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.xml', './vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.bin')

pd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
pd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def drawText(frame,scale,rectX,rectY,rectColor,text):

    textSize, _ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale, 3)

    top = max(rectY - rectThinkness, textSize[0])
    
    cv2.putText(frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3)

    
def plateRecognition(frame):
    
    #name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
    pd_blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    pd_net.setInput(pd_blob)
    out_pb = pd_net.forward()

    # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
    for detection in out_pb.reshape(-1, 7):
        conf = detection[2]
        if conf < 0.4:
            continue

        classId = int(detection[1])
        if classId == 1:#car
            xmin = int(detection[3] * FRAME_WIDTH)
            ymin = int(detection[4] * FRAME_HEIGHT)
            xmax = int(detection[5] * FRAME_WIDTH)
            ymax = int(detection[6] * FRAME_HEIGHT)
            
            rectW = xmax - xmin
            if rectW < 72:  # Minimal weight in vehicle-attributes-recognition-barrier-0039  is 72
                continue
            
            attrImg = frame[ymin:ymax+1, xmin:xmax+1]
            
            attr_blob = cv2.dnn.blobFromImage(attrImg, size=(72, 72), ddepth=cv2.CV_8U)
            attr_net.setInput(attr_blob,'input')
            
            out_color = attr_net.forward("color")
            out_type = attr_net.forward("type")

            carColor = "Color: " + CAR_COLORS[np.argmax(out_color.flatten())]      
            carType = "Type: " + CAR_TYPES[np.argmax(out_type.flatten())]
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), vColor,rectThinkness)
            
            if bShowColor:
                drawText(frame,rectW*0.002,xmin,ymin,vColor,carColor +" " + carType)
            
        
        elif classId == 2:#plate
            xmin = int(detection[3] * FRAME_WIDTH)
            ymin = int(detection[4] * FRAME_HEIGHT)
            xmax = int(detection[5] * FRAME_WIDTH)
            ymax = int(detection[6] * FRAME_HEIGHT)

            xmin = max(0, xmin - 5)
            ymin = max(0, ymin - 5)
            xmax = min(xmax + 5, FRAME_WIDTH - 1)
            ymax = min(ymax + 5, FRAME_HEIGHT - 1)

            rectW = xmax - xmin
            if rectW < 93:  # Minimal weight in plate-recognition-barrier-0001 is 94
                continue

            # Crop a license plate. Do some offsets to better fit a plate.
            lpImg = frame[ymin:ymax+1, xmin:xmax+1]
            blob = cv2.dnn.blobFromImage(lpImg, size=(94, 24), ddepth=cv2.CV_8U)
            lpr_net.setInput(blob, 'data')
            lpr_net.setInput(seq_ind, 'seq_ind')
            out_lpr = lpr_net.forward()
          
            content = ''
            for idx in np.int0(out_lpr.reshape(-1)):
                if idx == -1:
                    break
                content += items[idx]
                
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), pColor,rectThinkness)
            drawText(frame,rectW*0.008,xmin,ymin,pColor,content)
            
    showImg = imutils.resize(frame, height=600)
    cv2.imshow("showImg",showImg)  


for imagePath in paths.list_images(TEST_PATH):
    print(imagePath)
    img = cv2.imread(imagePath)
    if img is None:
        continue
    FRAME_WIDTH = img.shape[1]
    FRAME_HEIGHT = img.shape[0]

    plateRecognition(img)
    #if bShowColor:
    cv2.waitKey(0)
    