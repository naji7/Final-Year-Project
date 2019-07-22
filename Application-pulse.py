import cv2
import tensorflow as tf
from numpy.distutils.system_info import boost_python_info

import FisherFace.FisherDetect as fish
import TensorFlow.TensorDetect as tensor

#####################################d###################intializing tensorflow variables start

###############################################################################file paths start
RETRAINED_LABELS_TXT_FILE_LOC = "C:/Users/Nimzan/PycharmProjects/FinalApplication/TensorFlow/retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = "C:/Users/Nimzan/PycharmProjects/FinalApplication/TensorFlow/retrained_graph.pb"
###############################################################################file paths end

retrainedGraphFile= tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb')      # Loading the trained graph
graphDef = tf.GraphDef()                                                       # creating graphdef object
graphDef.ParseFromString(retrainedGraphFile.read())                            # parsing the graph
tf.import_graph_def(graphDef, name='')                                         # import the graph into the current default Graph

sess=tf.Session()
finalTensor = sess.graph.get_tensor_by_name('final_result:0')

#####################################d###################intializing tensorflow variables end



mouth_cascade = cv2.CascadeClassifier('TensorflowMouth/haarcascade_mcs_mouth.xml')



cascPath ='haarcascade_filters/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)


import serial
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

arduino = serial.Serial('COM3', 115200, timeout=10)

filename = "C:/Users/Nimzan/PycharmProjects/FinalApplication/Pulse/model.sav"
loaded_model = joblib.load(filename)




while True:


    #################################################################################################################### pulse start
    data = arduino.readline()[:-2]  # the last bit gets rid of the new-line chars
    if data:
        # print(data)# create data
        bpm = int(data)
        if (39 > bpm or 200 < bpm):
            print("BPM Exceeds the limit")
            continue
        else:
            # print(bpm)
            Pulseemotions = loaded_model.predict([[bpm]])
            # print(emotion)

            print("BPM : " + str(bpm) + "    Expected Emotions : " + Pulseemotions[0])

    #################################################################################################################### pulse end

    ret, frame = video_capture.read()                #capturing frames in in videostream
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #converting the frame to grayscale

    #####################################mouth start

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15 * h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        break

    #####################################mouth end


    #####################################detecting faces start
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    #####################################detecting faces end

    #####################################selecting primary target face start
    if len(faces)>0:
        print("Primary Face Found")
        primaryFaceCoordinate = faces[0]
        print(primaryFaceCoordinate)

        [x, y, w, h]=primaryFaceCoordinate

        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (350, 350)) #cropping the face 350*350

        #####################################emotion start
        fisherEmotion = fish.findEmotion(face)#fisherface

        cv2.imwrite("tensorFace.jpg",face)
        image = cv2.imread('C:/Users/Nimzan/PycharmProjects/FinalApplication/tensorFace.jpg')
        tensorEmotions, tensorValues = tensor.getEmotions(image, sess, finalTensor)
        tensorEmotion = tensorEmotions[0]
        #####################################emotion end

        #####################################comparison start
        if fisherEmotion=="happy":
            finalEmotion = fisherEmotion
        elif fisherEmotion=="surprise":
            finalEmotion=fisherEmotion
        elif fisherEmotion==tensorEmotions[0]:
            finalEmotion = fisherEmotion
        elif fisherEmotion==tensorEmotions[1]:
            finalEmotion = fisherEmotion
        elif fisherEmotion==tensorEmotions[2]:
            finalEmotion = fisherEmotion
        else:
            finalEmotion=tensorEmotions[0]
        print(finalEmotion)

        #####################################comparison end


    else:
        print("No face Found")
    #####################################selecting primary target face end

    if finalEmotion!=Pulseemotions[0]:
        print("Emotion is not relavant to pulse")
        print(" ")

    else:
        print("Emotion seems legit")
        break


    #####################################writing on frame start
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 25)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2

    cv2.putText(frame, finalEmotion,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (500, 25)q
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2

    cv2.putText(frame, fisherEmotion,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    '''
    #####################################writing on frame end

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)    #drawing rectangle aronf the face
    cv2.imshow('Video', frame)                                  #Display updated frame


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# releasing the capture after all done
video_capture.release()
cv2.destroyAllWindows()