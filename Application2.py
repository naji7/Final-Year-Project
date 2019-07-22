import cv2
import math
import tensorflow as tf
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







cascPath ='haarcascade_filters/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()                #capturing frames in in videostream
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #converting the frame to grayscale

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


    else:
        print("No face Found")
    #####################################selecting primary target face end



    #####################################writing on frame start

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 25)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    text=tensorEmotions[0]+" : "+str(math.ceil(tensorValues[0]*100)/100)

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    #####################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    text = tensorEmotions[1] + " : " + str(math.ceil(tensorValues[1]*100)/100)

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    #####################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 75)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    text = tensorEmotions[2] + " : " + str(math.ceil(tensorValues[2]*100)/100)

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    #####################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 100)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    text = tensorEmotions[3] + " : " + str(math.ceil(tensorValues[3]*100)/100)

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    #####################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 125)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    text = tensorEmotions[4] + " : " + str(math.ceil(tensorValues[4]*100)/100)

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    #####################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 150)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    text = tensorEmotions[5] + " : " + str(math.ceil(tensorValues[5]*100)/100)

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    #####################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 175)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    text = tensorEmotions[6] + " : " + str(math.ceil(tensorValues[6]*100)/100)

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    #####################################writing on frame end

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)    #drawing rectangle aronf the face
    cv2.imshow('Video', frame)                                  #Display updated frame


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing the capture after all done
video_capture.release()
cv2.destroyAllWindows()