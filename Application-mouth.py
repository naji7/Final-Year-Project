import cv2
import tensorflow as tf
import FisherFace.FisherDetect as fish
import TensorFlow.TensorDetect as tensor
import TensorflowMouth.TensorMouthDetect as tensormouth

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



#####################################d###################intializing tensorflowMouth variables start
###############################################################################file paths start
RETRAINED_LABELS_TXT_FILE_LOCmouth = "C:/Users/Niru/Desktop/FinalApplication/TensorflowMouth/retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOCmouth = "C:/Users/Niru/Desktop/FinalApplication/TensorflowMouth/retrained_graph.pb"
###############################################################################file paths end

retrainedGraphFilemouth= tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOCmouth, 'rb')      # Loading the trained graph
graphDefmouth = tf.GraphDef()                                                       # creating graphdef object
graphDefmouth.ParseFromString(retrainedGraphFilemouth.read())                            # parsing the graph
tf.import_graph_def(graphDefmouth, name='')                                         # import the graph into the current default Graph

sessmouth=tf.Session()
finalTensormouth = sessmouth.graph.get_tensor_by_name('final_result:0')
#####################################d###################intializing tensorflowMouth variables end

while True:

    ret, frame = video_capture.read()                #capturing frames in in videostream
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #converting the frame to grayscale

    #####################################mouth start

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15 * h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        mouth = gray[y:y + h, x:x + w]
        mouth2 = cv2.resize(mouth, (130, 70))
        cv2.imwrite('mouth.jpg', mouth2)
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

        #####################################comparison end

        if len(mouth_rects)>0:
            imagemouth = cv2.imread('C:/Users/Nimzan/PycharmProjects/FinalApplication/mouth.jpg')
            tensorEmotionsmouth, tensorValuesmouth = tensormouth.getEmotions(imagemouth, sessmouth, finalTensormouth)
            tensorEmotionmouth = tensorEmotionsmouth[0]
            print(tensorEmotionmouth)

            #####################################comparison mouth start
            if tensorEmotionmouth == "fear":
                finalemotion=tensorEmotionmouth
            #####################################comparison mouth end


    else:
        print("No face Found")
    #####################################selecting primary target face end



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
    bottomLeftCornerOfText = (500, 25)
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