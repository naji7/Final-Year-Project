import TensorFlow.TensorDetect as tensor
import tensorflow as tf
import cv2


'''
image=cv2.imread('C:/Users/Nimzan/PycharmProjects/FinalApplication/tensorFace.jpg')

while True:
    tensorEmotions, tensorValues = tensor.getEmotions(image)
    tensorEmotion = tensorEmotions[0]

    print(tensorEmotion)

'''

###############################################################################file paths start
RETRAINED_LABELS_TXT_FILE_LOC = "C:/Users/Nimzan/PycharmProjects/FinalApplication/TensorFlow/retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = "C:/Users/Nimzan/PycharmProjects/FinalApplication/TensorFlow/retrained_graph.pb"
###############################################################################file paths end

retrainedGraphFile= tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') # Loading the trained graph
graphDef = tf.GraphDef()                                                       # creating graphdef object
graphDef.ParseFromString(retrainedGraphFile.read())                            # parsing the graph
_ = tf.import_graph_def(graphDef, name='')                                     # import the graph into the current default Graph

sess=tf.Session()
finalTensor = sess.graph.get_tensor_by_name('final_result:0')

while True:
    image = cv2.imread('C:/Users/Nimzan/PycharmProjects/FinalApplication/tensorFace.jpg')
    tensorEmotions, tensorValues = tensor.getEmotions(image,sess,finalTensor)
    tensorEmotion = tensorEmotions[0]

    print(tensorEmotion)
