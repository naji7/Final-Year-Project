import os
from tkinter.filedialog import askdirectory
from tkinter import *

import cv2
import tensorflow as tf
import FisherFace.FisherDetect as fish
import TensorFlow.TensorDetect as tensor
from PIL import Image, ImageOps
from tkinter.font import Font
from tkinter.tix import *

import pygame
from mutagen.id3 import ID3
from tkinter import *

from pygame import *
import os
from tkinter.filedialog import askdirectory



root = Tk()
root.minsize(300, 300)



def fingerAndFacial():
    def gohead():

        third = Toplevel()

        third.title("Music Player")
        third.geometry("500x360+120+120")
        label = Label(third, text='Music Player')
        label.pack()
        label = Label(third)
        label.pack()
        label12 = Label(third, text=finalEmotion, font=20).pack()

        label = Label(third)
        label.pack()
        label = Label(third)
        label.pack()
        label = Label(third)
        label.pack()

        v = StringVar()
        songlabel = Label(third, textvariable=v, width=35)

        index = 0

        def updatelabel():
            global index
            global songname

            # v.set(realnames[index])
            # v.set("happy")
            # return songname


        def nextsong(event):
            global index
            # index += 1
            # pygame.mixer.music.load(listofsongs[index])
            # pygame.mixer.music.play()
            if finalEmotion == "happy":
                #pygame.init()
                #drum = pygame.mixer.Sound("happy.ogg")
                #drum.play()
                mixer.init()
                pygame.mixer.Sound("happy.ogg")
                pygame.mixer.music.load("happy.ogg")
                pygame.mixer.music.play()


            if finalEmotion == "surprise":
                mixer.init()
                pygame.mixer.Sound("sad.ogg")
                pygame.mixer.music.load("sad.ogg")
                pygame.mixer.music.play()
            if finalEmotion == "fear":
                mixer.init()
                pygame.mixer.Sound("fear.ogg")
                pygame.mixer.music.load("fear.ogg")
                pygame.mixer.music.play()

            updatelabel()

        def stopsong(event):
            pygame.mixer.music.stop()
            v.set("")
            # return songname





       # listbox = Listbox(third)
       # listbox.pack()

        nextbutton = Button(third, text='Play Music')
        nextbutton.pack()
        label = Label(third)
        label.pack()
        stopbutton = Button(third, text='Stop Music')
        stopbutton.pack()

        nextbutton.bind("<Button-1>", nextsong)

        stopbutton.bind("<Button-1>", stopsong)

        songlabel.pack()

    def combine():
        gohead()
        second.destroy()



    second = Toplevel()
    second.geometry("500x370+120+120")
    label11 = Label(second).pack()
    frame1 = Frame(second)

    frame1.pack()

    scrollBar = Scrollbar(frame1)
    scrollBar.pack(side = RIGHT, fill= Y)
    listbox = Listbox(frame1,yscrollcommand = scrollBar.set)
    listbox.pack(side=LEFT, fill="both" )

    ###############################################################################file paths start
    RETRAINED_LABELS_TXT_FILE_LOC = "C:/Users/Nimzan/PycharmProjects/FinalApplication/TensorFlow/retrained_labels.txt"
    RETRAINED_GRAPH_PB_FILE_LOC = "C:/Users/Nimzan/PycharmProjects/FinalApplication/TensorFlow/retrained_graph.pb"
    ###############################################################################file paths end

    retrainedGraphFile = tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb')  # Loading the trained graph
    graphDef = tf.GraphDef()  # creating graphdef object
    graphDef.ParseFromString(retrainedGraphFile.read())  # parsing the graph
    tf.import_graph_def(graphDef, name='')  # import the graph into the current default Graph

    sess = tf.Session()
    finalTensor = sess.graph.get_tensor_by_name('final_result:0')

    #####################################d###################intializing tensorflow variables end

    mouth_cascade = cv2.CascadeClassifier('TensorflowMouth/haarcascade_mcs_mouth.xml')

    cascPath = 'haarcascade_filters/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    import serial
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    arduino = serial.Serial('COM3', 115200, timeout=10)

    filename = "C:/Users/Nimzan/PycharmProjects/FinalApplication/Pulse/model.sav"
    loaded_model = joblib.load(filename)


    n = 21
    total_numbers = n
    sum = 0
    #while True:
    while (n >= 0):

        list = []
        for i in range(22):

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
                    listbox.insert(END, "BPM : " + str(bpm) + "    Expected Emotions : " + Pulseemotions[0])

            #################################################################################################################### pulse end

            ret, frame = video_capture.read()  # capturing frames in in videostream
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting the frame to grayscale

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

            if len(faces) > 0:
                print("Primary Face Found")
                # label1 = Label(second,text='primary face found', font=20 ).pack()
                listbox.insert(END, "Primary face found")

                primaryFaceCoordinate = faces[0]
                print(primaryFaceCoordinate)
                # label2 = Label(second, text=primaryFaceCoordinate, font=20).pack()
                listbox.insert(END, primaryFaceCoordinate)

                [x, y, w, h] = primaryFaceCoordinate

                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (350, 350))  # cropping the face 350*350

                #####################################emotion start
                fisherEmotion = fish.findEmotion(face)  # fisherface

                cv2.imwrite("tensorFace.jpg", face)
                image = cv2.imread('C:/Users/Nimzan/PycharmProjects/FinalApplication/tensorFace.jpg')
                tensorEmotions, tensorValues = tensor.getEmotions(image, sess, finalTensor)
                tensorEmotion = tensorEmotions[0]
                #####################################emotion end

                #####################################comparison start
                if fisherEmotion == "happy":
                    finalEmotion = fisherEmotion
                elif fisherEmotion == "surprise":
                    finalEmotion = fisherEmotion
                elif fisherEmotion == tensorEmotions[0]:
                    finalEmotion = fisherEmotion
                elif fisherEmotion == tensorEmotions[1]:
                    finalEmotion = fisherEmotion
                elif fisherEmotion == tensorEmotions[2]:
                    finalEmotion = fisherEmotion
                else:
                    finalEmotion = tensorEmotions[0]

                print(finalEmotion)

                #####################################comparison end


            else:
                print("No face Found")
            #####################################selecting primary target face end

            if finalEmotion != Pulseemotions[0]:
                print("Emotion is not relavant to pulse")
                listbox.insert(END, "Emotion is not relavant to pulse")

            else:
                print("Emotion seems legit")
                listbox.insert(END, "Emotion seems legit")

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

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # drawing rectangle aronf the face
            cv2.imshow('Video', frame)  # Display updated frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sum += bpm
            n -= 1
            list.append(finalEmotion)



    print(list)

    def most_frequent(list):
        counter = 0
        num = list[0]

        for i in list:
            curr_frequency = list.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i

        return num

    print(most_frequent(list))

    scrollBar.config(command=listbox.yview)
    label11 = Label(second).pack()
    label1 = Label(second, text=finalEmotion, font=20).pack()
    label11 = Label(second).pack()
    nextbutton = Button(second, width=20, text='PLAY MUSIC',command=combine, bg="Gray", fg="White")
    nextbutton.pack()
    # releasing the capture after all done

    average = sum / total_numbers
    print(average)
    fin = round(average)
    print(fin)
    Pulseemotions = loaded_model.predict([[average]])
    print("BPM : " + str(fin) + "    Expected Emotions : " + Pulseemotions[0])

    video_capture.release()
    cv2.destroyAllWindows()


def ChangeLabelText(m):
    m.config(text = 'You pressed the button!')

def quit():
    root.destroy()

myFont = Font(family="Times New Roman", size=12)
label = Label(root, text='Mood Based Music Player', fg="Gray", font= "none 16 bold")
label.pack()

canvas = Canvas(root, width = 500, height = 300)
canvas.pack()
photo = PhotoImage(file = 'img1.png')
canvas.create_image(0,0, anchor = NW, image=photo)

nextbutton = Button(root,width = 20, text='START',  command=fingerAndFacial, bg= "Gray",fg="White")
nextbutton.pack()






root.geometry("500x370+120+120")
root.mainloop()
