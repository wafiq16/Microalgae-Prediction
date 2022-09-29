import tkinter
from tkinter import messagebox 
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import serial
import csv


class App:
     def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.video_source = video_source

         self.inEMaR=tkinter.DoubleVar()
         self.inEMaG=tkinter.DoubleVar()
         self.inEMaB=tkinter.DoubleVar()
         self.inEMeR=tkinter.DoubleVar()
         self.inEMeG=tkinter.DoubleVar()
         self.inEMeB=tkinter.DoubleVar()
         
         self.dataR=tkinter.IntVar()
         self.dataG=tkinter.IntVar()
         self.dataB=tkinter.IntVar()
         
         self.a=0
         self.cntImg=0
         self.cntSense=0
         self.counter=0
         
         
         #self.window.geometry("1024x600")
 
         # open video source (by default this will try to open the computer webcam)
         self.vid = MyVideoCapture(self.video_source)
         
         self.ser = MySerialEvent()

         self.frameL=tkinter.Frame(window, width = self.vid.width, height = self.vid.height)
         self.frameL.grid(row=0, column=0)

         self.frameR=tkinter.Frame(window, height = self.vid.height)
         self.frameR.grid(row=0, column=1)

         self.frameRL=tkinter.Frame(self.frameR)
         self.frameRL.grid(row=0, column=0)
         
         self.frameRR=tkinter.Frame(self.frameR)
         self.frameRR.grid(row=0, column=1)
         
         # Create a canvas that can fit the above video source size
         self.canvas = tkinter.Canvas(self.frameL,  width = self.vid.width, height = self.vid.height)
         self.canvas.pack()

         self.LabelH = tkinter.Label(self.frameRL, text="Histogram", width=13)
         self.LabelH.pack()

         self.LabelHMa = tkinter.Label(self.frameRL, text="MAX", width=13)
         self.LabelHMa.pack()

         self.EntryHMaR = tkinter.Entry(self.frameRL, textvariable=self.inEMaR, width=16, bg="red", fg="white")
         self.EntryHMaR.pack()

         self.EntryHMaG = tkinter.Entry(self.frameRL, textvariable=self.inEMaG, width=16, bg="green", fg="white")
         self.EntryHMaG.pack()

         self.EntryHMaB = tkinter.Entry(self.frameRL, textvariable=self.inEMaB, width=16, bg="blue", fg="white")
         self.EntryHMaB.pack()

         self.LabelHMe = tkinter.Label(self.frameRL, text="MEAN", width=13)
         self.LabelHMe.pack()

         self.EntryHMeR = tkinter.Entry(self.frameRL, textvariable=self.inEMeR, width=16, bg="red", fg="white")
         self.EntryHMeR.pack()

         self.EntryHMeG = tkinter.Entry(self.frameRL, textvariable=self.inEMeG, width=16, bg="green", fg="white")
         self.EntryHMeG.pack()

         self.EntryHMeB = tkinter.Entry(self.frameRL, textvariable=self.inEMeB, width=16, bg="blue", fg="white")
         self.EntryHMeB.pack()

         self.LabelS = tkinter.Label(self.frameRL, text="", width=13)
         self.LabelS.pack()

         self.LabelSense = tkinter.Label(self.frameRL, text="Sensor RGB", width=13)
         self.LabelSense.pack()

         self.EntrySenseR = tkinter.Entry(self.frameRL, textvariable=self.dataR, width=16, bg="red", fg="white")
         self.EntrySenseR.pack()

         self.EntrySenseG = tkinter.Entry(self.frameRL, textvariable=self.dataG, width=16, bg="green", fg="white")
         self.EntrySenseG.pack()

         self.EntrySenseB = tkinter.Entry(self.frameRL, textvariable=self.dataB, width=16, bg="blue", fg="white")
         self.EntrySenseB.pack()

         self.LabelSa = tkinter.Label(self.frameRL, text="", width=13)
         self.LabelSa.pack()

         self.ButtonCap = tkinter.Button(self.frameRR, text="Capture", width=13, command=self.snapshot)
         self.ButtonCap.pack()
         
         self.ButtonIW = tkinter.Button(self.frameRR, text="Image With\nWhite Light", width=13, command=self.ImageWhite)
         self.ButtonIW.pack()
         
         self.ButtonIR = tkinter.Button(self.frameRR, text="Image With\nRed Light", width=13, command=self.ImageRed)
         self.ButtonIR.pack()
         
         self.ButtonIG = tkinter.Button(self.frameRR, text="Image With\nGreen Light", width=13, command=self.ImageGreen)
         self.ButtonIG.pack()
         
         self.ButtonIB = tkinter.Button(self.frameRR, text="Image With\nBlue Light", width=13, command=self.ImageBlue)
         self.ButtonIB.pack()
         
         self.ButtonSense = tkinter.Button(self.frameRR, text="Sensor RGB", width=13, command=self.SensorRGB)
         self.ButtonSense.pack()
         
         self.ButtonStop = tkinter.Button(self.frameRR, text="Stop", width=13, command=self.Stop)
         self.ButtonStop.pack()
#          
#          self.ButtonCal = tkinter.Button(self.frameRR, text="Calibrate\nRGB Sensor", width=13, command=self.Cal)
#          self.ButtonCal.pack()
          
         # Button that lets the user take a snapshot
         # self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
         # self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         
         self.delay = 15
            
         self.update()
         
         self.window.mainloop()
 
     def snapshot(self):
         # Get a frame from the video source
         if self.a==6:
             ret, frame = self.vid.get_frame()
             if ret:
                 cv2.imwrite("frame-" + str(self.counter) + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                 with open('hist-'+str(self.counter)+'.csv', 'w') as csvfile:
                     filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                     filewriter.writerow(['MaxR', 'MaxG', 'MaxB', 'MeanR', 'MeanG', 'MeanB'])
                     filewriter.writerow([self.EntryHMaR.get(), self.EntryHMaG.get(), self.EntryHMaB.get(), self.EntryHMeR.get(), self.EntryHMeG.get(), self.EntryHMeB.get()])
#                          self.cntImg+=1
    
         if self.a==8:
             with open('data'+str(self.counter)+'.csv', 'w') as csvfile:
                 filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                 filewriter.writerow(['SenseR', 'SenseG', 'SenseB'])
                 filewriter.writerow([self.EntrySenseR.get(), self.EntrySenseG.get(), self.EntrySenseB.get()])
#                      self.cntSense+=1
                      
     def SensorRGB(self):
         # Get a frame from the video source
         self.a=7
         
     def Cal(self):
         # Get a frame from the video source
         self.a=9
         
     def ImageWhite(self):
         # Get a frame from the video source
         self.a=2
         
     def ImageRed(self):
         # Get a frame from the video source
         self.a=3
         
     def ImageGreen(self):
         # Get a frame from the video source
         self.a=4
         
     def ImageBlue(self):
         # Get a frame from the video source
         self.a=5
         
     def Stop(self):
         # Get a frame from the video source
         self.a=0
             

     def update(self):
         # Get a frame from the video source
         
         def zero():
             self.ser.sendserial(b"m")
             self.ser.sendserial(b"n")
             self.a=1
         
         def one():
             self.EntryHMaR.delete(0,"end")
             self.EntryHMaG.delete(0,"end")
             self.EntryHMaB.delete(0,"end")
             self.EntryHMeR.delete(0,"end")
             self.EntryHMeG.delete(0,"end")
             self.EntryHMeB.delete(0,"end")
             self.EntrySenseR.delete(0,"end")
             self.EntrySenseG.delete(0,"end")
             self.EntrySenseB.delete(0,"end")
             self.canvas.delete("all")
             #self.ser.sendserial(b"m")
             #self.ser.sendserial(b"n")
         
         def two():
             self.ser.sendserial(b"w")
             self.a=6
             
         def three():
             self.ser.sendserial(b"r")
             self.a=6
             
         def four():
             self.ser.sendserial(b"g")
             self.a=6
             
         def five():
             self.ser.sendserial(b"b")
             self.a=6
         
         def six():      
             ret, frame = self.vid.get_frame()
             
             ret1, display = self.vid.get_frame()
                          
             display = cv2.rectangle(display, (0,int(self.vid.height)), (int(self.vid.width),int(self.vid.height/2)), (255,0,0), 1)
             
             frame = frame[int(self.vid.height/2):int(self.vid.height),0:0+int(self.vid.width-1)]
                      
             histR=cv2.calcHist([frame], [0], None, [256], [0, 256])
             histG=cv2.calcHist([frame], [1], None, [256], [0, 256])
             histB=cv2.calcHist([frame], [2], None, [256], [0, 256])

             sumAllR=0
             sumMulR=0
             sumAllG=0
             sumMulG=0
             sumAllB=0
             sumMulB=0
     
             if ret:
                 self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(display))
                 self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                 for i in range(0,256):
                     if(histR[i]==np.max(histR)):
                         self.inEMaR.set(i)
                     sumAllR+=histR[i]
                     sumMulR+=i*histR[i]
                     if(histG[i]==np.max(histG)):
                         self.inEMaG.set(i)
                     sumAllG+=histG[i]
                     sumMulG+=i*histG[i]
                     if(histB[i]==np.max(histB)):
                         self.inEMaB.set(i)
                     sumAllB+=histB[i]
                     sumMulB+=i*histB[i]
                           
                 self.inEMeR.set(int(sumMulR/sumAllR))
                 self.inEMeG.set(int(sumMulG/sumAllG))
                 self.inEMeB.set(int(sumMulB/sumAllB))
                 
                 if (self.counter!=100):
                     self.ButtonCap.invoke()
                     self.counter+=1
                 else:
                     messagebox.showinfo("info", "100 data saved")
                     self.counter=0
                 
        
         def seven():
             self.ser.flushbuf()
             self.ser.sendserial(b"l")
             self.a=8
        
         def eight():
             self.ser.flushbuf()
             self.ser.sendserial(b"a")
             self.ser.flushbuf()
             data=self.ser.readserial()
             x=data.split(",")
             self.dataR.set(x[3])
             self.dataG.set(x[4])
             self.dataB.set(x[5])
             
             if (self.counter!=100):
                 self.ButtonCap.invoke()
                 self.counter+=1
                 time.sleep(1/2)
             else:
                 messagebox.showinfo("info", "100 data saved")
                 self.counter=0
             
         def nine():
             self.ser.flushbuf()
             self.ser.sendserial(b"n")
             time.sleep(1)
             self.ser.flushbuf()
             self.ser.sendserial(b"z")
             self.ser.flushbuf()
             time.sleep(1)
             self.ser.sendserial(b"l")
             time.sleep(1)
             self.ser.flushbuf()
             self.ser.sendserial(b"q")
             self.ser.flushbuf()
             time.sleep(1)
             self.a=0
         
         switcher={
            0:zero,
            1:one,
            2:two,
            3:three,
            4:four,
            5:five,
            6:six,
            7:seven,
            8:eight,
            9:nine,
         }
         func = switcher.get(self.a)
         func()
 
         self.window.after(1, self.update)

class MySerialEvent:
     def __init__(self):
         self.ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=15)
         self.ser.flush()
    
     def readserial(self):
         # Get a frame from the video source
         line = self.ser.readline().decode().rstrip()
         return line
         
     def sendserial(self,a):
         # Get a frame from the video source
         self.ser.write(a)
         
     def flushbuf(self):
         # Get a frame from the video source
         self.ser.flush()
             
    

 
class MyVideoCapture:
     def __init__(self, video_source=0):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)

         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (ret, None)
 
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()
 
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Trial Program Final Project")
