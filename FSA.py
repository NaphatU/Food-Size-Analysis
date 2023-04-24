from cmath import pi
from itertools import count
from msilib.schema import SelfReg
from re import I
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
from cv2 import filter2D, norm, polarToCart
from idna import valid_contextj
from matplotlib import image
from DIP_Functions import *
import pymeanshift as pms
import numpy as np
import skimage
import scipy.stats as st
import matplotlib.pyplot as plt
from skimage import measure
from skimage import filters
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import math

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(360, 586)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imageLb = QtWidgets.QLabel(self.centralwidget)
        self.imageLb.setGeometry(QtCore.QRect(30, 20, 300, 300))
        self.imageLb.setText("")
        self.imageLb.setObjectName("label")
        self.openBtn = QtWidgets.QPushButton(self.centralwidget)
        self.openBtn.setGeometry(QtCore.QRect(20, 360, 71, 31))
        self.openBtn.setObjectName("pushButton")
        self.grayBtn = QtWidgets.QPushButton(self.centralwidget)
        self.grayBtn.setGeometry(QtCore.QRect(20, 400, 71, 31))
        self.grayBtn.setObjectName("grayBtn")
        self.showImBtn = QtWidgets.QPushButton(self.centralwidget)
        self.showImBtn.setGeometry(QtCore.QRect(20, 440, 71, 31))
        self.showImBtn.setObjectName("showImBtn")
        self.smoothBtn = QtWidgets.QPushButton(self.centralwidget)
        self.smoothBtn.setGeometry(QtCore.QRect(110, 360, 71, 31))
        self.smoothBtn.setObjectName("smoothBtn")
        self.meanshiftBtn = QtWidgets.QPushButton(self.centralwidget)
        self.meanshiftBtn.setGeometry(QtCore.QRect(110, 400, 71, 31))
        self.meanshiftBtn.setObjectName("meanshiftBtn")
        self.graphBtn = QtWidgets.QPushButton(self.centralwidget)
        self.graphBtn.setGeometry(QtCore.QRect(110, 440, 71, 31))
        self.graphBtn.setObjectName("graphBtn")
        self.commitBtn = QtWidgets.QPushButton(self.centralwidget)
        self.commitBtn.setGeometry(QtCore.QRect(250, 360, 75, 31))
        self.commitBtn.setObjectName("commitBtn")
        self.undoBtn = QtWidgets.QPushButton(self.centralwidget)
        self.undoBtn.setGeometry(QtCore.QRect(250, 400, 75, 31))
        self.undoBtn.setObjectName("undoBtn")
        self.resetBtn = QtWidgets.QPushButton(self.centralwidget)
        self.resetBtn.setGeometry(QtCore.QRect(250, 440, 75, 31))
        self.resetBtn.setObjectName("resetBtn")
        self.displayText = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.displayText.setGeometry(QtCore.QRect(20, 480, 221, 81))
        self.displayText.setObjectName("displayText")
        self.analyzeBtn = QtWidgets.QPushButton(self.centralwidget)
        self.analyzeBtn.setGeometry(QtCore.QRect(250, 480, 75, 81))
        self.analyzeBtn.setObjectName("analyzeBtn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 360, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.openBtn.clicked.connect(self.loadImage)
        self.grayBtn.clicked.connect(self.grayConvert)
        self.showImBtn.clicked.connect(self.showImg)
        self.commitBtn.clicked.connect(self.CommitChange)
        self.smoothBtn.clicked.connect(self.smoothImg)
        self.meanshiftBtn.clicked.connect(self.meanshiftImg)
        self.undoBtn.clicked.connect(self.Undo)
        self.resetBtn.clicked.connect(self.ResetImage)
        self.graphBtn.clicked.connect(self.plotHist)
        self.analyzeBtn.clicked.connect(self.fullFuction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Chewed food size analysis"))
        self.openBtn.setText(_translate("MainWindow", "Open Image"))
        self.grayBtn.setText(_translate("MainWindow", "Grayscale"))
        self.showImBtn.setText(_translate("MainWindow", "Zoom"))
        self.smoothBtn.setText(_translate("MainWindow", "Smooth"))
        self.meanshiftBtn.setText(_translate("MainWindow", "Meanshift"))
        self.graphBtn.setText(_translate("MainWindow", "Graph plot"))
        self.commitBtn.setText(_translate("MainWindow", "Commit"))
        self.undoBtn.setText(_translate("MainWindow", "Undo"))
        self.resetBtn.setText(_translate("MainWindow", "Reset"))
        self.analyzeBtn.setText(_translate("MainWindow", "Analyze"))
        

    def loadImage(self):
        self.filename = QFileDialog.getOpenFileName(filter="image (*.*)")[0]
        self.image = cv2.imread(self.filename)
        self.image = imutils.resize(self.image,width=600)
        self.originimage = self.image
        self.areaPixel = []
        self.setPhoto(self.image)

    def ResetImage(self):
        self.image = self.originimage
        self.tempimg = self.image
        self.setPhoto(self.originimage)

    def setPhoto(self,image):
        self.tmp = image
        image = imutils.resize(image,width=300)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.imageLb.setPixmap(QtGui.QPixmap.fromImage(image))

    def gradientImg(self):
        xmask = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float64)
        ymask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)
        self.tempimg = imgradient(self.image,xmask,ymask)
        self.setPhoto(self.tempimg)

    def laplacianImg(self):
        mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=np.float64)
        self.tempimg = cv2.filter2D(self.image,-1,mask)
        self.image = self.tempimg
        self.setPhoto(self.tempimg)

    def CommitChange(self):
        self.image = self.tempimg

    def Undo(self):
        self.tempimg = self.image
        self.setPhoto(self.image)

    def smoothImg(self):
        mask = np.array([[1,1,1],[1,1,1],[1,1,1]],dtype=np.float64)
        mask = mask/8
        self.tempimg = cv2.filter2D(self.image,-1,mask)
        self.setPhoto(self.tempimg)

    def changeMedianwsize(self):
        self.medianwsize = 11
        self.tempimg = cv2.medianBlur(self.image,self.medianwsize)
        self.setPhoto(self.tempimg)   

    def changeSharpeningK(self,value):
        k = value/10
        self.labelKLaplacian.setText("k="+str(k))
        self.tempimg = LaplacianSharpening(self.image, k)        
        self.setPhoto(self.tempimg)

    def meanshiftImg(self):
        (self.tempimg, labels_image, number_regions) = pms.segment(self.image, spatial_radius=5, range_radius=5, min_density=5)
        self.setPhoto(self.tempimg)
        #destination 60

    def grayConvert(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.tempimg = gray
        self.tempimg = cv2.cvtColor(self.tempimg, cv2.COLOR_BGR2RGB)
        self.setPhoto(self.tempimg)

    def medianFilter(self):
        self.tempimg = cv2.medianBlur(self.image,3)
        self.image = self.tempimg
        self.setPhoto(self.tempimg)

    def th(self, value):
        retval, threshold = cv2.threshold(self.image, value, 255, cv2.THRESH_BINARY)
        self.tempimg = threshold
        self.setPhoto(self.tempimg)

    def showImg(self):
        oImg = self.originimage
        edgeMap = self.tempimg
        pout1 = cv2.cvtColor(edgeMap, cv2.COLOR_BGR2RGB)
        plt.imshow(pout1)
        plt.show()

    def findEdge2(self):
        imgBn = binaryImg(self.image)
        img = ~imgBn[0]
        img = img.astype(np.uint8)

        self.bn = ~img

        oImg = self.originimage
        edgeMap = ~self.bn
        # self.bn = O
        nrow, ncol = edgeMap.shape
        fe = findEdge(edgeMap,nrow,ncol) 
        oe = np.ones((nrow,ncol))
        r,g,b=cv2.split(oImg)
        r = r*oe
        g = g*fe
        b = b*fe
        pout = cv2.merge((r,g,b))
        pout = pout.astype(np.uint8)
        self.tempimg = pout
        self.setPhoto(self.tempimg)
        both = np.hstack((imgBn[1], ~img))
        plt.imshow(both)
        # plt.imshow(pout1)
        plt.show()

    def labelTest(self):
        im = self.bn
        im = ~im
        all_labels = measure.label(im)
        im_labels = measure.label(im, background=0)
        # print(all_labels)
        pout = (im_labels) 
        self.lbImgBn =pout
        pout1 = (pout==0)
        pout = pout.astype(np.uint8)  
        pout*=255
        self.lbIm = pout
        plt.figure(figsize=(9, 3.5))
        plt.subplot(131)
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(pout1, cmap='nipy_spectral')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(im_labels, cmap='nipy_spectral')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        self.displayText.setPlainText("Label..")
        propsa = measure.regionprops(self.lbIm)
        length = len(propsa)
        self.countLb = countM(self.lbIm)
        print(length)
        self.setPhoto(self.lbIm)


    def morpho(self):       
        img = ~self.bn
        img = img.astype(np.uint8)
        self.bn = img

    def countSizeImg(self):
        propsa = measure.regionprops(self.lbIm)
        length = len(propsa)
        self.countLb = length

    
    def particleC(self):
        img = ~self.bn
        pout = particleCount(img)
        print (pout)

    def analyzeImg(self):
        img2 = self.lbImgBn
        countPt2 = particleCount(~self.bn)
        sq = math.sqrt(600)
        mmp = list()
        k=0
        for i in range(countPt2+1):
            img = (img2==i)
            img = img.astype(np.uint8)  
            img*=255
            cnj = 0
            for j in img:
                if j.any() > 0:
                    cnj += 1
            if(cnj>0):
                cnj *= (150*150)/600
                mmp.append(cnj)
                # print("area = "+str(cnj))
                if(i>0):
                    self.displayText.appendPlainText("NO:"+str(i)+"  Area  =  "+str("{:.2f}".format(cnj)))
                    k+=1
        mmp.pop(0)
        sum = 0.0
        for m in mmp:
            sum += m
            print(m)
        sum/=k
        print("MEAN = "+str(sum)) 
        fMed = mmp
        fMed.sort()
        Med = fMed[int(k/2)]
        self.displayText.appendPlainText("NUMBER OF PIECES = "+str("{:.2f}".format(k)))
        self.displayText.appendPlainText("MEDIAN = "+str("{:.2f}".format(Med)))
        self.displayText.appendPlainText("MEAN = "+str("{:.2f}".format(sum)))
        self.areaPixel = mmp
    
    def fullFuction(self):
        imgBn = binaryImgGaussian2(self.image)
        imgbn2 = ~imgBn[0]
        img = ~imgBn[0]
        img = img.astype(np.uint8)
        kernelSize = [(3,3),(3,3)]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize[0])
        mpOpen = cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize[1])
        mpClose = cv2.morphologyEx(mpOpen, cv2.MORPH_OPEN,kernel)
        self.bn = ~mpClose


        oImg = self.originimage
        edgeMap = ~self.bn
        # self.bn = O
        nrow, ncol = edgeMap.shape
        fe = findEdge(edgeMap,nrow,ncol) 
        oe = np.ones((nrow,ncol))
        r,g,b=cv2.split(oImg)
        r = r*oe
        g = g*fe
        b = b*fe
        pout = cv2.merge((r,g,b))
        pout = pout.astype(np.uint8)
        self.tempimg = pout
        self.setPhoto(self.tempimg)
        im = self.bn
        im = ~im
        all_labels = measure.label(im)
        im_labels = measure.label(im, background=0)
        # print(all_labels)
        pout = (im_labels) 
        self.lbImgBn =pout
        pout1 = (pout==0)
        pout = pout.astype(np.uint8)  
        pout*=255
        self.lbIm = pout
        self.displayText.setPlainText("Label..")
        propsa = measure.regionprops(self.lbIm)
        length = len(propsa)
        self.countLb = countM(self.lbIm)
        img2 = self.lbImgBn
        countPt2 = particleCount(~self.bn)
        sq = math.sqrt(600)
        mmp = list()
        k=0
        for i in range(countPt2+1):
            img = (img2==i)
            img = img.astype(np.uint8)  
            img*=255
            cnj = 0
            for j in img:
                if j.any() > 0:
                    cnj += 1
            if(cnj>0):
                cnj *= (150*150)/600
                mmp.append(cnj)
                # print("area = "+str(cnj))
                if(i>0):
                    self.displayText.appendPlainText("NO:"+str(i)+"  Area  =  "+str("{:.2f}".format(cnj)))
                    k+=1
        mmp.pop(0)
        sum = 0.0
        for m in mmp:
            sum += m
            # print(m)
        sum/=k
        # print("MEAN = "+str(sum)) 
        fMed = mmp
        fMed.sort()
        Med = fMed[int(k/2)]
        self.displayText.appendPlainText("NUMBER OF PIECES = "+str("{:.2f}".format(k)))
        self.displayText.appendPlainText("MEDIAN = "+str("{:.2f}".format(Med)))
        self.displayText.appendPlainText("MEAN = "+str("{:.2f}".format(sum)))
        self.areaPixel = mmp
        

    def plotHist(self):
        x = self.areaPixel
        if(len(x)>0):
            plt.style.use('ggplot')
            plt.hist(x, bins=100)
            plt.ylabel("Frequency")
            plt.xlabel("Size (sq. mm)")
            plt.title("")
            plt.show()
        else:
            self.displayText.setPlainText("Press Analyze Botton..")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
