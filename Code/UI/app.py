# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pathlib import Path
from Code.show_data import showData
from Code.pyRadiomics import extractRadiomics
from Code.data import CustomDataset
from PIL import ImageOps, Image
import cv2
import numpy as np


class Ui_MainWindow(object):

    def __init__(self):
        super().__init__()

        self.filePath = ''
        self.folderPath = ''
        self.currentDir = os.getcwd()
        self.images = []
        self.image_name = []
        self.image_folder = []
        self.folder_name = []
        self.masks = []
        self.mask_folder = []
        # self.output_dir = os.path.abspath('..\\greyscale_images')

    def setupUi(self, MainWindow):
        self.window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(951, 633)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setEnabled(True)
        self.frame.setGeometry(QtCore.QRect(10, 20, 541, 391))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.noNormaButton = QtWidgets.QRadioButton(self.frame)
        self.noNormaButton.setGeometry(QtCore.QRect(20, 250, 141, 20))
        self.noNormaButton.setObjectName("noNormaButton")
        self.minMaxButton = QtWidgets.QRadioButton(self.frame)
        self.minMaxButton.setGeometry(QtCore.QRect(20, 280, 171, 20))
        self.minMaxButton.setObjectName("minMaxButton")
        self.meanStdButton = QtWidgets.QRadioButton(self.frame)
        self.meanStdButton.setGeometry(QtCore.QRect(20, 310, 181, 20))
        self.meanStdButton.setObjectName("meanStdButton")
        self.perButton = QtWidgets.QRadioButton(self.frame)
        self.perButton.setGeometry(QtCore.QRect(20, 340, 161, 20))
        self.perButton.setObjectName("perButton")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(30, 200, 121, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.area1Button = QtWidgets.QRadioButton(self.frame)
        self.area1Button.setGeometry(QtCore.QRect(30, 110, 161, 20))
        self.area1Button.setObjectName("area1Button")
        self.area2Button = QtWidgets.QRadioButton(self.frame)
        self.area2Button.setGeometry(QtCore.QRect(30, 130, 141, 20))
        self.area2Button.setObjectName("area2Button")
        self.bothButton = QtWidgets.QRadioButton(self.frame)
        self.bothButton.setGeometry(QtCore.QRect(30, 150, 141, 20))
        self.bothButton.setObjectName("bothButton")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setGeometry(QtCore.QRect(30, 30, 121, 31))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setGeometry(QtCore.QRect(10, 70, 171, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(0, 0, 551, 391))
        self.widget.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.widget.setObjectName("widget")
        self.widget_5 = QtWidgets.QWidget(self.widget)
        self.widget_5.setGeometry(QtCore.QRect(10, 30, 191, 151))
        self.widget_5.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.widget_5.setObjectName("widget_5")
        self.widget_6 = QtWidgets.QWidget(self.widget)
        self.widget_6.setGeometry(QtCore.QRect(10, 200, 191, 171))
        self.widget_6.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.widget_6.setObjectName("widget_6")
        self.csvButton = QtWidgets.QPushButton(self.widget)
        self.csvButton.setGeometry(QtCore.QRect(380, 330, 150, 50))
        self.csvButton.setStyleSheet("background-color: rgb(210, 204, 204);\n"
"")
        self.csvButton.setObjectName("csvButton")
        self.statusLabel = QtWidgets.QLabel(self.widget)
        self.statusLabel.setGeometry(QtCore.QRect(220, 340, 151, 31))
        self.statusLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.statusLabel.setStyleSheet("")
        self.statusLabel.setText("")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.statusLabel.setObjectName("statusLabel")
        self.loadMaskButton = QtWidgets.QPushButton(self.widget)
        self.loadMaskButton.setGeometry(QtCore.QRect(210, 30, 150, 50))
        self.loadMaskButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.loadMaskButton.setObjectName("loadMaskButton")
        self.loadMaskFolderButton = QtWidgets.QPushButton(self.widget)
        self.loadMaskFolderButton.setGeometry(QtCore.QRect(380, 30, 150, 50))
        self.loadMaskFolderButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.loadMaskFolderButton.setObjectName("loadMaskFolderButton")
        self.generateSegButton = QtWidgets.QPushButton(self.widget)
        self.generateSegButton.setGeometry(QtCore.QRect(380, 100, 150, 50))
        self.generateSegButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.generateSegButton.setObjectName("generateSegButton")
        self.generateNormButton = QtWidgets.QPushButton(self.widget)
        self.generateNormButton.setGeometry(QtCore.QRect(380, 200, 150, 50))
        self.generateNormButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.generateNormButton.setObjectName("generateNormButton")
        self.widget_6.raise_()
        self.widget_5.raise_()
        self.csvButton.raise_()
        self.statusLabel.raise_()
        self.loadMaskButton.raise_()
        self.loadMaskFolderButton.raise_()
        self.generateSegButton.raise_()
        self.generateNormButton.raise_()
        self.widget.raise_()
        self.noNormaButton.raise_()
        self.minMaxButton.raise_()
        self.meanStdButton.raise_()
        self.perButton.raise_()
        self.label.raise_()
        self.area1Button.raise_()
        self.area2Button.raise_()
        self.bothButton.raise_()
        self.label_5.raise_()
        self.label_6.raise_()
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 420, 541, 201))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.loadDataButton = QtWidgets.QPushButton(self.frame_2)
        self.loadDataButton.setGeometry(QtCore.QRect(40, 110, 150, 50))
        self.loadDataButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.loadDataButton.setObjectName("loadDataButton")
        self.showButton = QtWidgets.QPushButton(self.frame_2)
        self.showButton.setGeometry(QtCore.QRect(350, 130, 150, 50))
        self.showButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.showButton.setObjectName("showButton")
        self.button3D = QtWidgets.QCheckBox(self.frame_2)
        self.button3D.setGeometry(QtCore.QRect(50, 170, 131, 21))
        self.button3D.setObjectName("button3D")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(200, 10, 121, 31))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.LoadFolderButton = QtWidgets.QPushButton(self.frame_2)
        self.LoadFolderButton.setGeometry(QtCore.QRect(40, 50, 150, 50))
        self.LoadFolderButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.LoadFolderButton.setObjectName("LoadFolderButton")
        self.widget_4 = QtWidgets.QWidget(self.frame_2)
        self.widget_4.setGeometry(QtCore.QRect(0, 0, 541, 201))
        self.widget_4.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.widget_4.setObjectName("widget_4")
        self.widget_4.raise_()
        self.loadDataButton.raise_()
        self.showButton.raise_()
        self.button3D.raise_()
        self.label_3.raise_()
        self.LoadFolderButton.raise_()
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(560, 20, 381, 601))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.loadRadioDataButton = QtWidgets.QPushButton(self.frame_3)
        self.loadRadioDataButton.setGeometry(QtCore.QRect(20, 90, 150, 50))
        self.loadRadioDataButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.loadRadioDataButton.setObjectName("loadRadioDataButton")
        self.classifyButton = QtWidgets.QPushButton(self.frame_3)
        self.classifyButton.setGeometry(QtCore.QRect(20, 150, 150, 50))
        self.classifyButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.classifyButton.setObjectName("classifyButton")
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        self.label_2.setGeometry(QtCore.QRect(130, 10, 121, 31))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.frame_3)
        self.label_4.setGeometry(QtCore.QRect(190, 160, 121, 31))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.widget_3 = QtWidgets.QWidget(self.frame_3)
        self.widget_3.setGeometry(QtCore.QRect(0, 0, 381, 601))
        self.widget_3.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.widget_3.setObjectName("widget_3")
        self.show_classifier = QtWidgets.QLabel(self.widget_3)
        self.show_classifier.setEnabled(True)
        self.show_classifier.setGeometry(QtCore.QRect(20, 240, 350, 350))
        self.show_classifier.setText("")
        self.show_classifier.setObjectName("show_classifier")
        self.widget_3.raise_()
        self.loadRadioDataButton.raise_()
        self.classifyButton.raise_()
        self.label_2.raise_()
        self.label_4.raise_()
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(0, 0, 961, 641))
        self.widget_2.setStyleSheet("background-color: rgb(81, 93, 78);")
        self.widget_2.setObjectName("widget_2")
        self.widget_2.raise_()
        self.frame.raise_()
        self.frame_2.raise_()
        self.frame_3.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.loadDataButton.clicked.connect(lambda: self.getFile())
        self.LoadFolderButton.clicked.connect(lambda: self.getFolder())
        self.showButton.clicked.connect(lambda: self.showImages())
        self.loadMaskButton.clicked.connect(lambda: self.getMask())
        self.loadMaskFolderButton.clicked.connect(lambda: self.getMaskFolder())
        self.csvButton.clicked.connect((lambda: self.radiomics()))


    def getFile(self):
        # Open window to choose file
        self.filePath, _ = QFileDialog.getOpenFileNames(self.window, 'Choose an image', "${HOME}", "Formats: (*.png )")

        for filePath in self.filePath:
            image = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
            name = os.path.basename(filePath)
            # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.images.append(image)
            self.image_name.append(name)

    def getFolder(self):
        # Open window to choose file
        self.folderPath = QFileDialog.getExistingDirectory(self.window, 'Choose a patient Directory', "${HOME}")

        if self.folderPath:
            files = os.listdir(self.folderPath)

            for fileName in files:
                filePath = os.path.join(self.folderPath, fileName)
                if fileName.lower().endswith(('.png')):
                    image = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
                    nameFolder = os.path.basename(self.folderPath)
                    self.image_folder.append(image)
                    self.folder_name.append(nameFolder)
        else:
            print("Empty directory")


    def getMask(self):
        # Open window to choose file
        self.filePath, _ = QFileDialog.getOpenFileNames(self.window, 'Choose an image', "${HOME}", "Formats: (*.png )")

        for filePath in self.filePath:
            mask = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
            #name = os.path.basename(filePath)
            self.masks.append(mask)


    def getMaskFolder(self):
        # Open window to choose file
        self.folderPath = QFileDialog.getExistingDirectory(self.window, 'Choose a patient Directory', "${HOME}")

        if self.folderPath:
            files = os.listdir(self.folderPath)

            for fileName in files:
                filePath = os.path.join(self.folderPath, fileName)
                if fileName.lower().endswith(('.png')):
                    maskF = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
                    #nameFolder = os.path.basename(self.folderPath)
                    self.mask_folder.append(maskF)
        else:
            print("Empty directory")

    def showImages(self):
        if self.images:
            print("Number of images:", len(self.images))
            showData(self.images, self.image_name)
        elif self.image_folder:
            print("Number of images in the folder:", len(self.image_folder))
            showData(self.image_folder, self.folder_name)
        else:
            print("No files selected.")



    def radiomics(self):
        if self.radioButton.isChecked():
            extractRadiomics(self.images)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Textural analysis"))
        self.noNormaButton.setText(_translate("MainWindow", "No normalization"))
        self.minMaxButton.setText(_translate("MainWindow", "Min-Max normalization"))
        self.meanStdButton.setText(_translate("MainWindow", "Mean/Std normalization"))
        self.perButton.setText(_translate("MainWindow", "Percentile normalization"))
        self.label.setText(_translate("MainWindow", "Normalization"))
        self.area1Button.setText(_translate("MainWindow", "Internal region"))
        self.area2Button.setText(_translate("MainWindow", "External region"))
        self.bothButton.setText(_translate("MainWindow", "Both"))
        self.label_5.setText(_translate("MainWindow", "Segmentation"))
        self.label_6.setText(_translate("MainWindow", "Select the area to analyze:"))
        self.csvButton.setText(_translate("MainWindow", "Generate CSV"))
        self.loadMaskButton.setText(_translate("MainWindow", "Load Mask"))
        self.loadMaskFolderButton.setText(_translate("MainWindow", "Load Mask Folder"))
        self.generateSegButton.setText(_translate("MainWindow", "Generate Segmentation"))
        self.generateNormButton.setText(_translate("MainWindow", "Normalization"))
        self.loadDataButton.setText(_translate("MainWindow", "Load Data"))
        self.showButton.setText(_translate("MainWindow", "Show Data"))
        self.button3D.setText(_translate("MainWindow", "3D Image"))
        self.label_3.setText(_translate("MainWindow", "Data"))
        self.LoadFolderButton.setText(_translate("MainWindow", "Load Folder"))
        self.loadRadioDataButton.setText(_translate("MainWindow", "Load Data"))
        self.classifyButton.setText(_translate("MainWindow", "Classify"))
        self.label_2.setText(_translate("MainWindow", "Classification"))
        self.label_4.setText(_translate("MainWindow", "Classifier"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
