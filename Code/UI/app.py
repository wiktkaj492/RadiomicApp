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
from Code.mask import Segmentation
from Code.normalize import NoNormalization, MinMaxNormalization, MeanStdNormalization, PercentileNormalization
from Code.pyRadiomics import Radiomics
from Code.ROI import ROI
from PIL import ImageOps, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Ui_MainWindow(object):

    def __init__(self):
        super().__init__()

        self.filePath = ''
        self.folderPath = ''
        self.currentDir = os.getcwd()
        self.input_images = []
        self.input_images_names = []
        self.input_images_paths = []
        self.folders_paths = []
        self.input_masks = []
        self.input_mask_path = []
        self.emptyMasks = []
        self.newMaskFolder = []
        self.input_param_file = ''

    def setupUi(self, MainWindow):
        self.window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1060, 633)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setEnabled(True)
        self.frame.setGeometry(QtCore.QRect(10, 20, 921, 381))
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
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(0, 0, 921, 381))
        self.widget.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.widget.setObjectName("widget")
        self.widget_5 = QtWidgets.QWidget(self.widget)
        self.widget_5.setGeometry(QtCore.QRect(10, 10, 301, 151))
        self.widget_5.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.widget_5.setObjectName("widget_5")
        self.label_5 = QtWidgets.QLabel(self.widget_5)
        self.label_5.setGeometry(QtCore.QRect(80, 0, 121, 31))
        self.label_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.bothButton = QtWidgets.QRadioButton(self.widget_5)
        self.bothButton.setGeometry(QtCore.QRect(10, 120, 141, 20))
        self.bothButton.setObjectName("bothButton")
        self.label_6 = QtWidgets.QLabel(self.widget_5)
        self.label_6.setGeometry(QtCore.QRect(10, 40, 171, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.area2Button = QtWidgets.QRadioButton(self.widget_5)
        self.area2Button.setGeometry(QtCore.QRect(10, 100, 141, 20))
        self.area2Button.setObjectName("area2Button")
        self.area1Button = QtWidgets.QRadioButton(self.widget_5)
        self.area1Button.setGeometry(QtCore.QRect(10, 80, 161, 20))
        self.area1Button.setObjectName("area1Button")
        self.widget_6 = QtWidgets.QWidget(self.widget)
        self.widget_6.setGeometry(QtCore.QRect(10, 180, 401, 191))
        self.widget_6.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.widget_6.setObjectName("widget_6")
        self.label_8 = QtWidgets.QLabel(self.widget_6)
        self.label_8.setGeometry(QtCore.QRect(0, 30, 301, 31))
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label = QtWidgets.QLabel(self.widget_6)
        self.label.setGeometry(QtCore.QRect(80, 10, 121, 21))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.numberBits = QtWidgets.QSpinBox(self.widget_6)
        self.numberBits.setGeometry(QtCore.QRect(340, 150, 42, 22))
        self.numberBits.setMinimum(1)
        self.numberBits.setMaximum(8)
        self.numberBits.setObjectName("numberBits")
        self.label_2 = QtWidgets.QLabel(self.widget_6)
        self.label_2.setGeometry(QtCore.QRect(220, 150, 111, 21))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.statusMaskLabel = QtWidgets.QLabel(self.widget)
        self.statusMaskLabel.setGeometry(QtCore.QRect(360, 60, 231, 31))
        self.statusMaskLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.statusMaskLabel.setStyleSheet("")
        self.statusMaskLabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.statusMaskLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.statusMaskLabel.setText("")
        self.statusMaskLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.statusMaskLabel.setObjectName("statusMaskLabel")
        self.csvStatusLabel = QtWidgets.QLabel(self.widget)
        self.csvStatusLabel.setGeometry(QtCore.QRect(730, 310, 151, 31))
        self.csvStatusLabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.csvStatusLabel.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.csvStatusLabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.csvStatusLabel.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.csvStatusLabel.setText("")
        self.csvStatusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.csvStatusLabel.setObjectName("csvStatusLabel")
        self.csvButton = QtWidgets.QPushButton(self.widget)
        self.csvButton.setGeometry(QtCore.QRect(730, 250, 150, 50))
        self.csvButton.setStyleSheet("background-color: rgb(210, 204, 204);\n"
"")
        self.csvButton.setObjectName("csvButton")
        self.emptyMasksList = QtWidgets.QListWidget(self.widget)
        self.emptyMasksList.setGeometry(QtCore.QRect(630, 40, 281, 171))
        self.emptyMasksList.setStyleSheet("background-color: rgb(239, 239, 239);")
        self.emptyMasksList.setFrameShape(QtWidgets.QFrame.Panel)
        self.emptyMasksList.setObjectName("emptyMasksList")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setGeometry(QtCore.QRect(710, 10, 121, 31))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.loadParamStatus = QtWidgets.QLabel(self.widget)
        self.loadParamStatus.setGeometry(QtCore.QRect(490, 310, 151, 31))
        self.loadParamStatus.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.loadParamStatus.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.loadParamStatus.setFrameShape(QtWidgets.QFrame.Panel)
        self.loadParamStatus.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.loadParamStatus.setText("")
        self.loadParamStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.loadParamStatus.setObjectName("loadParamStatus")
        self.loadParamFile = QtWidgets.QPushButton(self.widget)
        self.loadParamFile.setGeometry(QtCore.QRect(490, 250, 150, 50))
        self.loadParamFile.setStyleSheet("background-color: rgb(210, 204, 204);\n"
"")
        self.loadParamFile.setObjectName("loadParamFile")
        self.widget_6.raise_()
        self.widget_5.raise_()
        self.statusMaskLabel.raise_()
        self.csvStatusLabel.raise_()
        self.csvButton.raise_()
        self.emptyMasksList.raise_()
        self.label_7.raise_()
        self.loadParamStatus.raise_()
        self.loadParamFile.raise_()
        self.widget.raise_()
        self.noNormaButton.raise_()
        self.minMaxButton.raise_()
        self.meanStdButton.raise_()
        self.perButton.raise_()
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 420, 541, 201))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.widget_4 = QtWidgets.QWidget(self.frame_2)
        self.widget_4.setGeometry(QtCore.QRect(0, 0, 541, 201))
        self.widget_4.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.widget_4.setObjectName("widget_4")
        self.label_3 = QtWidgets.QLabel(self.widget_4)
        self.label_3.setGeometry(QtCore.QRect(210, 10, 121, 31))
        self.label_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.loadMaskFolderButton = QtWidgets.QPushButton(self.widget_4)
        self.loadMaskFolderButton.setGeometry(QtCore.QRect(360, 120, 150, 50))
        self.loadMaskFolderButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.loadMaskFolderButton.setObjectName("loadMaskFolderButton")
        self.loadDataButton = QtWidgets.QPushButton(self.widget_4)
        self.loadDataButton.setGeometry(QtCore.QRect(20, 60, 150, 50))
        self.loadDataButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.loadDataButton.setObjectName("loadDataButton")
        self.LoadFolderButton = QtWidgets.QPushButton(self.widget_4)
        self.LoadFolderButton.setGeometry(QtCore.QRect(20, 120, 150, 50))
        self.LoadFolderButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.LoadFolderButton.setObjectName("LoadFolderButton")
        self.imageNumber = QtWidgets.QLabel(self.widget_4)
        self.imageNumber.setGeometry(QtCore.QRect(190, 70, 151, 31))
        self.imageNumber.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.imageNumber.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.imageNumber.setFrameShape(QtWidgets.QFrame.Panel)
        self.imageNumber.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.imageNumber.setText("")
        self.imageNumber.setAlignment(QtCore.Qt.AlignCenter)
        self.imageNumber.setObjectName("imageNumber")
        self.maskNumber = QtWidgets.QLabel(self.widget_4)
        self.maskNumber.setGeometry(QtCore.QRect(190, 130, 151, 31))
        self.maskNumber.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.maskNumber.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.maskNumber.setFrameShape(QtWidgets.QFrame.Panel)
        self.maskNumber.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.maskNumber.setText("")
        self.maskNumber.setAlignment(QtCore.Qt.AlignCenter)
        self.maskNumber.setObjectName("maskNumber")
        self.loadMaskButton = QtWidgets.QPushButton(self.widget_4)
        self.loadMaskButton.setGeometry(QtCore.QRect(360, 60, 150, 50))
        self.loadMaskButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.loadMaskButton.setObjectName("loadMaskButton")
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(0, 0, 1071, 641))
        self.widget_2.setStyleSheet("background-color: rgb(81, 93, 78);")
        self.widget_2.setObjectName("widget_2")
        self.loadDataList = QtWidgets.QListWidget(self.widget_2)
        self.loadDataList.setGeometry(QtCore.QRect(570, 440, 231, 181))
        self.loadDataList.setStyleSheet("background-color: rgb(239, 239, 239);")
        self.loadDataList.setFrameShape(QtWidgets.QFrame.Panel)
        self.loadDataList.setObjectName("loadDataList")
        self.loadMasksList = QtWidgets.QListWidget(self.widget_2)
        self.loadMasksList.setGeometry(QtCore.QRect(820, 440, 231, 181))
        self.loadMasksList.setStyleSheet("background-color: rgb(239, 239, 239);")
        self.loadMasksList.setFrameShape(QtWidgets.QFrame.Panel)
        self.loadMasksList.setObjectName("loadMasksList")
        self.label_4 = QtWidgets.QLabel(self.widget_2)
        self.label_4.setGeometry(QtCore.QRect(620, 410, 121, 21))
        self.label_4.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.label_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_9 = QtWidgets.QLabel(self.widget_2)
        self.label_9.setGeometry(QtCore.QRect(880, 410, 121, 21))
        self.label_9.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.label_9.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.widget_3 = QtWidgets.QWidget(self.widget_2)
        self.widget_3.setGeometry(QtCore.QRect(940, 20, 111, 381))
        self.widget_3.setStyleSheet("background-color: rgb(186, 188, 162);")
        self.widget_3.setObjectName("widget_3")
        self.resetAppButton = QtWidgets.QPushButton(self.widget_3)
        self.resetAppButton.setGeometry(QtCore.QRect(10, 20, 91, 50))
        self.resetAppButton.setStyleSheet("background-color: rgb(210, 204, 204);")
        self.resetAppButton.setObjectName("resetAppButton")
        self.widget_2.raise_()
        self.frame.raise_()
        self.frame_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.loadDataButton.clicked.connect(lambda: self.getFile())
        self.LoadFolderButton.clicked.connect(lambda: self.getFolder())
        # self.showButton.clicked.connect(lambda: self.showImages())
        self.loadMaskButton.clicked.connect(lambda: self.getMask())
        self.loadMaskFolderButton.clicked.connect(lambda: self.getMaskFolder())
        # self.generateSegButton.clicked.connect(lambda: self.chooseSegmentation())
        # self.roiButton.clicked.connect(lambda: self.getROI())
        # self.generateNormButton.clicked.connect(lambda: self.chooseNormalization())
        # self.csvButton.clicked.connect(lambda: self.radiomics())
        self.csvButton.clicked.connect(lambda: self.generateCsvAll())
        self.loadParamFile.clicked.connect(lambda: self.getParamFile())
        self.resetAppButton.clicked.connect(lambda: self.reset())

    def save_debug_images(self, images, filenames, folder="debug_images"):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        for image, filename in zip(images, filenames):
            # Normalize and map the mask to a colormap for better visibility
            plt.imshow(image, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.savefig(os.path.join(folder, filename))
            plt.close()

    def getFile(self):
        # Open window to choose file
        self.filePath, _ = QFileDialog.getOpenFileNames(self.window, 'Choose an image', "${HOME}", "Formats: (*.png )")

        if not self.filePath:  # Jeśli lista jest pusta, pokaż ostrzeżenie
            QMessageBox.warning(self.window, "No Image Selected", "Please select images.")
            return

        for filePath in self.filePath:
            image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            # name = os.path.basename(filePath)
            self.input_images.append((image, filePath))
            self.save_debug_images([image], ["noNorm"])

        self.loadDataList.addItems([name for _, name in self.input_images])
        self.imageNumber.setText(f"Images: {len(self.input_images)}")

        # print(f"Image: {len(self.input_images)}")

    def getFolder(self):
        # Open window to choose file
        self.folderPath = QFileDialog.getExistingDirectory(self.window, 'Choose a patient Directory', "${HOME}")

        if self.folderPath:
            for root, dirs, files in os.walk(self.folderPath):
                for fileName in files:
                    if fileName.lower().endswith(('.png')):
                        filePath = os.path.join(root, fileName)
                        image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
                        self.input_images.append((image, filePath))
                        print(f"Image: {len(self.input_images)}")

            self.imageNumber.setText(f"Images: {len(self.input_images)}")
            self.loadDataList.addItems([name for _, name in self.input_images])
        else:
            QMessageBox.warning(self.window, "No Images Selected", "Please select patient directory.")

    def getMask(self):
        # Open window to choose file
        self.filePath, _ = QFileDialog.getOpenFileNames(self.window, 'Choose an image', "${HOME}", "Formats: (*.png )")

        if not self.filePath:  # Jeśli lista jest pusta, pokaż ostrzeżenie
            QMessageBox.warning(self.window, "No Mask Selected", "Please select Masks.")
            return

        for filePath in self.filePath:
            mask = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
            self.input_masks.append(mask)
            self.input_mask_path.append(filePath)

            self.loadMasksList.addItems(self.input_mask_path)
            self.maskNumber.setText(f"Masks: {len(self.input_masks)}")
            print(f"Masks: {len(self.input_masks)}")

    def getMaskFolder(self):

        # Open window to choose file
        self.folderPath = QFileDialog.getExistingDirectory(self.window, 'Choose a patient Directory', "${HOME}")

        if self.folderPath:

            for root, dirs, files in os.walk(self.folderPath):
                for fileName in files:
                    if fileName.lower().endswith(('.png')):
                        filePath = os.path.join(root, fileName)
                        maskF = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
                        # nameFolder = os.path.basename(self.folderPath)
                        self.input_masks.append(maskF)
                        self.input_mask_path.append(filePath)
                        print(f"Masks Folder: {len(self.input_masks)}")

            self.maskNumber.setText(f"Masks: {len(self.input_masks)}")
            self.loadMasksList.addItems(self.input_mask_path)
        else:
            QMessageBox.warning(self.window, "No Masks Selected", "Please select masks directory.")

    def getParamFile(self):
        self.input_param_file, _ = QFileDialog.getOpenFileName(self.window, 'Choose a parameter file', "${HOME}",
                                                               "Formats: (*.yaml)")

        if not self.input_param_file:
            QMessageBox.warning(self.window, "No Param File", "Please select file with parameters.")
            return

        messageParam = ""
        messageParam = "Succesfull"
        self.loadParamStatus.setText(messageParam)

    def reset(self):
        self.filePath = ''
        self.folderPath = ''
        self.input_images = []
        self.input_images_names = []
        self.input_images_paths = []
        self.folders_paths = []
        self.input_masks = []
        self.input_mask_path = []
        self.emptyMasks = []
        self.newMaskFolder = []
        self.input_param_file = ''
        self.imageNumber.clear()
        self.maskNumber.clear()
        self.loadDataList.clear()
        self.loadMasksList.clear()
        self.emptyMasksList.clear()
        self.statusMaskLabel.clear()
        self.csvStatusLabel.clear()
        self.loadParamStatus.clear()

    def showImages(self):
        if self.images:
            print("Number of images:", len(self.input_images))
            # e[0] - images, e[1] - images_paths
            showData([e[0] for e in self.input_images], [e[1] for e in self.input_images])
        else:
            print("No files selected.")

    def chooseSegmentation(self, input_masks):
        segmentation = Segmentation()
        message = ""
        new_mask_sitk = None
        newMasks = []

        if self.area1Button.isChecked():
            new_mask_sitk = segmentation.segmentationMask(input_masks, [1])
        elif self.area2Button.isChecked():
            new_mask_sitk = segmentation.segmentationMask(input_masks, [2])
        elif self.bothButton.isChecked():
            new_mask_sitk = segmentation.segmentationMask(input_masks, [1, 2])

        if new_mask_sitk is not None:
            newMasks.extend(new_mask_sitk)
            # print(f"New Masks: {len(self.newMasks)}")
            message = "Successful! New mask/masks created"
        else:
            QMessageBox.warning(self.window, "No Region Selected",
                                "No region selected for mask. Please select a region.")

        self.statusMaskLabel.setText(message)
        return newMasks

    def getROI(self, normImages, newMasks):
        roi = ROI()
        empty_masks = []
        empty_masks_images = []
        messageROI = ""

        roiImgsMasks = []

        for (image, image_path), mask in zip(normImages, newMasks):
            if np.all(mask == 0):
                empty_masks.append(mask)
                empty_masks_images.append(image_path)
            else:
                roiImg, roiMask = roi.roiImage(image, mask)
                roiImgsMasks.append((roiImg, image_path, roiMask))

        '''
        if not roiImgsMasks:
            messageROI = "No region of interest selected."
            QMessageBox.warning(self.window, "No ROI Selected", messageROI)
        else:
            messageROI = "Successful! ROI created"
            self.statusROILabel.setText(messageROI)
            self.emptyMasksList.addItems(empty_masks_images)
        '''

        return roiImgsMasks

    def chooseNormalization(self, input_images, numberBits, masks):
        no_normalization = NoNormalization()
        min_max_normalization = MinMaxNormalization()
        mean_std_normalization = MeanStdNormalization()
        percentile_normalization = PercentileNormalization()
        normalization_selected = False
        empty_masks = []
        empty_masks_images = []

        # self.numberBits.value()
        normImages = []

        for (image, image_path), mask in zip(input_images, masks):
            if np.all(mask == 0):
                empty_masks.append(mask)
                empty_masks_images.append(image_path)
            else:
                new_image = None
                if self.noNormaButton.isChecked():
                    new_image = no_normalization.normalize(image, mask, numberBits)
                    normalization_selected = True
                elif self.minMaxButton.isChecked():
                    new_image = min_max_normalization.normalize(image, numberBits, mask)
                    normalization_selected = True
                elif self.meanStdButton.isChecked():
                    new_image = mean_std_normalization.normalize(image, numberBits, mask)
                    normalization_selected = True
                elif self.perButton.isChecked():
                    new_image = percentile_normalization.normalize(image, numberBits, mask)
                    normalization_selected = True

                if new_image is not None:  # Dodanie tylko jeśli new_image nie jest None
                    normImages.append((new_image, image_path, mask))

        if not normalization_selected:
            message = "No normalization method selected."
            QMessageBox.warning(self.window, "No Normalization Selected", message)

        self.emptyMasksList.addItems(empty_masks_images)

        return normImages

    def radiomics(self, normImages, input_param_file):
        radiomics = Radiomics()
        messageRadiomic = ""

        base_path = os.path.abspath(os.path.join("..", ".."))
        self.output_path = os.path.join(base_path, "Results")
        os.makedirs(self.output_path, exist_ok=True)

        radiomics.extractRadiomics(normImages, input_param_file, self.output_path)
        messageRadiomic = "CSV files created "
        self.csvStatusLabel.setText(messageRadiomic)

    def generateCsvAll(self):
        newMasks = self.chooseSegmentation(self.input_masks)
        normImages = self.chooseNormalization(self.input_images, self.numberBits.value(), newMasks)

        if not self.input_param_file:
            QMessageBox.warning(self.window, "No Param File", "Please select file with parameters.")
        else:
            # roiImgsMasks = self.getROI(normImages, newMasks)
            self.radiomics(normImages, self.input_param_file)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Textural analysis"))
        self.noNormaButton.setText(_translate("MainWindow", "No normalization"))
        self.minMaxButton.setText(_translate("MainWindow", "Min-Max normalization"))
        self.meanStdButton.setText(_translate("MainWindow", "Mean/Std normalization"))
        self.perButton.setText(_translate("MainWindow", "Percentile normalization"))
        self.label_5.setText(_translate("MainWindow", "Segmentation"))
        self.bothButton.setText(_translate("MainWindow", "Both"))
        self.label_6.setText(_translate("MainWindow", "Select the region to analyze:"))
        self.area2Button.setText(_translate("MainWindow", "External region"))
        self.area1Button.setText(_translate("MainWindow", "Internal region"))
        self.label_8.setText(_translate("MainWindow", "Choose normalization and number bits to :"))
        self.label.setText(_translate("MainWindow", "Normalization"))
        self.label_2.setText(_translate("MainWindow", "Number of bits:"))
        self.csvButton.setText(_translate("MainWindow", "Generate CSV"))
        self.label_7.setText(_translate("MainWindow", "Empty Masks"))
        self.loadParamFile.setText(_translate("MainWindow", "Load Parameter File"))
        self.label_3.setText(_translate("MainWindow", "Data"))
        self.loadMaskFolderButton.setText(_translate("MainWindow", "Load Mask Folder"))
        self.loadDataButton.setText(_translate("MainWindow", "Load Data"))
        self.LoadFolderButton.setText(_translate("MainWindow", "Load Folder"))
        self.loadMaskButton.setText(_translate("MainWindow", "Load Mask"))
        self.label_4.setText(_translate("MainWindow", "Images"))
        self.label_9.setText(_translate("MainWindow", "Masks"))
        self.resetAppButton.setText(_translate("MainWindow", "Reset"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
