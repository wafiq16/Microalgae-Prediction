# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'mainwindow.ui'
##
# Created by: Qt User Interface Compiler version 6.2.2
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
                           QFont, QFontDatabase, QGradient, QIcon,
                           QImage, QKeySequence, QLinearGradient, QPainter,
                           QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QWidget)

import sys


class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1366, 768)
        self.widget = QWidget(Widget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(0, 0, 1366, 768))
        self.View = QPushButton(self.widget)
        self.View.setObjectName(u"View")
        self.View.setEnabled(True)
        self.View.setGeometry(QRect(20, 20, 90, 30))
        self.View.setCheckable(False)
        self.View.setChecked(False)
        self.View.setAutoDefault(False)
        self.View.setFlat(False)
        self.View_Cam_1_Tittle = QPushButton(self.widget)
        self.View_Cam_1_Tittle.setObjectName(u"View_Cam_1_Tittle")
        self.View_Cam_1_Tittle.setEnabled(True)
        self.View_Cam_1_Tittle.setGeometry(QRect(40, 60, 90, 30))
        self.View_Cam_1 = QLabel(self.widget)
        self.View_Cam_1.setObjectName(u"View_Cam_1")
        self.View_Cam_1.setGeometry(QRect(40, 100, 480, 320))
        self.View_Cam_2_Tittle = QPushButton(self.widget)
        self.View_Cam_2_Tittle.setObjectName(u"View_Cam_2_Tittle")
        self.View_Cam_2_Tittle.setGeometry(QRect(610, 60, 90, 30))
        self.View_Cam_2 = QLabel(self.widget)
        self.View_Cam_2.setObjectName(u"View_Cam_2")
        self.View_Cam_2.setGeometry(QRect(610, 100, 480, 320))
        self.Controller = QPushButton(self.widget)
        self.Controller.setObjectName(u"Controller")
        self.Controller.setGeometry(QRect(1130, 20, 140, 30))
        self.Save_Cam_1 = QPushButton(self.widget)
        self.Save_Cam_1.setObjectName(u"Save_Cam_1")
        self.Save_Cam_1.setGeometry(QRect(1130, 110, 220, 30))
        self.Save_Cam_2 = QPushButton(self.widget)
        self.Save_Cam_2.setObjectName(u"Save_Cam_2")
        self.Save_Cam_2.setGeometry(QRect(1130, 150, 220, 31))
        self.Save_To_Csv = QPushButton(self.widget)
        self.Save_To_Csv.setObjectName(u"Save_To_Csv")
        self.Save_To_Csv.setGeometry(QRect(1180, 490, 120, 31))
        self.RGB_Red = QLineEdit(self.widget)
        self.RGB_Red.setObjectName(u"RGB_Red")
        self.RGB_Red.setGeometry(QRect(1130, 230, 220, 30))
        self.RGB_Sensor = QPushButton(self.widget)
        self.RGB_Sensor.setObjectName(u"RGB_Sensor")
        self.RGB_Sensor.setGeometry(QRect(1130, 190, 220, 30))
        self.RGB_Green = QLineEdit(self.widget)
        self.RGB_Green.setObjectName(u"RGB_Green")
        self.RGB_Green.setGeometry(QRect(1130, 270, 220, 30))
        self.RGB_Blue = QLineEdit(self.widget)
        self.RGB_Blue.setObjectName(u"RGB_Blue")
        self.RGB_Blue.setGeometry(QRect(1130, 310, 220, 30))
        self.Class = QPushButton(self.widget)
        self.Class.setObjectName(u"Class")
        self.Class.setGeometry(QRect(1130, 350, 220, 30))
        self.Class_Type = QLineEdit(self.widget)
        self.Class_Type.setObjectName(u"Class_Type")
        self.Class_Type.setGeometry(QRect(1130, 390, 220, 30))
        self.Class_Concentration = QLineEdit(self.widget)
        self.Class_Concentration.setObjectName(u"Class_Concentration")
        self.Class_Concentration.setGeometry(QRect(1130, 430, 220, 30))
        self.Data = QPushButton(self.widget)
        self.Data.setObjectName(u"Data")
        self.Data.setGeometry(QRect(20, 450, 89, 31))

        self.Mean_Cam_1 = QPushButton(self.widget)
        self.Mean_Cam_1.setObjectName(u"Mean_Cam_1")
        self.Mean_Cam_1.setGeometry(QRect(60, 530, 90, 30))
        self.Mean_R_Cam_1 = QLineEdit(self.widget)
        self.Mean_R_Cam_1.setObjectName(u"Mean_R_Cam_1")
        self.Mean_R_Cam_1.setGeometry(QRect(160, 570, 113, 29))
        self.Mean_R_Cam_1.setReadOnly(True)
        self.pushButton_13 = QPushButton(self.widget)
        self.pushButton_13.setObjectName(u"pushButton_13")
        self.pushButton_13.setGeometry(QRect(60, 570, 90, 30))
        self.pushButton_14 = QPushButton(self.widget)
        self.pushButton_14.setObjectName(u"pushButton_14")
        self.pushButton_14.setGeometry(QRect(60, 610, 90, 30))
        self.Mean_G_Cam_1 = QLineEdit(self.widget)
        self.Mean_G_Cam_1.setObjectName(u"Mean_G_Cam_1")
        self.Mean_G_Cam_1.setGeometry(QRect(160, 610, 113, 29))
        self.Mean_G_Cam_1.setReadOnly(True)
        self.pushButton_15 = QPushButton(self.widget)
        self.pushButton_15.setObjectName(u"pushButton_15")
        self.pushButton_15.setGeometry(QRect(60, 650, 90, 30))
        self.Mean_B_Cam_1 = QLineEdit(self.widget)
        self.Mean_B_Cam_1.setObjectName(u"Mean_B_Cam_1")
        self.Mean_B_Cam_1.setGeometry(QRect(160, 650, 113, 29))
        self.Mean_B_Cam_1.setReadOnly(True)
        self.Max_Cam_1 = QPushButton(self.widget)
        self.Max_Cam_1.setObjectName(u"Max_Cam_1")
        self.Max_Cam_1.setGeometry(QRect(300, 530, 90, 30))
        self.Max_B_Cam_1 = QLineEdit(self.widget)
        self.Max_B_Cam_1.setObjectName(u"Max_B_Cam_1")
        self.Max_B_Cam_1.setGeometry(QRect(400, 650, 113, 29))
        self.Max_B_Cam_1.setReadOnly(True)
        self.pushButton_30 = QPushButton(self.widget)
        self.pushButton_30.setObjectName(u"pushButton_30")
        self.pushButton_30.setGeometry(QRect(300, 570, 90, 30))
        self.Max_R_Cam_1 = QLineEdit(self.widget)
        self.Max_R_Cam_1.setObjectName(u"Max_R_Cam_1")
        self.Max_R_Cam_1.setGeometry(QRect(400, 570, 113, 29))
        self.Max_R_Cam_1.setReadOnly(True)
        self.pushButton_31 = QPushButton(self.widget)
        self.pushButton_31.setObjectName(u"pushButton_31")
        self.pushButton_31.setGeometry(QRect(300, 610, 90, 30))
        self.Max_G_Cam_1 = QLineEdit(self.widget)
        self.Max_G_Cam_1.setObjectName(u"Max_G_Cam_1")
        self.Max_G_Cam_1.setGeometry(QRect(400, 610, 113, 29))
        self.Max_G_Cam_1.setReadOnly(True)
        self.pushButton_32 = QPushButton(self.widget)
        self.pushButton_32.setObjectName(u"pushButton_32")
        self.pushButton_32.setGeometry(QRect(300, 650, 90, 30))
        self.Camera_1_RGB = QPushButton(self.widget)
        self.Camera_1_RGB.setObjectName(u"Camera_1_RGB")
        self.Camera_1_RGB.setGeometry(QRect(40, 490, 90, 30))

        self.Mean_B_Cam_2 = QLineEdit(self.widget)
        self.Mean_B_Cam_2.setObjectName(u"Mean_B_Cam_2")
        self.Mean_B_Cam_2.setGeometry(QRect(650, 650, 113, 29))
        self.Mean_B_Cam_2.setReadOnly(True)
        self.pushButton_53 = QPushButton(self.widget)
        self.pushButton_53.setObjectName(u"pushButton_53")
        self.pushButton_53.setGeometry(QRect(550, 570, 90, 30))
        self.pushButton_54 = QPushButton(self.widget)
        self.pushButton_54.setObjectName(u"pushButton_54")
        self.pushButton_54.setGeometry(QRect(790, 610, 90, 30))
        self.Max_R_Cam_2 = QLineEdit(self.widget)
        self.Max_R_Cam_2.setObjectName(u"Max_R_Cam_2")
        self.Max_R_Cam_2.setGeometry(QRect(890, 570, 113, 29))
        self.Max_R_Cam_2.setReadOnly(True)
        self.Mean_G_Cam_2 = QLineEdit(self.widget)
        self.Mean_G_Cam_2.setObjectName(u"Mean_G_Cam_2")
        self.Mean_G_Cam_2.setGeometry(QRect(650, 610, 113, 29))
        self.Mean_G_Cam_2.setReadOnly(True)
        self.pushButton_55 = QPushButton(self.widget)
        self.pushButton_55.setObjectName(u"pushButton_55")
        self.pushButton_55.setGeometry(QRect(550, 650, 90, 30))
        self.Max_B_Cam_2 = QLineEdit(self.widget)
        self.Max_B_Cam_2.setObjectName(u"Max_B_Cam_2")
        self.Max_B_Cam_2.setGeometry(QRect(890, 650, 113, 29))
        self.Max_B_Cam_2.setReadOnly(True)
        self.Max_G_Cam_2 = QLineEdit(self.widget)
        self.Max_G_Cam_2.setObjectName(u"Max_G_Cam_2")
        self.Max_G_Cam_2.setGeometry(QRect(890, 610, 113, 29))
        self.Max_G_Cam_2.setReadOnly(True)
        self.pushButton_56 = QPushButton(self.widget)
        self.pushButton_56.setObjectName(u"pushButton_56")
        self.pushButton_56.setGeometry(QRect(790, 530, 90, 30))
        self.pushButton_57 = QPushButton(self.widget)
        self.pushButton_57.setObjectName(u"pushButton_57")
        self.pushButton_57.setGeometry(QRect(790, 650, 90, 30))
        self.Mean_Cam_2 = QPushButton(self.widget)
        self.Mean_Cam_2.setObjectName(u"Mean_Cam_2")
        self.Mean_Cam_2.setGeometry(QRect(550, 530, 90, 30))
        self.pushButton_59 = QPushButton(self.widget)
        self.pushButton_59.setObjectName(u"pushButton_59")
        self.pushButton_59.setGeometry(QRect(790, 570, 90, 30))
        self.Mean_R_Cam_2 = QLineEdit(self.widget)
        self.Mean_R_Cam_2.setObjectName(u"Mean_R_Cam_2")
        self.Mean_R_Cam_2.setGeometry(QRect(650, 570, 113, 29))
        self.Mean_R_Cam_2.setReadOnly(True)
        self.pushButton_60 = QPushButton(self.widget)
        self.pushButton_60.setObjectName(u"pushButton_60")
        self.pushButton_60.setGeometry(QRect(550, 610, 90, 30))
        self.Camera_2_RGB = QPushButton(self.widget)
        self.Camera_2_RGB.setObjectName(u"Camera_2_RGB")
        self.Camera_2_RGB.setGeometry(QRect(530, 490, 90, 30))

        self.Next_Data = QPushButton(self.widget)
        self.Next_Data.setObjectName(u"Next_Data")
        self.Next_Data.setGeometry(QRect(1260, 690, 89, 31))
        self.Train = QPushButton(self.widget)
        self.Train.setObjectName(u"Train")
        self.Train.setGeometry(QRect(1080, 590, 89, 50))
        self.Test = QPushButton(self.widget)
        self.Test.setObjectName(u"Test")
        self.Test.setGeometry(QRect(1180, 590, 89, 50))

        self.retranslateUi(Widget)

        self.View.setDefault(False)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(
            QCoreApplication.translate("Widget", u"Widget", None))
        self.View.setText(QCoreApplication.translate("Widget", u"View", None))
        self.View_Cam_1_Tittle.setText(
            QCoreApplication.translate("Widget", u"Camera 1", None))
        self.View_Cam_1.setText(
            QCoreApplication.translate("Widget", u"TextLabel", None))
        self.View_Cam_2_Tittle.setText(
            QCoreApplication.translate("Widget", u"Camera 2", None))
        self.View_Cam_2.setText(
            QCoreApplication.translate("Widget", u"TextLabel", None))
        self.Controller.setText(QCoreApplication.translate(
            "Widget", u"Controller", None))
        self.Save_Cam_1.setText(QCoreApplication.translate(
            "Widget", u"Save Camera 1", None))
        self.Save_Cam_2.setText(QCoreApplication.translate(
            "Widget", u"Save Camera 2", None))
        self.Save_To_Csv.setText(QCoreApplication.translate(
            "Widget", u"Save to CSV", None))
        self.RGB_Red.setPlaceholderText(
            QCoreApplication.translate("Widget", u"Red", None))
        self.RGB_Sensor.setText(QCoreApplication.translate(
            "Widget", u"RGB Sensor", None))
        self.RGB_Green.setPlaceholderText(
            QCoreApplication.translate("Widget", u"Green", None))
        self.RGB_Blue.setPlaceholderText(
            QCoreApplication.translate("Widget", u"Blue", None))
        self.Class.setText(QCoreApplication.translate(
            "Widget", u"Class", None))
        self.Class_Type.setPlaceholderText(
            QCoreApplication.translate("Widget", u"Type", None))
        self.Class_Concentration.setText("")
        self.Class_Concentration.setPlaceholderText(
            QCoreApplication.translate("Widget", u"Concentration", None))
        self.Data.setText(QCoreApplication.translate("Widget", u"Data", None))
        self.Mean_Cam_1.setText(
            QCoreApplication.translate("Widget", u"Mean RGB", None))
        self.pushButton_13.setText(
            QCoreApplication.translate("Widget", u"R", None))
        self.pushButton_14.setText(
            QCoreApplication.translate("Widget", u"G", None))
        self.pushButton_15.setText(
            QCoreApplication.translate("Widget", u"B", None))
        self.Max_Cam_1.setText(
            QCoreApplication.translate("Widget", u"Max RGB", None))
        self.pushButton_30.setText(
            QCoreApplication.translate("Widget", u"R", None))
        self.pushButton_31.setText(
            QCoreApplication.translate("Widget", u"G", None))
        self.pushButton_32.setText(
            QCoreApplication.translate("Widget", u"B", None))
        self.Camera_1_RGB.setText(
            QCoreApplication.translate("Widget", u"Camera 1", None))
        self.pushButton_53.setText(
            QCoreApplication.translate("Widget", u"R", None))
        self.pushButton_54.setText(
            QCoreApplication.translate("Widget", u"G", None))
        self.pushButton_55.setText(
            QCoreApplication.translate("Widget", u"B", None))
        self.pushButton_56.setText(
            QCoreApplication.translate("Widget", u"Max RGB", None))
        self.pushButton_57.setText(
            QCoreApplication.translate("Widget", u"B", None))
        self.Mean_Cam_2.setText(
            QCoreApplication.translate("Widget", u"Mean RGB", None))
        self.pushButton_59.setText(
            QCoreApplication.translate("Widget", u"R", None))
        self.pushButton_60.setText(
            QCoreApplication.translate("Widget", u"G", None))
        self.Camera_2_RGB.setText(
            QCoreApplication.translate("Widget", u"Camera 2", None))
        self.Next_Data.setText(QCoreApplication.translate(
            "Widget", u"Next Data", None))
        self.Train.setText(QCoreApplication.translate(
            "Widget", u"Train", None))
        self.Test.setText(QCoreApplication.translate("Widget", u"Test", None))
    # retranslateUi
