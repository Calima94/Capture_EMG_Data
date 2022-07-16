# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)
        MainWindow.setMinimumSize(QtCore.QSize(600, 400))
        MainWindow.setMaximumSize(QtCore.QSize(600, 400))
        MainWindow.setStyleSheet("*{\n"
"border:none;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.left_container = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(100)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_container.sizePolicy().hasHeightForWidth())
        self.left_container.setSizePolicy(sizePolicy)
        self.left_container.setMinimumSize(QtCore.QSize(100, 0))
        self.left_container.setMaximumSize(QtCore.QSize(150, 16777215))
        self.left_container.setStyleSheet("")
        self.left_container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.left_container.setFrameShadow(QtWidgets.QFrame.Raised)
        self.left_container.setObjectName("left_container")
        self.gridLayout = QtWidgets.QGridLayout(self.left_container)
        self.gridLayout.setContentsMargins(0, -1, 0, 9)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_2 = QtWidgets.QFrame(self.left_container)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_2.setStyleSheet("QToolBox::tab{\n"
"    border-radius: 5px;\n"
"\n"
"    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"\n"
"QFrame{\n"
"    \n"
"    background-color: rgb(136, 138, 133);\n"
"    border-radius: 5px;\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.toolbox = QtWidgets.QToolBox(self.frame_2)
        self.toolbox.setStyleSheet("background-color: rgb(136, 138, 133);")
        self.toolbox.setObjectName("toolbox")
        self.Menu = QtWidgets.QWidget()
        self.Menu.setGeometry(QtCore.QRect(0, 0, 82, 315))
        self.Menu.setObjectName("Menu")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.Menu)
        self.verticalLayout_4.setContentsMargins(0, -1, 0, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btn_home = QtWidgets.QPushButton(self.Menu)
        self.btn_home.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_home.setStyleSheet("QPushButton:hover{\n"
"    border-top-left-radius:15px;\n"
"    border-top-right-radius:15px;\n"
"    border-bottom-left-radius:15px;\n"
"    border-bottom-right-radius:15px;\n"
"    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"\n"
"")
        self.btn_home.setAutoDefault(False)
        self.btn_home.setObjectName("btn_home")
        self.verticalLayout_4.addWidget(self.btn_home)
        self.btn_funcoes = QtWidgets.QPushButton(self.Menu)
        self.btn_funcoes.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_funcoes.setStyleSheet("QPushButton:hover{\n"
"    border-top-left-radius:15px;\n"
"    border-top-right-radius:15px;\n"
"    border-bottom-left-radius:15px;\n"
"    border-bottom-right-radius:15px;\n"
"    \n"
"    background-color: rgb(211, 215, 207);\n"
"}")
        self.btn_funcoes.setObjectName("btn_funcoes")
        self.verticalLayout_4.addWidget(self.btn_funcoes)
        self.btn_contatos = QtWidgets.QPushButton(self.Menu)
        self.btn_contatos.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_contatos.setStyleSheet("QPushButton:hover{\n"
"    border-top-left-radius:15px;\n"
"    border-top-right-radius:15px;\n"
"    border-bottom-left-radius:15px;\n"
"    border-bottom-right-radius:15px;\n"
"    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"")
        self.btn_contatos.setObjectName("btn_contatos")
        self.verticalLayout_4.addWidget(self.btn_contatos)
        self.btn_sobre = QtWidgets.QPushButton(self.Menu)
        self.btn_sobre.setMinimumSize(QtCore.QSize(0, 30))
        self.btn_sobre.setStyleSheet("QPushButton:hover{\n"
"    border-top-left-radius:15px;\n"
"    border-top-right-radius:15px;\n"
"    border-bottom-left-radius:15px;\n"
"    border-bottom-right-radius:15px;\n"
"    \n"
"    background-color: rgb(211, 215, 207);\n"
"}")
        self.btn_sobre.setObjectName("btn_sobre")
        self.verticalLayout_4.addWidget(self.btn_sobre)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.toolbox.addItem(self.Menu, "")
        self.horizontalLayout_2.addWidget(self.toolbox)
        self.gridLayout.addWidget(self.frame_2, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.left_container)
        self.main_container = QtWidgets.QFrame(self.centralwidget)
        self.main_container.setStyleSheet("QFrame{\n"
"    \n"
"    border-radius: 5px;\n"
"}\n"
"")
        self.main_container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_container.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_container.setObjectName("main_container")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.main_container)
        self.verticalLayout.setContentsMargins(0, -1, 0, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.top = QtWidgets.QFrame(self.main_container)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.top.sizePolicy().hasHeightForWidth())
        self.top.setSizePolicy(sizePolicy)
        self.top.setMinimumSize(QtCore.QSize(0, 30))
        self.top.setStyleSheet("QFrame{\n"
"    \n"
"    background-color: rgb(136, 138, 133);\n"
"    border-radius: 5px;\n"
"}")
        self.top.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.top.setFrameShadow(QtWidgets.QFrame.Raised)
        self.top.setObjectName("top")
        self.label = QtWidgets.QLabel(self.top)
        self.label.setGeometry(QtCore.QRect(50, 10, 331, 16))
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.top)
        self.main_center = QtWidgets.QFrame(self.main_container)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.main_center.sizePolicy().hasHeightForWidth())
        self.main_center.setSizePolicy(sizePolicy)
        self.main_center.setStyleSheet("QFrame{\n"
"    \n"
"    background-color: rgb(136, 138, 133);\n"
"    border-radius: 5px;\n"
"}")
        self.main_center.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_center.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_center.setObjectName("main_center")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.main_center)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.Pages = QtWidgets.QStackedWidget(self.main_center)
        self.Pages.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Pages.setAutoFillBackground(False)
        self.Pages.setObjectName("Pages")
        self.page_home = QtWidgets.QWidget()
        self.page_home.setObjectName("page_home")
        self.label_3 = QtWidgets.QLabel(self.page_home)
        self.label_3.setGeometry(QtCore.QRect(110, 240, 231, 20))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.label_3.setStyleSheet("background-position: center;")
        self.label_3.setObjectName("label_3")
        self.frame_6 = QtWidgets.QFrame(self.page_home)
        self.frame_6.setEnabled(True)
        self.frame_6.setGeometry(QtCore.QRect(110, 0, 223, 228))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMaximumSize(QtCore.QSize(223, 228))
        self.frame_6.setStyleSheet("background-image: url(:/logo_imagem/ufabc.png);\n"
"background-repeat: no-repeat;\n"
"background-position: center;")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.Pages.addWidget(self.page_home)
        self.page_funcoes = QtWidgets.QWidget()
        self.page_funcoes.setObjectName("page_funcoes")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.page_funcoes)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.frame_main = QtWidgets.QFrame(self.page_funcoes)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_main.sizePolicy().hasHeightForWidth())
        self.frame_main.setSizePolicy(sizePolicy)
        self.frame_main.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_main.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_main.setObjectName("frame_main")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame_main)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_3 = QtWidgets.QFrame(self.frame_main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMinimumSize(QtCore.QSize(0, 30))
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 30))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        self.label_2.setGeometry(QtCore.QRect(140, 0, 91, 17))
        self.label_2.setObjectName("label_2")
        self.verticalLayout_9.addWidget(self.frame_3)
        self.frame = QtWidgets.QFrame(self.frame_main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.comboBox_braco = QtWidgets.QComboBox(self.frame)
        self.comboBox_braco.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-top-left-radius:5px;\n"
"border-top-right-radius:5px;\n"
"border-bottom-left-radius:5px;\n"
"border-bottom-right-radius:5px;\n"
"    \n"
"background-color: rgb(211, 215, 207);")
        self.comboBox_braco.setObjectName("comboBox_braco")
        self.comboBox_braco.addItem("")
        self.comboBox_braco.addItem("")
        self.comboBox_braco.addItem("")
        self.verticalLayout_7.addWidget(self.comboBox_braco)
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_7.addWidget(self.label_6)
        self.lineEdit_amostra = QtWidgets.QLineEdit(self.frame)
        self.lineEdit_amostra.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-top-left-radius:5px;\n"
"border-top-right-radius:5px;\n"
"border-bottom-left-radius:5px;\n"
"border-bottom-right-radius:5px;\n"
"    \n"
"background-color: rgb(211, 215, 207);")
        self.lineEdit_amostra.setObjectName("lineEdit_amostra")
        self.verticalLayout_7.addWidget(self.lineEdit_amostra)
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_7.addWidget(self.label_7)
        self.lineEdit_var = QtWidgets.QLineEdit(self.frame)
        self.lineEdit_var.setStyleSheet("font: \'Quantidade de Amostra\'11pt \"Ubuntu\";\n"
"color: rgb(0, 0, 0);\n"
"border-top-left-radius:5px;\n"
"border-top-right-radius:5px;\n"
"border-bottom-left-radius:5px;\n"
"border-bottom-right-radius:5px;\n"
"    \n"
"background-color: rgb(211, 215, 207);")
        self.lineEdit_var.setObjectName("lineEdit_var")
        self.verticalLayout_7.addWidget(self.lineEdit_var)
        self.comboBox_classe = QtWidgets.QComboBox(self.frame)
        self.comboBox_classe.setStyleSheet("color: rgb(0, 0, 0);\n"
"border-top-left-radius:5px;\n"
"border-top-right-radius:5px;\n"
"border-bottom-left-radius:5px;\n"
"border-bottom-right-radius:5px;\n"
"    \n"
"background-color: rgb(211, 215, 207);")
        self.comboBox_classe.setObjectName("comboBox_classe")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.comboBox_classe.addItem("")
        self.verticalLayout_7.addWidget(self.comboBox_classe)
        self.start = QtWidgets.QPushButton(self.frame)
        self.start.setStyleSheet("border-top-left-radius:5px;\n"
"border-top-right-radius:5px;\n"
"border-bottom-left-radius:5px;\n"
"border-bottom-right-radius:5px;\n"
"    \n"
"background-color: rgb(255, 255, 255);")
        self.start.setObjectName("start")
        self.verticalLayout_7.addWidget(self.start)
        self.verticalLayout_9.addWidget(self.frame)
        self.verticalLayout_8.addWidget(self.frame_main)
        self.Pages.addWidget(self.page_funcoes)
        self.page_contatos = QtWidgets.QWidget()
        self.page_contatos.setObjectName("page_contatos")
        self.label_12 = QtWidgets.QLabel(self.page_contatos)
        self.label_12.setGeometry(QtCore.QRect(20, 10, 31, 31))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.page_contatos)
        self.label_13.setGeometry(QtCore.QRect(20, 40, 241, 17))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.page_contatos)
        self.label_14.setGeometry(QtCore.QRect(20, 70, 91, 17))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.page_contatos)
        self.label_15.setGeometry(QtCore.QRect(20, 90, 361, 17))
        self.label_15.setObjectName("label_15")
        self.Pages.addWidget(self.page_contatos)
        self.page_sobre = QtWidgets.QWidget()
        self.page_sobre.setObjectName("page_sobre")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.page_sobre)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_16 = QtWidgets.QLabel(self.page_sobre)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_6.addWidget(self.label_16)
        self.label_4 = QtWidgets.QLabel(self.page_sobre)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.Pages.addWidget(self.page_sobre)
        self.verticalLayout_5.addWidget(self.Pages)
        self.verticalLayout.addWidget(self.main_center)
        self.baseboard = QtWidgets.QFrame(self.main_container)
        self.baseboard.setMinimumSize(QtCore.QSize(0, 35))
        self.baseboard.setStyleSheet("QFrame{\n"
"\n"
"    background-color: rgb(136, 138, 133);\n"
"    border-radius: 5px;\n"
"}\n"
"")
        self.baseboard.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.baseboard.setFrameShadow(QtWidgets.QFrame.Raised)
        self.baseboard.setObjectName("baseboard")
        self.base = QtWidgets.QLabel(self.baseboard)
        self.base.setGeometry(QtCore.QRect(0, 0, 61, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.base.setFont(font)
        self.base.setObjectName("base")
        self.verticalLayout.addWidget(self.baseboard)
        self.horizontalLayout.addWidget(self.main_container)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.toolbox.setCurrentIndex(0)
        self.Pages.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_home.setText(_translate("MainWindow", "Home"))
        self.btn_funcoes.setText(_translate("MainWindow", "Funções"))
        self.btn_contatos.setText(_translate("MainWindow", "Contatos"))
        self.btn_sobre.setText(_translate("MainWindow", "Sobre"))
        self.toolbox.setItemText(self.toolbox.indexOf(self.Menu), _translate("MainWindow", "Menu"))
        self.label.setText(_translate("MainWindow", "Sistema de Captura de Posições do Braço"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Projeto de Iniciação Cientifica</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Parâmetros"))
        self.comboBox_braco.setItemText(0, _translate("MainWindow", "Braço"))
        self.comboBox_braco.setItemText(1, _translate("MainWindow", "Esquerdo"))
        self.comboBox_braco.setItemText(2, _translate("MainWindow", "Direito"))
        self.label_6.setText(_translate("MainWindow", "Quantidade de Amostra"))
        self.label_7.setText(_translate("MainWindow", "Variância"))
        self.comboBox_classe.setItemText(0, _translate("MainWindow", "Selecione a Quantidade de Classe"))
        self.comboBox_classe.setItemText(1, _translate("MainWindow", "2"))
        self.comboBox_classe.setItemText(2, _translate("MainWindow", "3"))
        self.comboBox_classe.setItemText(3, _translate("MainWindow", "4"))
        self.comboBox_classe.setItemText(4, _translate("MainWindow", "5"))
        self.comboBox_classe.setItemText(5, _translate("MainWindow", "6"))
        self.comboBox_classe.setItemText(6, _translate("MainWindow", "7"))
        self.comboBox_classe.setItemText(7, _translate("MainWindow", "8"))
        self.comboBox_classe.setItemText(8, _translate("MainWindow", "9"))
        self.comboBox_classe.setItemText(9, _translate("MainWindow", "10"))
        self.start.setText(_translate("MainWindow", "Start"))
        self.label_12.setText(_translate("MainWindow", "Git"))
        self.label_13.setText(_translate("MainWindow", "link:  https://github.com/alans96"))
        self.label_14.setText(_translate("MainWindow", "Linkedin"))
        self.label_15.setText(_translate("MainWindow", "https://www.linkedin.com/in/alan-silva-061048236/"))
        self.label_16.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Sobre</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "  O desenvolvimento deste aplicativo,  tem como foco  no  rastrea\n"
"mento e identificao de pontos de  interesse,  relativos  ao  braço \n"
"de uma pessoa em  que  se  captura a  movimentação,  utilizando \n"
"uma   webcam   comum,   conjuntamente   com   ferramentas   da \n"
"biblioteca OpenCV  que  permite   relizar   o   processamento   da \n"
"imagem    e    paralelamente    realizandoa    captura     de     sinais \n"
"EMG(Eletromiográficos) com o dispositivo Myo - armband . \n"
"   A  identificação  dos  pontos   de   interesse   foi   realizada   pelo \n"
"framework open-source MediaPipe e a criação da interface gráfica \n"
"foi feita utilizando-se da biblioteca PyQt5."))
        self.base.setText(_translate("MainWindow", "PyArmy"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

