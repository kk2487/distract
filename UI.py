import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *

class App(QWidget):
    def __init__(self):
        super(App, self).__init__()
        self.title = 'Driver DISTRACTION'
        self.left = 25
        self.top = 25
        self.width = 1400
        self.height = 700
        self.initUI()
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        font = QtGui.QFont("Times", 30, QtGui.QFont.Bold) 
        self.label2 = QLabel("WARNING SYSTEM FOR DISTRACTED DRIVER",self)
        self.label2.move(150, 20)
        self.label2.setFont(font)
        
        self.button1 = QPushButton('Video Input', self)
        self.button1.setToolTip('Video Input')
        self.button1.move(300,100)
        self.button1.resize(150,40)
        self.button1.clicked.connect(self.video_input_fn)        
        
        self.button2 = QPushButton('Live Feed', self)
        self.button2.setToolTip('Camera Input')
        self.button2.move(500,100)
        self.button2.resize(150,40)
       
        self.button3 = QPushButton('Train Model', self)
        self.button3.setToolTip('Train new Model')
        self.button3.move(700,100)
        self.button3.resize(150,40)
        
        self.button4 = QPushButton('Select model', self)
        self.button4.setToolTip('Select model')
        self.button4.move(700,150)
        self.button4.resize(150,40)
        self.button4.clicked.connect(self.choose_model)        
        
        self.show()
    
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "./","Mp4 (*.mp4)", options=opt)
        return fileUrl
    def md_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getExistingDirectory(self,"Input Model", "./")
        return fileUrl
                                        
    @pyqtSlot()  
    def video_input_fn(self):
        mv_fileUrl = self.mv_Chooser()
        print(mv_fileUrl)

    @pyqtSlot()
    def live_input_fn(self):
        print("Live feed")

    @pyqtSlot()
    def choose_model(self): 
        md_fileUrl = self.md_Chooser()
        print(md_fileUrl)

if __name__ == '__main__':
    qt_env = QApplication(sys.argv)
    process = App()
    sys.exit(qt_env.exec_())
