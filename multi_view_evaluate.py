import sys, random, matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5 as Qt

import pyqtgraph as pg
import numpy as np

import argparse

import ml_overfit

class SecondWindow(QtWidgets.QWidget):
    def __init__(self, data, autoscale=False, parent=None):
        super(SecondWindow, self).__init__(parent)
        self.data = data
        self.autoscale = autoscale
        self.setupUi(self)
        self.setFocus()

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 600)



        self.setLayout(QtWidgets.QGridLayout())
        # self.layout().setFocusPolicy(0)

        self.widges = [[],[],[]]

        for i in range(len(self.widges)):
            for y in range(2):
                w1 = pg.ImageView(self)
                # w1.mousePressEvent = self.printCursor
                self.widges[i].append(w1)
                self.layout().addWidget(w1,i,y)
            # self.layout().addWidget(la)

        # gradients for temp = ‘thermal’, ‘flame’, ‘yellowy’, ‘bipolar’, ‘spectrum’, ‘cyclic’, ‘greyclip’, ‘grey’

        #Set temperature graph style
        for y in range(2):
            self.widges[0][y].setPredefinedGradient('thermal')

        self.showFigures()

        self.t = 0
        self.paused = False
        self.ended  = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.draw)
        self.timer.start(16)

    def mousePressEvent(self,event):
        self.setFocus()

    def tbounds(self):
        if self.t >= self.data.data[0][0].shape[2]:
            self.t = self.data.data[0][0].shape[2] - 1
            self.ended  = True
            self.paused = True
        elif self.t < 0:
            self.t = 0

    def draw(self):
        # if not self.ended:
        if not self.paused:
            self.t += 1
            self.tbounds()
        for i in range(len(self.widges)):
            for x in range(2):
                if self.autoscale:
                    self.widges[i][x].setImage(self.data.data[i][x][self.t])
                else:
                    min = self.data.min[i]
                    max = self.data.max[i]
                    self.widges[i][x].setImage(self.data.data[i][x][self.t],levels=(min,max))

    def keyPressEvent(self, event):
        self.setFocus()
        if event.key() == QtCore.Qt.Key_Q \
        or event.key() == QtCore.Qt.Key_Escape:
            self.closeEvent()
            # self.deleteLater()
        elif event.key() == QtCore.Qt.Key_Backspace:
            self.ended=False
            self.t=0
        elif event.key() == QtCore.Qt.Key_Space:
            self.paused = not self.paused
            if self.ended:
                self.ended=False
                self.t=0
        elif event.key() == QtCore.Qt.Key_Right:
            self.t+=1
            self.tbounds()
        elif event.key() == QtCore.Qt.Key_Left:
            self.t-=1
            self.tbounds()

        else:
            print (event.key())
        # event.accept()

    # def focusChanged(self,event):
    #     print ('focus changed')
    #     print (event)

    def showFigures(self):
        for i in range(len(self.widges)):
            for x in range(2):
                min = self.data.min[i]
                max = self.data.max[i]
                self.widges[i][x].setImage(self.data.data[i][x][0],levels=(min,max))
                self.widges[i][x].show()

    def closeEvent(self, event=None):
        self.timer.stop()

class DataLoader():
    def __init__(self):

        self.batch, self.x_amt,self.y_amt, time_scale = ml_overfit.load_and_norm_data()

        self.num_c = ml_overfit.get_num_channels()

        print ("Creating model")
        self.model = ml_overfit.create_model(self.x_amt,self.y_amt, self.num_c)
        #lkfjds
        print ('Loading weights')
        self.model.load_weights('weights')
        print ('Loaded')

    def predict(self,use_iter=False):
        #Iterative calculation - predict each from previous step of prediction
        if use_iter:
            start_point = 0
            num_pred = self.batch.shape[0]

            predictions = np.empty(shape=(self.batch.shape))
            predictions[:start_point]=self.batch[:start_point]

            pred = self.batch[start_point][None]

            for i in range(start_point,num_pred):
                pred = self.model.predict(pred)
                predictions[i]=pred[0]

        #Sequential prediction - predict each from previous steps labels
        else:
            predictions = self.model.predict(self.batch)

        # del self.model
        self.predictions = np.roll(predictions,-1,axis=0)

    def scale(self):
        self.data=[ [0,0] for _ in range(3)]

        self.data[0][0] = self.predictions[:,:,:,0]
        self.data[0][1] = self.batch[:,:,:,0]

        self.data[1][0] = self.predictions[:,:,:,1:3]
        self.data[1][1] = self.batch[:,:,:,1:3]

        self.data[2][0] = self.predictions[:,:,:,3:5]
        self.data[2][1] = self.batch[:,:,:,3:5]

        fill_val = 0
        for i in range(1,3):
            for y in [0,1]:
                # print ('--',i,y,'---')
                # print (self.data[i][y])
                fill_val = np.mean(self.data[i][y])
                self.data[i][y] = np.pad(self.data[i][y],((0,0),(0,0),(0,0),(0,1)),constant_values=fill_val)

        self.min = [100 for _ in range(3)]
        self.max = [-100 for _ in range(3)]

        for i in range(3):
            for x in range(2):
                if self.data[i][x].max() > self.max[i]:
                    self.max[i] = self.data[i][x].max()
                if self.data[i][x].min() < self.min[i]:
                    self.min[i] =self.data[i][x].min()


        del self.predictions
        del self.batch

if __name__ == '__main__':

    #Let QT take arguments it recognizes first
    app = QtWidgets.QApplication(sys.argv)

    parser = argparse.ArgumentParser()

    parser.add_argument('--seq',action='store_true',help='Run model predictions on batch steps')
    parser.add_argument('--iter',action='store_true',help='Run model predictions on own output')
    parser.add_argument('--scale',action='store_true',help='Autoscales displays to their current value')

    args = parser.parse_args()

    model = DataLoader()

    model.predict(not args.seq and args.iter)

    model.scale()

    form = SecondWindow(model,args.scale)
    form.show()
    sys.exit(app.exec_())
