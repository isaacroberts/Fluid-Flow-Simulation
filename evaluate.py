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

        self.widges = []

        for i in range(self.data.num_c):
            self.widges.append([])
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

    def draw(self):
        # if not self.ended:
        pass

    def pause_or_play(self):
        PLAY_RATE = 20

        for w_l in self.widges:
            for w in w_l:
                if w.playRate==0:
                    w.play(PLAY_RATE)
                else:
                    w.play(0)
        self.paused = not self.paused

    def keyPressEvent(self, event):
        self.setFocus()
        if event.key() == QtCore.Qt.Key_Q \
        or event.key() == QtCore.Qt.Key_Escape:
            exit(0)
            # self.closeEvent()
            # self.deleteLater()
        elif event.key() == QtCore.Qt.Key_Backspace:
            for w_l in self.widges:
                for w in w_l:
                    w.setCurrentIndex(0)

        elif event.key() == QtCore.Qt.Key_Space:
            self.pause_or_play()
        elif event.key() == QtCore.Qt.Key_Right \
          or event.key() == QtCore.Qt.Key_Left:
            skip = 1 * (1 if event.key() == QtCore.Qt.Key_Right else -1)
            for w_l in self.widges:
                for w in w_l:
                    w.jumpFrames(skip)

        else:
            print (event.key())
        # event.accept()

    # def focusChanged(self,event):
    #     print ('focus changed')
    #     print (event)

    def showFigures(self):
        for i in range(len(self.widges)):
            for x in range(2):
                if self.autoscale:
                    self.widges[i][x].setImage(self.data.data[i][x])
                    self.widges[i][x].show()
                else:
                    min = self.data.min[i]
                    max = self.data.max[i]
                    self.widges[i][x].setImage(self.data.data[i][x],levels=(min,max))
                    self.widges[i][x].show()

    def closeEvent(self, event=None):
        self.timer.stop()

class DataLoader():
    def __init__(self):

        self.batch, self.x_amt,self.y_amt, time_scale = ml_overfit.load_and_norm_data()

        self.num_w = ml_overfit.get_num_channels()
        self.num_c = 4

        print ("Creating model")
        self.model = ml_overfit.create_model(self.x_amt,self.y_amt, self.num_w)
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

            prev = self.batch[start_point][None]

            for i in range(start_point,num_pred):
                pred = self.model.predict(prev)

                # T(t) = T(t-1) + d_T(t)
                pred[:,:,:,0] = prev[:,:,:,0] + pred[:,:,:,1]
                # V(t) = V(t-1) + a(t)
                pred[:,:,:,2:4] =prev[:,:,:,2:4] + pred[:,:,:,4:6]

                predictions[i]=pred[0]

                prev = pred

        #Sequential prediction - predict each from previous steps labels
        else:
            predictions = self.model.predict(self.batch)

        # del self.model
        self.predictions = np.roll(predictions,-1,axis=0)
        self.predictions = self.predictions[:-1]
        self.batch = self.batch[1:]


    def scale(self):
        self.data=[ [0,0] for _ in range(self.num_c)]

        self.data[0][0] = self.predictions[:,:,:,0]
        self.data[0][1] = self.batch[:,:,:,0]

        self.data[1][0] = self.predictions[:,:,:,1]
        self.data[1][1] = self.batch[:,:,:,1]

        self.data[2][0] = self.predictions[:,:,:,2:4]
        self.data[2][1] = self.batch[:,:,:,2:4]

        self.data[3][0] = self.predictions[:,:,:,4:6]
        self.data[3][1] = self.batch[:,:,:,4:6]

        fill_val = 0
        for i in range(2,4):
            for y in [0,1]:
                # print ('--',i,y,'---')
                # print (self.data[i][y])
                fill_val = np.mean(self.data[i][y])
                self.data[i][y] = np.pad(self.data[i][y],((0,0),(0,0),(0,0),(0,1)),constant_values=fill_val)

        self.min = [100 for _ in range(self.num_c)]
        self.max = [-100 for _ in range(self.num_c)]

        for i in range(self.num_c):
            x=1
            _min = self.data[i][x].min()
            _max = self.data[i][x].max()

            if _min < self.min[i]:
                self.min[i] = _min
            if  _max > self.max[i]:
                self.max[i] = _max


        del self.predictions
        del self.batch

if __name__ == '__main__':

    #Let QT take arguments it recognizes first
    app = QtWidgets.QApplication(sys.argv)

    parser = argparse.ArgumentParser()

    parser.add_argument('--seq',action='store_true',help='Run model predictions on batch steps')
    parser.add_argument('--iter',action='store_true',help='Run model predictions on own output')
    parser.add_argument('--loose',action='store_true',help='Scales per graph instead of per channel')

    args = parser.parse_args()

    model = DataLoader()

    model.predict(not args.seq and args.iter)

    model.scale()

    form = SecondWindow(model,args.loose)
    form.show()
    sys.exit(app.exec_())
