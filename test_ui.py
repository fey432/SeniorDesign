#region imports
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QSize, QObject, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import threading, schedule, time, sys, os, cv2, random
from multiprocessing import Process
from pynput import keyboard
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
import time
#endregion

#region ====================Global Variables ========================
predict_Grouping = -1
suggested_app_2_value = -1
status_bar_color = "background-color: rgb(85, 142, 72);"

time_right_now = time.strftime("%H:%M", time.localtime())
name_return = "unknown"
auth_check = True
iterations = 0

temp_mean = 72
temp_deviation = 0.1
current_temp = 72
min_temp = current_temp
max_temp = current_temp
avg_temp = current_temp

co2_mean = 0.25
co2_deviation = 0.01
current_co2 = 0
min_co2 = current_co2
max_co2 = current_co2
avg_co2 = current_co2

ppm_mean = 25
ppm_deviation = 0.5
current_ppm = 25
min_ppm = current_ppm
max_ppm = current_ppm
avg_ppm = current_ppm

oxy_press_mean = 3.2
oxy_press_deviation = 0.025
current_oxy_press = 3.2
min_oxy_press = current_oxy_press
max_oxy_press = current_oxy_press
avg_oxy_press = current_oxy_press

nit_press_mean = 11.5
nit_press_deviation = 0.025
current_nit_press = 11.5
min_nit_press = current_nit_press
max_nit_press = current_nit_press
avg_nit_press = current_nit_press

stress_mean = 5
stress_deviation = 0.025
current_stress = 5
min_stress = current_stress
max_stress = current_stress
avg_stress = current_stress

#endregion
#=====================Misc Functions =========================
def calcAvg_nonZero(input_array):
    input_sum = 0
    count = 0
    for i in range (61):
        if(not(input_array[i]) == 0):
            input_sum = input_sum + input_array[i]
            count = count + 1

    return input_sum/count

def rng_temp_module():
    global current_temp
    global temp_mean
    global temp_deviation
    current_temp = round(random.normalvariate(temp_mean,temp_deviation),2)

schedule.every().seconds.do(rng_temp_module)

def increase_temp_module():
    global temp_mean
    global temp_deviation

    while(temp_mean < 102):
        temp_mean = temp_mean + random.normalvariate(0.6, 0.1)
        time.sleep(0.5)
def decrease_temp_module():
    global temp_mean
    global temp_deviation

    while(temp_mean > 32):
        temp_mean = temp_mean - random.normalvariate(0.2, 0.1)
        time.sleep(0.5)
def return_to_normal_temp_module():
    global temp_mean
    global temp_deviation

    while(temp_mean > 73 or temp_mean < 71):
        if(temp_mean > 73):
            temp_mean = temp_mean - abs(random.normalvariate(1.5, 0.1))
            time.sleep(0.5)
        elif(temp_mean < 71):
            temp_mean = temp_mean + abs(random.normalvariate(1.5, 0.1))
            time.sleep(0.5)
        else:
            break

def rng_co2_module():
    global current_co2
    global co2_mean
    global co2_deviation
    current_co2 = round(random.normalvariate(co2_mean, co2_deviation),2)
def increase_co2_module():
    global co2_mean
    global co2_deviation

    while(co2_mean < 10):
        co2_mean = co2_mean + random.normalvariate(0.2, 0.25)
        time.sleep(0.5)
def return_to_normal_co2_module():
    global co2_mean
    global co2_deviation

    while(co2_mean > 0.25):
        co2_mean = abs(co2_mean - abs(random.normalvariate(0.2, 0.05)))
        time.sleep(0.5)
schedule.every().seconds.do(rng_co2_module)

def rng_ppm_module():
    global current_ppm
    global ppm_mean
    global ppm_deviation
    current_ppm = round(random.normalvariate(ppm_mean, ppm_deviation),2)
def increase_ppm_module():
    global ppm_mean
    global ppm_deviation

    while(ppm_mean < 450):
        ppm_mean = ppm_mean + abs(random.normalvariate(8, 0.5))
        time.sleep(0.5)
def return_to_normal_ppm_module():
    global ppm_mean
    global ppm_deviation

    while(ppm_mean > 26):
        ppm_mean = ppm_mean - abs(random.normalvariate(5, 0.25))
        time.sleep(0.5)
schedule.every().seconds.do(rng_ppm_module)

def rng_oxy_press():
    global current_oxy_press
    global oxy_press_mean
    global oxy_press_deviation
    current_oxy_press = round(random.normalvariate(oxy_press_mean, oxy_press_deviation),2)
def increase_oxy_press_module():
    global oxy_press_mean
    global oxy_press_deviation

    while(oxy_press_mean < 20):
        oxy_press_mean = oxy_press_mean + random.normalvariate(0.2, 0.05)
        time.sleep(0.5)

def decrease_oxy_press_module():
    global oxy_press_mean
    global oxy_press_deviation

    while(oxy_press_mean > 1):
        oxy_press_mean = abs(oxy_press_mean - random.normalvariate(0.05, 0.05))
        time.sleep(0.5)
def return_to_normal_oxy_press_module():
    global oxy_press_mean
    global oxy_press_deviation

    while(oxy_press_mean > 3.45 or oxy_press_mean < 2.95):
        if(oxy_press_mean > 3.45):
            oxy_press_mean = oxy_press_mean - abs(random.normalvariate(0.1, 0.05))
            time.sleep(0.5)
        elif(oxy_press_mean < 2.95):
            oxy_press_mean = oxy_press_mean + abs(random.normalvariate(0.1, 0.05))
            time.sleep(0.5)
        else:
            break
schedule.every().seconds.do(rng_oxy_press)

def rng_nit_press():
    global current_nit_press
    global nit_press_mean
    global nit_press_deviation
    current_nit_press = round(random.normalvariate(nit_press_mean, nit_press_deviation),2)
def increase_nit_press_module():
    global nit_press_mean
    global nit_press_deviation

    while(nit_press_mean < 30):
        nit_press_mean = nit_press_mean + random.normalvariate(0.05, 0.05)
        time.sleep(0.5)
def decrease_nit_press_module():
    global nit_press_mean
    global nit_press_deviation

    while(nit_press_mean > 8):
        nit_press_mean = abs(nit_press_mean - random.normalvariate(0.05, 0.05))
        time.sleep(0.5)
def return_to_normal_nit_press_module():
    global nit_press_mean
    global nit_press_deviation

    while(nit_press_mean > 11.75 or nit_press_mean < 11.25):
        if(nit_press_mean > 11.75):
            nit_press_mean = nit_press_mean - abs(random.normalvariate(0.1, 0.05))
            time.sleep(0.5)
        elif(nit_press_mean < 11.25):
            nit_press_mean = nit_press_mean + abs(random.normalvariate(0.1, 0.05))
            time.sleep(0.5)
        else:
            break

schedule.every().seconds.do(rng_nit_press)


def rng_stress():
    global current_stress
    global stress_mean
    global stress_deviation
    current_stress = round(random.normalvariate(stress_mean, stress_deviation),2)
def increase_stress_module():
    global stress_mean
    global stress_deviation

    while(stress_mean < 98):
        stress_mean = stress_mean + abs(random.normalvariate(1, 0.75))
        time.sleep(0.5)
def return_to_normal_stress_module():
    global stress_mean
    global stress_deviation

    while(stress_mean > 2):
        stress_mean = abs(stress_mean - abs(random.normalvariate(1, 1)))
        time.sleep(0.5)
schedule.every().seconds.do(rng_stress)

#region==================Machine Learning Prediction============
def train_DT(features, label):
    clf = DecisionTreeClassifier()
    clf.fit(features, label)

    return clf

def predict_Module(clf):
    global current_temp, current_co2, current_ppm, current_stress, current_oxy_press, current_nit_press
    global predict_Grouping
    predict_Grouping = clf.predict([[current_temp, current_co2, current_ppm, current_stress, current_oxy_press, current_nit_press]])
    print(predict_Grouping)
#endregion
#=====================Scheduler Tools=========================
def run_continuously(interval=1):
    """Continuously run, while executing pending jobs at each
    elapsed time interval.
    @return cease_continuous_run: threading. Event which can
    be set to cease continuous run. Please note that it is
    *intended behavior that run_continuously() does not run
    missed jobs*. For example, if you've registered a job that
    should run every minute and you set a continuous run
    interval of one hour then your job won't be run 60 times
    at each interval but only once.
    """
    cease_continuous_run = threading.Event()

    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run.is_set():
                schedule.run_pending()
                time.sleep(interval)

    continuous_thread = ScheduleThread()
    continuous_thread.start()
    return cease_continuous_run
#region ===================== Threading Functions ===================
#Status Bar Stuff
def StatusBarClock():
    global time_right_now
    t = time.localtime()
    time_right_now = time.strftime("%H:%M",t)

def StatusBarSignal():
    global Signal_Strength_Flag
    global Signal_Strength
    print('Signal is now ' + str(1))#Signal_Strength))
#endregion
#================== Face Recognizer Threading =======================  
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.App = App

    def run(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('C:/Users/Raymond Fey/Desktop/MHIS_PY/script/trainer/trainer.yml')
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX

        #iniciate id counter
        id = 0

        # names related to ids: example ==> Marcelo: id=1,  etc
        #TODO The names list needs to be correct before running
        names = ['Example','Raymond'] 


        # capture from web cam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1920) # set video widht
        cap.set(4, 1080) # set video height
        # Define min window size to be recognized as a face
        minW = 0.1*cap.get(3)
        minH = 0.1*cap.get(4)

        t_end = time.time() + 5
        global name_return
        name_return = "unknown"
        while (self._run_flag):
            ret, img = cap.read()
            img = cv2.flip(img,1)
            if ret:
                self.change_pixmap_signal.emit(img)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # If confidence is greater than 55
                if (confidence > 85):
                    name_return = names[id]
                else:
                    name_return = "unknown"
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        #self.wait() 
        #.wait may cause a hang up from the OS

#====================Making Unclickable Widgets Clickable=====================
def clickable(widget):
    class Filter(QObject):
        clicked = pyqtSignal()
        def eventFilter(self,obj,event):
            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    self.clicked.emit()
                    return True

            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked

class App(QWidget):

    #region ======================Updating Plots=====================
    def _update_canvas_temp(self):
        t = np.linspace(-60, 0, 61)
        global current_temp
        self.ys.append(current_temp)
        self.ys.pop(0)
        avg_temp = calcAvg_nonZero(self.ys)
        ys_np = np.array(self.ys)
        self.plot_temp_stuff_ax.set_ylim([avg_temp-20, avg_temp+20])
        self.plot_monitor_temp_button_back_ax.set_ylim([avg_temp-20, avg_temp+20])
        self._line.set_data(t, ys_np)
        self.plot_monitor_temp_button_back_line.set_data(t, ys_np)
        self._line.figure.canvas.draw()
        self.plot_monitor_temp_button_back_line.figure.canvas.draw()

    def _update_canvas_co2(self):
        t = np.linspace(-60, 0, 61)
        global current_co2
        self.ys_co2.append(current_co2)
        self.ys_co2.pop(0)
        avg_co2 = calcAvg_nonZero(self.ys_co2)
        ys_np_co2 = np.array(self.ys_co2)
        self.plot_co2_stuff_ax.set_ylim([0, avg_co2+5])
        self.plot_monitor_co2_button_back_ax.set_ylim([0, avg_co2+5])
        self._line_co2.set_data(t, ys_np_co2)
        self.plot_monitor_co2_button_back_line.set_data(t, ys_np_co2)
        self._line_co2.figure.canvas.draw()
        self.plot_monitor_co2_button_back_line.figure.canvas.draw()

    def _update_canvas_ppm(self):
        t = np.linspace(-60, 0, 61)
        global current_ppm
        self.ys_ppm.append(current_ppm)
        self.ys_ppm.pop(0)
        avg_ppm = calcAvg_nonZero(self.ys_ppm)
        ys_np_ppm = np.array(self.ys_ppm)
        self.plot_ppm_stuff_ax.set_ylim([0, avg_ppm+25])
        self.plot_monitor_ppm_button_back_ax.set_ylim([0, avg_ppm+25])
        self._line_ppm.set_data(t, ys_np_ppm)
        self.plot_monitor_ppm_button_back_line.set_data(t, ys_np_ppm)
        self._line_ppm.figure.canvas.draw()
        self.plot_monitor_ppm_button_back_line.figure.canvas.draw()

    def _update_canvas_oxy_press(self):
        t = np.linspace(-60, 0, 61)
        global current_oxy_press
        self.ys_oxy_press.append(current_oxy_press)
        self.ys_oxy_press.pop(0)
        avg_oxy_press = calcAvg_nonZero(self.ys_oxy_press)
        ys_np_oxy_press = np.array(self.ys_oxy_press)
        self.plot_oxy_press_stuff_ax.set_ylim([avg_oxy_press-2, avg_oxy_press+2])
        self.plot_monitor_oxy_press_button_back_ax.set_ylim([avg_oxy_press-2, avg_oxy_press+2])
        self._line_oxy_press.set_data(t, ys_np_oxy_press)
        self.plot_monitor_oxy_press_button_back_line.set_data(t, ys_np_oxy_press)
        self._line_oxy_press.figure.canvas.draw()
        self.plot_monitor_oxy_press_button_back_line.figure.canvas.draw()

    def _update_canvas_nit_press(self):
        t = np.linspace(-60, 0, 61)
        global current_nit_press
        self.ys_nit_press.append(current_nit_press)
        self.ys_nit_press.pop(0)
        avg_nit_press = calcAvg_nonZero(self.ys_nit_press)
        ys_np_nit_press = np.array(self.ys_nit_press)
        self.plot_nit_press_stuff_ax.set_ylim([avg_nit_press-2, avg_nit_press+2])
        self.plot_monitor_nit_press_button_back_ax.set_ylim([avg_nit_press-2, avg_nit_press+2])
        self._line_nit_press.set_data(t, ys_np_nit_press)
        self.plot_monitor_nit_press_button_back_line.set_data(t, ys_np_nit_press)
        self._line_nit_press.figure.canvas.draw()
        self.plot_monitor_nit_press_button_back_line.figure.canvas.draw()

    def _update_canvas_stress(self):
        t = np.linspace(-60, 0, 61)
        global current_stress
        self.ys_stress.append(current_stress)
        self.ys_stress.pop(0)
        avg_stress = calcAvg_nonZero(self.ys_stress)
        ys_np_stress = np.array(self.ys_stress)
        self.plot_stress_stuff_ax.set_ylim([0, 100])
        self.plot_monitor_stress_button_back_ax.set_ylim([0, 100])
        self._line_stress.set_data(t, ys_np_stress)
        self.plot_monitor_stress_button_back_line.set_data(t, ys_np_stress)
        self._line_stress.figure.canvas.draw()
        self.plot_monitor_stress_button_back_line.figure.canvas.draw()
    #endregion
    def __init__(self):
        super().__init__()

        #Creates the Scheduler Threads
        stop_run_continously = run_continuously()

        #This functions allows the user to press default user and move
        #to the next screen using default user's model
        def func1():
            print("button clicked")
            self.stacked_widget.setCurrentIndex(1)
            self.home_label.setText("Welcome, Default User")
            self.thread.stop()
            schedule.clear('user_auth_task')
            global auth_check
            auth_check = False
            self.home_button.setVisible(True)

        #This function checks if a valid user is detected and should
        #move to their personalized screen 
        def func2():
            global auth_check
            while auth_check:
                global name_return
                print(name_return)
                if(not(name_return == "unknown")):
                    print("User detected")
                    time.sleep(2)
                    self.stacked_widget.setCurrentIndex(1)
                    self.home_label.setText("Welcome, " + name_return)
                    schedule.clear('user_auth_task')
                    self.thread.stop()
                    break
            self.home_button.setVisible(True)

        #This function changes the right side of the monitor screen,
        #depending on which left-side "button" was clicked
        def func3(index):
            if(index == 0):
                self.monitor_right_area.setCurrentIndex(index)
                self.monitor_temp_button_front.setStyleSheet("background-color: rgb(240,240,240);")
                self.monitor_co2_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_ppm_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_oxy_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_nit_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_stress_button_front.setStyleSheet("background-color: rgb(255,255,255);")
            elif(index == 1):
                self.monitor_right_area.setCurrentIndex(index) 
                self.monitor_temp_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_co2_button_front.setStyleSheet("background-color: rgb(240,240,240);")
                self.monitor_ppm_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_oxy_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_nit_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_stress_button_front.setStyleSheet("background-color: rgb(255,255,255);")
            elif(index == 2):
                self.monitor_right_area.setCurrentIndex(index) 
                self.monitor_temp_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_co2_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_ppm_button_front.setStyleSheet("background-color: rgb(240,240,240);")
                self.monitor_oxy_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_nit_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_stress_button_front.setStyleSheet("background-color: rgb(255,255,255);")
            elif(index == 3):
                self.monitor_right_area.setCurrentIndex(index)
                self.monitor_temp_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_co2_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_ppm_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_oxy_press_button_front.setStyleSheet("background-color: rgb(240,240,240);")
                self.monitor_nit_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_stress_button_front.setStyleSheet("background-color: rgb(255,255,255);")
            elif(index == 4):
                self.monitor_right_area.setCurrentIndex(index)
                self.monitor_temp_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_co2_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_ppm_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_oxy_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_nit_press_button_front.setStyleSheet("background-color: rgb(240,240,240);")
                self.monitor_stress_button_front.setStyleSheet("background-color: rgb(255,255,255);")
            elif(index == 5):
                self.monitor_right_area.setCurrentIndex(index)
                self.monitor_temp_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_co2_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_ppm_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_oxy_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_nit_press_button_front.setStyleSheet("background-color: rgb(255,255,255);")
                self.monitor_stress_button_front.setStyleSheet("background-color: rgb(240,240,240);")

        #This function sets the screen back to the home screen
        def func4():
            self.stacked_widget.setCurrentIndex(1)
            self.home_label.setText("Home")


        #region Basic Element Settings
        #Set Font
        font = QtGui.QFont()
        font.setFamily("Myriad Pro")
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(False)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        
        self.setWindowTitle("Welcome")
        self.disply_width = 1280
        self.display_height = 720
        self.resize(self.disply_width, self.display_height)
        self.setMaximumSize(QSize(1280, 720))
        #Creating Stacked Widget for multiple screens
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.setGeometry(0,76,1280,645)

        #endregion

        #region Creating different page layouts
        self.welcome_screen = QWidget()
        self.home_screen = QWidget()
        self.monitor_screen = QWidget()
        self.web_screen = QWidget()
        self.systems_screen = QWidget()
        self.settings_screen = QWidget()
        self.stress_relief_screen = QWidget()
        self.communications_screen = QWidget()
        self.maps_screen = QWidget()
        self.eject_screen = QWidget()

        self.monitor_screen.setStyle(QStyleFactory.create("Fusion"))
        self.home_screen.setStyle(QStyleFactory.create("Fusion"))

        self.stacked_widget.addWidget(self.welcome_screen)
        self.stacked_widget.addWidget(self.home_screen)
        self.stacked_widget.addWidget(self.monitor_screen)
        self.stacked_widget.addWidget(self.web_screen)
        self.stacked_widget.addWidget(self.systems_screen)
        self.stacked_widget.addWidget(self.settings_screen)
        self.stacked_widget.addWidget(self.stress_relief_screen)
        self.stacked_widget.addWidget(self.communications_screen)
        self.stacked_widget.addWidget(self.maps_screen)
        self.stacked_widget.addWidget(self.eject_screen)

        #endregion
        #region ========================Common Elements========================
        #Ribbon Bar

        def StatusBarColor():
            global current_temp, current_ppm, current_co2
            if(current_temp >= 95 or current_temp <= 50 or current_ppm >= 200 or current_co2 >= 2.0 or current_stress >= 75 or current_nit_press >=13 or current_nit_press <=10 or current_oxy_press >= 4 or current_oxy_press <= 2):
                self.top_bar.setStyleSheet("background-color: rgb(181, 23, 0);")
            elif((current_temp > 85 and current_temp < 95) or (current_temp < 65 and current_temp > 50) or (current_ppm > 50 and current_ppm < 200) or (current_co2 > 1.2 and current_co2 < 2.0) or (current_stress > 50 and current_stress < 75) or (current_oxy_press > 3.45 and current_oxy_press < 4) or (current_oxy_press < 2.95 and current_oxy_press > 2) or (current_nit_press > 11.75 and current_nit_press < 13) or (current_nit_press < 11.25 and current_nit_press > 10)):
                self.top_bar.setStyleSheet("background-color: rgb(255, 201, 9);")
            else:
                self.top_bar.setStyleSheet("background-color: rgb(85, 142, 72);")

        self.top_bar = QFrame(self)
        self.top_bar.setGeometry(QRect(0, 0, 1280, 75))
        global status_bar_color
        self.top_bar.setStyleSheet("background-color: rgb(85, 142, 72);")
        self.top_bar.setFrameShape(QFrame.StyledPanel)
        self.top_bar.setFrameShadow(QFrame.Raised)
        self.top_bar.setLineWidth(0)
        self.top_bar.setObjectName("top_bar")
        schedule.every().seconds.do(StatusBarColor)

        #Top Left Corner Text
        font.setPointSize(25)
        self.home_label = QLabel(self.top_bar)
        self.home_label.setGeometry(QRect(25, 20, 500, 40))
        self.home_label.setFont(font)
        self.home_label.setAcceptDrops(False)
        self.home_label.setStyleSheet("color: rgb(255, 255, 255);\n"
        "background-color: rgba(255, 255, 255, 0);")
        self.home_label.setFrameShape(QFrame.NoFrame)
        self.home_label.setFrameShadow(QFrame.Plain)
        self.home_label.setTextFormat(Qt.RichText)
        self.home_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.home_label.setObjectName("home_label")
        self.home_label.setText("Welcome")


        #Top Right Quick Bar
        self.frame = QFrame(self.top_bar)
        self.frame.setEnabled(True)
        self.frame.setGeometry(QRect(910, 15, 350, 45))
        self.frame.setStyleSheet("border-radius:22px;\n"
        "background-color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.frame.setObjectName("frame")

        #Time on top right
        def time_label_schedule():
            self.time_label.setText(time_right_now)
        self.time_label = QLabel(self.frame)
        self.time_label.setGeometry(QRect(266, 5, 60, 35))
        font.setPointSize(21)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(12)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.time_label.setFont(font)
        self.time_label.setStyleSheet("color: rgb(0, 0, 0);\n"
        "font-weight:100;")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setObjectName("time_label")
        schedule.every().seconds.do(StatusBarClock)
        schedule.every().seconds.do(time_label_schedule)

        #Temperature
        def temp_label_schedule():
            self.temp_label.setText(str(round(current_temp,1)) + "°F")
        self.temp_label = QLabel(self.frame)
        self.temp_label.setGeometry(QRect(140,5,85,35))
        font.setPointSize(21)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(12)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.temp_label.setFont(font)
        self.temp_label.setObjectName("temp_label")
        schedule.every(5).seconds.do(temp_label_schedule)

        #Home Button
        self.home_button = QLabel(self.frame)
        self.home_button.setPixmap(QPixmap("C:/Users/Raymond Fey/Desktop/MHIS_PY/script/home.png"))
        self.home_button.setGeometry(QRect(10,8,30,30))
        self.home_button.setScaledContents(True)
        self.home_button.setVisible(False)
        clickable(self.home_button).connect(func4)
        #endregion
        #region =========================Login Screen==========================
        #Set Background Color
        self.welcome_screen.setStyleSheet("background-color: rgb(255,255,255);")
        #Please Scan Face Text
        self.scan_face_label = QLabel(self.welcome_screen)
        self.scan_face_label.setGeometry(QRect(480, 75, 320, 50))
        font.setPointSize(25)
        self.scan_face_label.setFont(font)
        self.scan_face_label.setStyleSheet("color: rgb(95, 95, 95);")
        self.scan_face_label.setAlignment(Qt.AlignCenter)
        self.scan_face_label.setObjectName("scan_face_label")
        self.scan_face_label.setText("Please Scan Your Face")

        #Default User Button
        self.default_user_button = QPushButton(self.welcome_screen)
        self.default_user_button.setGeometry(QRect(520, 530, 250, 35))
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(12)
        self.default_user_button.setFont(font)
        self.default_user_button.setAcceptDrops(False)
        self.default_user_button.setStyleSheet("color: rgb(115, 190, 249);\n"
        "font-weight:100;")
        self.default_user_button.setAutoDefault(False)
        self.default_user_button.setDefault(False)
        self.default_user_button.setFlat(True)
        self.default_user_button.setObjectName("default_user_button")
        self.default_user_button.setText("Procced as Default User")
        self.default_user_button.clicked.connect(func1)

        #Webcam Image 
        # create the label that holds the image
        self.image_label = QLabel(self.welcome_screen)
        self.image_label.setGeometry(QRect(440,130,400,400))
        self.image_label.setAlignment(Qt.AlignCenter)
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        schedule.every().seconds.do(func2).tag('user_auth_task')
        #endregion
        #region =========================Home Screen===========================
        #Set Base Layer for Home Screen
        self.home_scrollArea = QScrollArea(self.home_screen)
        self.home_scrollArea.setStyle(QStyleFactory.create("Fusion"))
        self.home_scrollArea.setStyleSheet("background-color: rgb(255,255,255);")
        self.home_scrollArea.setGeometry(QRect(0,0,1280,645))
        self.home_scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.home_scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.home_scrollArea.setWidgetResizable(True)
        self.home_scrollArea.setObjectName("home_scrollArea")

        self.home_scrollArea_content = QWidget(self.home_scrollArea)
        self.home_scrollArea_content.setGeometry(QRect(0, 0, 1280, 1600))
        self.home_scrollArea_content.setObjectName("home_scrollArea_content")

        self.home_verticalLayout = QVBoxLayout(self.home_scrollArea_content)
        self.home_verticalLayout.setObjectName("home_verticalLayout")

        self.home_verticalLayout_content = QWidget(self.home_scrollArea_content)
        self.home_verticalLayout_content.setMinimumSize(QSize(0,1290))
        self.home_verticalLayout_content.setObjectName("home_verticalLayout_content")

        self.home_horizontalLayoutWidget = QWidget(self.home_verticalLayout_content)
        self.home_horizontalLayoutWidget.setGeometry(QRect(10,800,1231,481))
        self.home_horizontalLayoutWidget.setObjectName("home_horizontalLayoutWidget")

        self.application_window = QHBoxLayout(self.home_horizontalLayoutWidget)
        self.application_window.setContentsMargins(0, 0, 0, 0)
        self.application_window.setObjectName("application_window")

        self.app_col_1 = QVBoxLayout()
        self.app_col_1.setObjectName("app_col_1")
        self.app_col_2 = QVBoxLayout()
        self.app_col_2.setObjectName("app_col_2")
        self.app_col_3 = QVBoxLayout()
        self.app_col_3.setObjectName("app_col_3")

        #region Setting Apps in column 1
        self.app_1 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_1.sizePolicy().hasHeightForWidth())
        self.app_1.setSizePolicy(sizePolicy)
        self.app_1.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_1.setObjectName("app_1")
        self.app_col_1.addWidget(self.app_1)
        self.app_2 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_2.sizePolicy().hasHeightForWidth())
        self.app_2.setSizePolicy(sizePolicy)
        self.app_2.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_2.setObjectName("app_2")
        self.app_col_1.addWidget(self.app_2)
        self.app_3 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_3.sizePolicy().hasHeightForWidth())
        self.app_3.setSizePolicy(sizePolicy)
        self.app_3.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_3.setObjectName("app_3")
        self.app_col_1.addWidget(self.app_3)
        self.application_window.addLayout(self.app_col_1)
        #endregion

        #region Setting Apps in columm 2
        self.app_4 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_4.sizePolicy().hasHeightForWidth())
        self.app_4.setSizePolicy(sizePolicy)
        self.app_4.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_4.setObjectName("app_4")
        self.app_col_2.addWidget(self.app_4)
        self.app_5 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_5.sizePolicy().hasHeightForWidth())
        self.app_5.setSizePolicy(sizePolicy)
        self.app_5.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_5.setObjectName("app_2")
        self.app_col_2.addWidget(self.app_5)
        self.app_6 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_6.sizePolicy().hasHeightForWidth())
        self.app_6.setSizePolicy(sizePolicy)
        self.app_6.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_6.setObjectName("app_6")
        self.app_col_2.addWidget(self.app_6)
        self.application_window.addLayout(self.app_col_2)
        #endregion

        #region Setting Apps in columm 3
        self.app_7 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_7.sizePolicy().hasHeightForWidth())
        self.app_7.setSizePolicy(sizePolicy)
        self.app_7.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_7.setObjectName("app_7")
        self.app_col_3.addWidget(self.app_7)
        self.app_8 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_8.sizePolicy().hasHeightForWidth())
        self.app_8.setSizePolicy(sizePolicy)
        self.app_8.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_8.setObjectName("app_8")
        self.app_col_3.addWidget(self.app_8)
        self.app_9 = QPushButton(self.home_horizontalLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.app_9.sizePolicy().hasHeightForWidth())
        self.app_9.setSizePolicy(sizePolicy)
        self.app_9.setStyleSheet("color: rgb(0, 0, 0);")
        self.app_9.setObjectName("app_9")
        self.app_col_3.addWidget(self.app_9)
        self.application_window.addLayout(self.app_col_3)
        #endregion

        #region Suggested Apps Layout
        self.horizontalLayoutWidget_2 = QWidget(self.home_verticalLayout_content)
        self.horizontalLayoutWidget_2.setGeometry(QRect(20, 150, 1221, 401))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.suggested_app_1 = QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.suggested_app_1.sizePolicy().hasHeightForWidth())
        self.suggested_app_1.setSizePolicy(sizePolicy)
        self.suggested_app_1.setMinimumSize(QSize(300, 300))
        self.suggested_app_1.setStyleSheet("color: rgb(0, 0, 0);")
        self.suggested_app_1.setFlat(False)
        self.suggested_app_1.setObjectName("suggested_app_1")
        self.horizontalLayout.addWidget(self.suggested_app_1)
        self.suggested_app_2 = QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.suggested_app_2.sizePolicy().hasHeightForWidth())
        self.suggested_app_2.setSizePolicy(sizePolicy)
        self.suggested_app_2.setMinimumSize(QSize(300, 300))
        self.suggested_app_2.setStyleSheet("color: rgb(0, 0, 0);")
        self.suggested_app_2.setObjectName("suggested_app_2")
        self.horizontalLayout.addWidget(self.suggested_app_2)
        self.suggested_app_3 = QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.suggested_app_1.sizePolicy().hasHeightForWidth())
        self.suggested_app_3.setSizePolicy(sizePolicy)
        self.suggested_app_3.setMinimumSize(QSize(300, 300))
        self.suggested_app_3.setStyleSheet("color: rgb(0, 0, 0);")
        self.suggested_app_3.setObjectName("suggested_app_3")
        self.horizontalLayout.addWidget(self.suggested_app_3)
        self.suggested_app_1.setText("App 1")
        self.suggested_app_2.setText("App 2")
        
        self.suggested_app_3.setText("App 3")

        #endregion
        
        self.home_verticalLayout.addWidget(self.home_verticalLayout_content)
        self.home_scrollArea.setWidget(self.home_scrollArea_content)

        #endregion
        #region ========================Monitor Screen=========================
        
        #Set Background Color
        self.monitor_screen.setStyleSheet("background-color: rgb(255,255,255);")
        #Partioning Horizontally
        self.horizontalLayoutWidget = QWidget(self.monitor_screen)
        self.horizontalLayoutWidget.setGeometry(QRect(0, 0, 1280, 645))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        #region Left Side scroll area
        self.scrollArea_left = QScrollArea(self.horizontalLayoutWidget)
        self.scrollArea_left.setWidgetResizable(True)
        self.scrollArea_left.setObjectName("scrollArea_left")
        self.scrollAreaWidgetContents_left = QWidget()
        self.scrollAreaWidgetContents_left.setStyleSheet("background-color:rgb(250,250,250);")
        self.scrollAreaWidgetContents_left.setGeometry(QRect(0, 0, 388, 637))
        self.scrollAreaWidgetContents_left.setObjectName("scrollAreaWidgetContents_left")
        self.scrollArea_left.setWidget(self.scrollAreaWidgetContents_left)

        self.horizontalLayout.addWidget(self.scrollArea_left)
        self.horizontalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        #endregion
        #region Create the Different "Buttons To Press On"
        #region Temperature
        self.monitor_buttons_container = QVBoxLayout(self.scrollArea_left)
        self.monitor_buttons_container.setGeometry(QRect(0,0,320,645))
        self.monitor_temp_button_front = QWidget()
        self.monitor_temp_button_back = QWidget(self.monitor_temp_button_front)
        self.monitor_temp_button_back.setMaximumSize(QSize(120,100))
        self.monitor_temp_button_back_vbox = QVBoxLayout(self.monitor_temp_button_back)
        self.plot_monitor_temp_button_back = FigureCanvas(Figure(figsize=(1.5,1)))
        self.monitor_temp_button_back_vbox.addWidget(self.plot_monitor_temp_button_back)
        self.monitor_buttons_container.addWidget(self.monitor_temp_button_front)
        self.plot_monitor_temp_button_back_ax = self.plot_monitor_temp_button_back.figure.subplots()
        self.plot_monitor_temp_button_back_ax.set_yticklabels([])
        self.plot_monitor_temp_button_back_ax.set_xticklabels([])
        self.plot_monitor_temp_button_back_ax.tick_params(length=0)
        self.plot_monitor_temp_button_back_ax.set_xlim([-60,0])
        t = np.linspace(-60, 0, 61)
        self.ys = [0] * 61
        ys_np = np.array(self.ys)
        self.plot_monitor_temp_button_back_line, = self.plot_monitor_temp_button_back_ax.plot(t, ys_np)
        clickable(self.monitor_temp_button_front).connect(lambda: func3(0))
        self.monitor_temp_button_front.setStyleSheet("background-color: rgb(240,240,240);")

        self.monitor_temp_button_label = QLabel(self.monitor_temp_button_front)
        self.monitor_temp_button_label.setGeometry(QRect(125,5,150,50))
        self.monitor_temp_button_label.setObjectName("monitor_temp_button_label")
        font.setPointSize(15)
        self.monitor_temp_button_label.setFont(font)
        self.monitor_temp_button_label.setText("Temperature")
        self.monitor_temp_button_value_label = QLabel(self.monitor_temp_button_front)
        self.monitor_temp_button_value_label.setGeometry(QRect(125,50,150,20))
        self.monitor_temp_button_value_label.setObjectName("monitor_temp_button_value_label")
        font.setPointSize(12)
        self.monitor_temp_button_value_label.setFont(font)
        def update_monitor_temp():
            global current_temp
            self.monitor_temp_button_value_label.setText(str(round(current_temp,1)) + "°F")
        schedule.every().seconds.do(update_monitor_temp)
        #endregion

        #region Co2 Levels
        self.monitor_co2_button_front = QWidget()
        self.monitor_co2_button_back = QWidget(self.monitor_co2_button_front)
        self.monitor_co2_button_back.setMaximumSize(QSize(120,100))
        self.monitor_co2_button_back_vbox = QVBoxLayout(self.monitor_co2_button_back)
        self.plot_monitor_co2_button_back = FigureCanvas(Figure(figsize=(1.5,1)))
        self.monitor_co2_button_back_vbox.addWidget(self.plot_monitor_co2_button_back)
        self.monitor_buttons_container.addWidget(self.monitor_co2_button_front)
        self.plot_monitor_co2_button_back_ax = self.plot_monitor_co2_button_back.figure.subplots()
        self.plot_monitor_co2_button_back_ax.set_yticklabels([])
        self.plot_monitor_co2_button_back_ax.set_xticklabels([])
        self.plot_monitor_co2_button_back_ax.tick_params(length=0)
        self.plot_monitor_co2_button_back_ax.set_xlim([-60,0])
        self.ys_co2 = [0] * 61
        ys_np_co2 = np.array(self.ys_co2)
        self.plot_monitor_co2_button_back_line, = self.plot_monitor_co2_button_back_ax.plot(t, ys_np_co2)
        clickable(self.monitor_co2_button_front).connect(lambda: func3(1))

        self.monitor_co2_button_label = QLabel(self.monitor_co2_button_front)
        self.monitor_co2_button_label.setGeometry(QRect(125,5,150,50))
        self.monitor_co2_button_label.setObjectName("monitor_co2_button_label")
        font.setPointSize(15)
        self.monitor_co2_button_label.setFont(font)
        self.monitor_co2_button_label.setText("CO2 Levels (%)")
        self.monitor_co2_button_value_label = QLabel(self.monitor_co2_button_front)
        self.monitor_co2_button_value_label.setGeometry(QRect(125,50,150,20))
        self.monitor_co2_button_value_label.setObjectName("monitor_co2_button_value_label")
        font.setPointSize(12)
        self.monitor_co2_button_value_label.setFont(font)
        def update_monitor_co2():
            global current_co2
            self.monitor_co2_button_value_label.setText(str(round(current_co2,2)) + "%")
        schedule.every().seconds.do(update_monitor_co2)
        #endregion
        
        #region PPM Levels
        self.monitor_ppm_button_front = QWidget()
        self.monitor_ppm_button_back = QWidget(self.monitor_ppm_button_front)
        self.monitor_ppm_button_back.setMaximumSize(QSize(120,100))
        self.monitor_ppm_button_back_vbox = QVBoxLayout(self.monitor_ppm_button_back)
        self.plot_monitor_ppm_button_back = FigureCanvas(Figure(figsize=(1.5,1)))
        self.monitor_ppm_button_back_vbox.addWidget(self.plot_monitor_ppm_button_back)
        self.monitor_buttons_container.addWidget(self.monitor_ppm_button_front)
        self.plot_monitor_ppm_button_back_ax = self.plot_monitor_ppm_button_back.figure.subplots()
        self.plot_monitor_ppm_button_back_ax.set_yticklabels([])
        self.plot_monitor_ppm_button_back_ax.set_xticklabels([])
        self.plot_monitor_ppm_button_back_ax.tick_params(length=0)
        self.plot_monitor_ppm_button_back_ax.set_xlim([-60,0])
        self.ys_ppm = [0] * 61
        ys_np_ppm = np.array(self.ys_ppm)
        self.plot_monitor_ppm_button_back_line, = self.plot_monitor_ppm_button_back_ax.plot(t, ys_np_ppm)
        clickable(self.monitor_ppm_button_front).connect(lambda: func3(2))

        self.monitor_ppm_button_label = QLabel(self.monitor_ppm_button_front)
        self.monitor_ppm_button_label.setGeometry(QRect(125,5,150,50))
        self.monitor_ppm_button_label.setObjectName("monitor_ppm_button_label")
        font.setPointSize(15)
        self.monitor_ppm_button_label.setFont(font)
        self.monitor_ppm_button_label.setText("AQI (PPM)")
        self.monitor_ppm_button_value_label = QLabel(self.monitor_ppm_button_front)
        self.monitor_ppm_button_value_label.setGeometry(QRect(125,50,150,20))
        self.monitor_ppm_button_value_label.setObjectName("monitor_ppm_button_value_label")
        font.setPointSize(12)
        self.monitor_ppm_button_value_label.setFont(font)
        def update_monitor_ppm():
            global current_ppm
            self.monitor_ppm_button_value_label.setText(str(round(current_ppm,2)) + " PPM")
        schedule.every().seconds.do(update_monitor_ppm)
        #endregion
        
        #region Oxygen Pressure
        self.monitor_oxy_press_button_front = QWidget()
        self.monitor_oxy_press_button_back = QWidget(self.monitor_oxy_press_button_front)
        self.monitor_oxy_press_button_back.setMaximumSize(QSize(120,100))
        self.monitor_oxy_press_button_back_vbox = QVBoxLayout(self.monitor_oxy_press_button_back)
        self.plot_monitor_oxy_press_button_back = FigureCanvas(Figure(figsize=(1.5,1)))
        self.monitor_oxy_press_button_back_vbox.addWidget(self.plot_monitor_oxy_press_button_back)
        self.monitor_buttons_container.addWidget(self.monitor_oxy_press_button_front)
        self.plot_monitor_oxy_press_button_back_ax = self.plot_monitor_oxy_press_button_back.figure.subplots()
        self.plot_monitor_oxy_press_button_back_ax.set_yticklabels([])
        self.plot_monitor_oxy_press_button_back_ax.set_xticklabels([])
        self.plot_monitor_oxy_press_button_back_ax.tick_params(length=0)
        self.plot_monitor_oxy_press_button_back_ax.set_xlim([-60,0])
        self.ys_oxy_press = [0] * 61
        ys_np_oxy_press = np.array(self.ys_oxy_press)
        self.plot_monitor_oxy_press_button_back_line, = self.plot_monitor_oxy_press_button_back_ax.plot(t, ys_np_oxy_press)
        clickable(self.monitor_oxy_press_button_front).connect(lambda: func3(3))

        self.monitor_oxy_press_button_label = QLabel(self.monitor_oxy_press_button_front)
        self.monitor_oxy_press_button_label.setGeometry(QRect(125,5,150,50))
        self.monitor_oxy_press_button_label.setObjectName("monitor_oxy_press_button_label")
        font.setPointSize(15)
        self.monitor_oxy_press_button_label.setFont(font)
        self.monitor_oxy_press_button_label.setText("Oxygen Pressure")
        self.monitor_oxy_press_button_value_label = QLabel(self.monitor_oxy_press_button_front)
        self.monitor_oxy_press_button_value_label.setGeometry(QRect(125,50,150,20))
        self.monitor_oxy_press_button_value_label.setObjectName("monitor_oxy_button_value_label")
        font.setPointSize(12)
        self.monitor_oxy_press_button_value_label.setFont(font)
        def update_monitor_oxy_press():
            global current_oxy_press
            self.monitor_oxy_press_button_value_label.setText(str(round(current_oxy_press,2)) + " PSIa")
        schedule.every().seconds.do(update_monitor_oxy_press)
        #endregion

        #region Nitrogen Pressure
        self.monitor_nit_press_button_front = QWidget()
        self.monitor_nit_press_button_back = QWidget(self.monitor_nit_press_button_front)
        self.monitor_nit_press_button_back.setMaximumSize(QSize(120,100))
        self.monitor_nit_press_button_back_vbox = QVBoxLayout(self.monitor_nit_press_button_back)
        self.plot_monitor_nit_press_button_back = FigureCanvas(Figure(figsize=(1.5,1)))
        self.monitor_nit_press_button_back_vbox.addWidget(self.plot_monitor_nit_press_button_back)
        self.monitor_buttons_container.addWidget(self.monitor_nit_press_button_front)
        self.plot_monitor_nit_press_button_back_ax = self.plot_monitor_nit_press_button_back.figure.subplots()
        self.plot_monitor_nit_press_button_back_ax.set_yticklabels([])
        self.plot_monitor_nit_press_button_back_ax.set_xticklabels([])
        self.plot_monitor_nit_press_button_back_ax.tick_params(length=0)
        self.plot_monitor_nit_press_button_back_ax.set_xlim([-60,0])
        self.ys_nit_press = [0] * 61
        ys_np_nit_press = np.array(self.ys_nit_press)
        self.plot_monitor_nit_press_button_back_line, = self.plot_monitor_nit_press_button_back_ax.plot(t, ys_np_nit_press)
        clickable(self.monitor_nit_press_button_front).connect(lambda: func3(4))

        self.monitor_nit_press_button_label = QLabel(self.monitor_nit_press_button_front)
        self.monitor_nit_press_button_label.setGeometry(QRect(125,5,150,50))
        self.monitor_nit_press_button_label.setObjectName("monitor_nit_press_button_label")
        font.setPointSize(15)
        self.monitor_nit_press_button_label.setFont(font)
        self.monitor_nit_press_button_label.setText("Nitrogen Pressure")
        self.monitor_nit_press_button_value_label = QLabel(self.monitor_nit_press_button_front)
        self.monitor_nit_press_button_value_label.setGeometry(QRect(125,50,150,20))
        self.monitor_nit_press_button_value_label.setObjectName("monitor_nit_press_button_value_label")
        font.setPointSize(12)
        self.monitor_nit_press_button_value_label.setFont(font)
        def update_monitor_nit():
            global current_nit
            self.monitor_nit_press_button_value_label.setText(str(round(current_nit_press,2)) + " PSIa")
        schedule.every().seconds.do(update_monitor_nit)
        #endregion
        
        #region Stress
        self.monitor_stress_button_front = QWidget()
        self.monitor_stress_button_back = QWidget(self.monitor_stress_button_front)
        self.monitor_stress_button_back.setMaximumSize(QSize(120,100))
        self.monitor_stress_button_back_vbox = QVBoxLayout(self.monitor_stress_button_back)
        self.plot_monitor_stress_button_back = FigureCanvas(Figure(figsize=(1.5,1)))
        self.monitor_stress_button_back_vbox.addWidget(self.plot_monitor_stress_button_back)
        self.monitor_buttons_container.addWidget(self.monitor_stress_button_front)
        self.plot_monitor_stress_button_back_ax = self.plot_monitor_stress_button_back.figure.subplots()
        self.plot_monitor_stress_button_back_ax.set_yticklabels([])
        self.plot_monitor_stress_button_back_ax.set_xticklabels([])
        self.plot_monitor_stress_button_back_ax.tick_params(length=0)
        self.plot_monitor_stress_button_back_ax.set_xlim([-60,0])
        self.ys_stress = [0] * 61
        ys_np_stress = np.array(self.ys_stress)
        self.plot_monitor_stress_button_back_line, = self.plot_monitor_stress_button_back_ax.plot(t, ys_np_stress)
        clickable(self.monitor_stress_button_front).connect(lambda: func3(5))

        self.monitor_stress_button_label = QLabel(self.monitor_stress_button_front)
        self.monitor_stress_button_label.setGeometry(QRect(125,5,150,50))
        self.monitor_stress_button_label.setObjectName("monitor_stress_button_label")
        font.setPointSize(15)
        self.monitor_stress_button_label.setFont(font)
        self.monitor_stress_button_label.setText("Stress")
        self.monitor_stress_button_value_label = QLabel(self.monitor_stress_button_front)
        self.monitor_stress_button_value_label.setGeometry(QRect(125,50,150,20))
        self.monitor_stress_button_value_label.setObjectName("monitor_stress_button_value_label")
        font.setPointSize(12)
        self.monitor_stress_button_value_label.setFont(font)
        def update_monitor_nit():
            global current_nit
            self.monitor_stress_button_value_label.setText(str(round(current_stress,2)) + "%")
        schedule.every().seconds.do(update_monitor_nit)
        #endregion
        #endregion
        #region Right Side scroll area and different views
        self.monitor_right_area = QStackedWidget()
        self.horizontalLayout.addWidget(self.monitor_right_area)
        self.monitor_right_area.setMinimumSize(QSize(960,0))

        self.temp_monitor = QWidget()
        self.co2_monitor = QWidget()
        self.ppm_monitor = QWidget()
        self.oxy_press_monitor = QWidget()
        self.nit_press_monitor = QWidget()
        self.stress_monitor = QWidget()

        self.monitor_right_area.addWidget(self.temp_monitor)
        self.monitor_right_area.addWidget(self.co2_monitor)     
        self.monitor_right_area.addWidget(self.ppm_monitor)  
        self.monitor_right_area.addWidget(self.oxy_press_monitor)
        self.monitor_right_area.addWidget(self.nit_press_monitor)
        self.monitor_right_area.addWidget(self.stress_monitor)

        #region Temp Screen on Right Side Monitor
        self.scrollArea_temp = QScrollArea(self.temp_monitor)
        self.scrollArea_temp.setGeometry(QRect(0,0,960,900))
        self.current_temp_label = QLabel(self.scrollArea_temp)
        self.current_temp_label.setGeometry(QRect(20, 30, 250, 40))
        self.current_temp_label.setObjectName("current_temp_label")
        font.setPointSize(25)
        self.current_temp_label.setFont(font)
        self.current_temp_label.setText("Temperature")
        self.current_temp_value = QLabel(self.scrollArea_temp)
        self.current_temp_value.setGeometry(QRect(25,480,100,100))
        self.current_temp_value.setObjectName("current_temp_value")
        self.current_temp_value.setFont(font)
        def getCurrentTempValue():
            self.current_temp_value.setText(str(round(current_temp,1)) + "°F")

        #Text for Current Temperatue Label
        self.current_temp_value_top = QLabel(self.scrollArea_temp)
        font.setPointSize(12)
        self.current_temp_value_top.setFont(font)
        self.current_temp_value_top.setGeometry(QRect(25,475,175,25))
        self.current_temp_value_top.setObjectName("current_temp_value_top")
        self.current_temp_value_top.setText("Current Temperature")
        schedule.every().seconds.do(getCurrentTempValue)

        #Text for Min Temperature Label
        self.min_temp_label = QLabel(self.scrollArea_temp)
        font.setPointSize(12)
        self.min_temp_label.setFont(font)
        self.min_temp_label.setGeometry(QRect(550,475,175,25))
        self.min_temp_label.setObjectName("min_temp_label")
        self.min_temp_label.setText("Min. Temperature")

        #Text for Min Temperature Value
        self.min_temp_value = QLabel(self.scrollArea_temp)
        font.setPointSize(12)
        self.min_temp_value.setFont(font)
        self.min_temp_value.setGeometry(QRect(700,475,175,25))
        self.min_temp_value.setObjectName("min_temp_value")
        def setMinTemp():
            global min_temp
            if(current_temp < min_temp):
                min_temp = current_temp
            self.min_temp_value.setText(str(min_temp) + "°F")
        schedule.every().seconds.do(setMinTemp)

        #Text for Man Temperature Label
        self.max_temp_label = QLabel(self.scrollArea_temp)
        font.setPointSize(12)
        self.max_temp_label.setFont(font)
        self.max_temp_label.setGeometry(QRect(550,500,175,25))
        self.max_temp_label.setObjectName("max_temp_label")
        self.max_temp_label.setText("Max Temperature")

        #Text for Max Temperature Value
        self.max_temp_value = QLabel(self.scrollArea_temp)
        font.setPointSize(12)
        self.max_temp_value.setFont(font)
        self.max_temp_value.setGeometry(QRect(700,500,175,25))
        self.max_temp_value.setObjectName("max_temp_value")
        def setMaxTemp():
            global max_temp
            if(current_temp > max_temp):
                max_temp = current_temp
            self.max_temp_value.setText(str(max_temp) + "°F")
        schedule.every().seconds.do(setMaxTemp)

        #Text for Average Temperature Label
        self.avg_temp_label = QLabel(self.scrollArea_temp)
        font.setPointSize(12)
        self.avg_temp_label.setFont(font)
        self.avg_temp_label.setGeometry(QRect(550,525,175,25))
        self.avg_temp_label.setObjectName("avg_temp_label")
        self.avg_temp_label.setText("Avg. Temperature")

        #Text for Avg Temperature Value
        self.avg_temp_value = QLabel(self.scrollArea_temp)
        font.setPointSize(12)
        self.avg_temp_value.setFont(font)
        self.avg_temp_value.setGeometry(QRect(700,525,175,25))
        self.avg_temp_value.setObjectName("avg_temp_value")
        def setAvgTemp():
            global avg_temp, iterations
            avg_temp = (avg_temp*iterations + current_temp)/(iterations+1)
            iterations = iterations + 1
            self.avg_temp_value.setText(str(round(avg_temp,2)) + "°F")
        schedule.every().seconds.do(setAvgTemp)
       
        #Combo Box for Temp
        self.comboBox_temp = QComboBox(self.scrollArea_temp)
        self.comboBox_temp.setGeometry(QRect(550, 70, 150, 25))
        self.comboBox_temp.setEditable(False)
        self.comboBox_temp.setCurrentText("Units")
        self.comboBox_temp.setObjectName("comboBox")
        self.comboBox_temp.addItems(["Farhenheit","Celcius", "Kelvin"])

        
        #Plot Figure for Temperature
        self.plot_temp_widget = QWidget(self.scrollArea_temp)
        self.plot_temp_widget.setGeometry(QRect(-50,75,600, 400))
        self.plot_temp_stuff = FigureCanvas(Figure(figsize=(2,2)))
        self.plot_widget = QVBoxLayout(self.plot_temp_widget)
        self.plot_widget.addWidget(self.plot_temp_stuff)
        self.plot_temp_stuff_ax = self.plot_temp_stuff.figure.subplots()

        #Plot attributes for Temperature
        #self.plot_temp_stuff_ax.set_yticklabels([])
        self.plot_temp_stuff_ax.set_xticklabels([])
        self.plot_temp_stuff_ax.tick_params(length=0)
        self.plot_temp_stuff_ax.grid()
        self.plot_temp_stuff_ax.set_xlim([-60,0])
        #Set up a Line2D.
        self._line, = self.plot_temp_stuff_ax.plot(t, ys_np)
        self._timer = self.plot_temp_stuff.new_timer(1000)
        self._timer.add_callback(self._update_canvas_temp)
        self._timer.start()
        #endregion

        #region Co2 Screen on Right Side Monitor
        self.scrollArea_co2 = QScrollArea(self.co2_monitor)
        self.scrollArea_co2.setGeometry(QRect(0,0,960,900))
        self.current_co2_label = QLabel(self.scrollArea_co2)
        self.current_co2_label.setGeometry(QRect(20, 30, 250, 40))
        self.current_co2_label.setObjectName("current_co2_label")
        font.setPointSize(25)
        self.current_co2_label.setFont(font)
        self.current_co2_label.setText("CO2 Levels (%)")
        self.current_co2_value = QLabel(self.scrollArea_co2)
        self.current_co2_value.setGeometry(QRect(25,480,100,100))
        self.current_co2_value.setObjectName("current_co2_value")
        self.current_co2_value.setFont(font)
        def getCurrentCO2Value():
            self.current_co2_value.setText(str(round(current_co2,2)) + "%")

        #Text for Current CO2 Label
        self.current_co2_value_top = QLabel(self.scrollArea_co2)
        font.setPointSize(12)
        self.current_co2_value_top.setFont(font)
        self.current_co2_value_top.setGeometry(QRect(25,475,175,25))
        self.current_co2_value_top.setObjectName("current_co2_value_top")
        self.current_co2_value_top.setText("Current CO2")
        schedule.every().seconds.do(getCurrentCO2Value)

        #Text for Min CO2 Label
        self.min_co2_label = QLabel(self.scrollArea_co2)
        font.setPointSize(12)
        self.min_co2_label.setFont(font)
        self.min_co2_label.setGeometry(QRect(550,475,175,25))
        self.min_co2_label.setObjectName("min_co2_label")
        self.min_co2_label.setText("Min. CO2")

        #Text for Min CO2 Value
        self.min_co2_value = QLabel(self.scrollArea_co2)
        font.setPointSize(12)
        self.min_co2_value.setFont(font)
        self.min_co2_value.setGeometry(QRect(650,475,175,25))
        self.min_co2_value.setObjectName("min_co2_value")
        def setMinCO2():
            global min_co2
            if(current_co2 < min_co2):
                min_co2 = current_co2
            self.min_co2_value.setText(str(min_co2) + "%")
        schedule.every().seconds.do(setMinCO2)

        #Text for Max CO2 Label
        self.max_co2_label = QLabel(self.scrollArea_co2)
        font.setPointSize(12)
        self.max_co2_label.setFont(font)
        self.max_co2_label.setGeometry(QRect(550,500,175,25))
        self.max_co2_label.setObjectName("max_co2_label")
        self.max_co2_label.setText("Max CO2")

        #Text for Max CO2 Value
        self.max_co2_value = QLabel(self.scrollArea_co2)
        font.setPointSize(12)
        self.max_co2_value.setFont(font)
        self.max_co2_value.setGeometry(QRect(650,500,175,25))
        self.max_co2_value.setObjectName("max_co2_value")
        def setMaxCO2():
            global max_co2
            if(current_co2 > max_co2):
                max_co2 = current_co2
            self.max_co2_value.setText(str(max_co2) + "%")
        schedule.every().seconds.do(setMaxCO2)

        #Text for Average CO2 Label
        self.avg_co2_label = QLabel(self.scrollArea_co2)
        font.setPointSize(12)
        self.avg_co2_label.setFont(font)
        self.avg_co2_label.setGeometry(QRect(550,525,175,25))
        self.avg_co2_label.setObjectName("avg_co2_label")
        self.avg_co2_label.setText("Avg. CO2")

        #Text for Avg CO2 Value
        self.avg_co2_value = QLabel(self.scrollArea_co2)
        font.setPointSize(12)
        self.avg_co2_value.setFont(font)
        self.avg_co2_value.setGeometry(QRect(650,525,175,25))
        self.avg_co2_value.setObjectName("avg_co2_value")
        def setAvgCO2():
            global avg_co2, iterations
            avg_co2 = (avg_co2*iterations + current_co2)/(iterations+1)
            iterations = iterations + 1
            self.avg_co2_value.setText(str(round(avg_co2,2)) + "%")
        schedule.every().seconds.do(setAvgCO2)

        #Plot Figure for CO2
        self.plot_co2_widget = QWidget(self.scrollArea_co2)
        self.plot_co2_widget.setGeometry(QRect(-50,75,600, 400))
        self.plot_co2_stuff = FigureCanvas(Figure(figsize=(2,2)))
        self.plot_widget_co2 = QVBoxLayout(self.plot_co2_widget)
        self.plot_widget_co2.addWidget(self.plot_co2_stuff)
        self.plot_co2_stuff_ax = self.plot_co2_stuff.figure.subplots()

        #Plot attributes for CO2
        self.plot_co2_stuff_ax.set_xticklabels([])
        self.plot_co2_stuff_ax.tick_params(length=0)
        self.plot_co2_stuff_ax.grid()
        self.plot_co2_stuff_ax.set_xlim([-60,0])
        #Set up a Line2D.
        self._line_co2, = self.plot_co2_stuff_ax.plot(t, ys_np_co2)
        self._timer_co2 = self.plot_co2_stuff.new_timer(1000)
        self._timer_co2.add_callback(self._update_canvas_co2)
        self._timer_co2.start()
        #endregion

        #region PPM Screen on Right Side Monitor
        self.scrollArea_ppm = QScrollArea(self.ppm_monitor)
        self.scrollArea_ppm.setGeometry(QRect(0,0,960,900))
        self.current_ppm_label = QLabel(self.scrollArea_ppm)
        self.current_ppm_label.setGeometry(QRect(20, 30, 250, 40))
        self.current_ppm_label.setObjectName("current_ppm_label")
        font.setPointSize(25)
        self.current_ppm_label.setFont(font)
        self.current_ppm_label.setText("AQI (PPM)")
        self.current_ppm_value = QLabel(self.scrollArea_ppm)
        self.current_ppm_value.setGeometry(QRect(25,480,150,100))
        self.current_ppm_value.setObjectName("current_ppm_value")
        self.current_ppm_value.setFont(font)
        def getCurrentPPMValue():
            self.current_ppm_value.setText(str(round(current_ppm,2)) + " PPM")

        #Text for Current PPM Label
        self.current_ppm_value_top = QLabel(self.scrollArea_ppm)
        font.setPointSize(12)
        self.current_ppm_value_top.setFont(font)
        self.current_ppm_value_top.setGeometry(QRect(25,475,175,25))
        self.current_ppm_value_top.setObjectName("current_ppm_value_top")
        self.current_ppm_value_top.setText("Current AQI")
        schedule.every().seconds.do(getCurrentPPMValue)

        #Text for Min PPM Label
        self.min_ppm_label = QLabel(self.scrollArea_ppm)
        font.setPointSize(12)
        self.min_ppm_label.setFont(font)
        self.min_ppm_label.setGeometry(QRect(550,475,175,25))
        self.min_ppm_label.setObjectName("min_ppm_label")
        self.min_ppm_label.setText("Min. AQI")

        #Text for Min PPM Value
        self.min_ppm_value = QLabel(self.scrollArea_ppm)
        font.setPointSize(12)
        self.min_ppm_value.setFont(font)
        self.min_ppm_value.setGeometry(QRect(650,475,175,25))
        self.min_ppm_value.setObjectName("min_ppm_value")
        def setMinPPM():
            global min_ppm
            if(current_ppm < min_ppm):
                min_ppm = current_ppm
            self.min_ppm_value.setText(str(min_ppm) + " PPM")
        schedule.every().seconds.do(setMinPPM)

        #Text for Max PPM Label
        self.max_ppm_label = QLabel(self.scrollArea_ppm)
        font.setPointSize(12)
        self.max_ppm_label.setFont(font)
        self.max_ppm_label.setGeometry(QRect(550,500,175,25))
        self.max_ppm_label.setObjectName("max_ppm_label")
        self.max_ppm_label.setText("Max AQI")

        #Text for Max PPM Value
        self.max_ppm_value = QLabel(self.scrollArea_ppm)
        font.setPointSize(12)
        self.max_ppm_value.setFont(font)
        self.max_ppm_value.setGeometry(QRect(650,500,175,25))
        self.max_ppm_value.setObjectName("max_ppm_value")
        def setMaxPPM():
            global max_ppm
            if(current_ppm > max_ppm):
                max_ppm = current_ppm
            self.max_ppm_value.setText(str(max_ppm) + " PPM")
        schedule.every().seconds.do(setMaxPPM)

        #Text for Average PPM Label
        self.avg_ppm_label = QLabel(self.scrollArea_ppm)
        font.setPointSize(12)
        self.avg_ppm_label.setFont(font)
        self.avg_ppm_label.setGeometry(QRect(550,525,175,25))
        self.avg_ppm_label.setObjectName("avg_ppm_label")
        self.avg_ppm_label.setText("Avg. AQI")

        #Text for Avg PPM Value
        self.avg_ppm_value = QLabel(self.scrollArea_ppm)
        font.setPointSize(12)
        self.avg_ppm_value.setFont(font)
        self.avg_ppm_value.setGeometry(QRect(650,525,175,25))
        self.avg_ppm_value.setObjectName("avg_ppm_value")
        def setAvgPPM():
            global avg_ppm, iterations
            avg_ppm = (avg_ppm*iterations + current_ppm)/(iterations+1)
            iterations = iterations + 1
            self.avg_ppm_value.setText(str(round(avg_ppm,2)) +  " PPM")
        schedule.every().seconds.do(setAvgPPM)

        #Plot Figure for ppm
        self.plot_ppm_widget = QWidget(self.scrollArea_ppm)
        self.plot_ppm_widget.setGeometry(QRect(-50,75,600, 400))
        self.plot_ppm_stuff = FigureCanvas(Figure(figsize=(2,2)))
        self.plot_widget_ppm = QVBoxLayout(self.plot_ppm_widget)
        self.plot_widget_ppm.addWidget(self.plot_ppm_stuff)
        self.plot_ppm_stuff_ax = self.plot_ppm_stuff.figure.subplots()

        #Plot attributes for ppm
        #self.plot_temp_stuff_ax.set_yticklabels([])
        self.plot_ppm_stuff_ax.set_xticklabels([])
        self.plot_ppm_stuff_ax.tick_params(length=0)
        self.plot_ppm_stuff_ax.grid()
        self.plot_ppm_stuff_ax.set_xlim([-60,0])
        #Set up a Line2D.
        self._line_ppm, = self.plot_ppm_stuff_ax.plot(t, ys_np_ppm)
        self._timer_ppm = self.plot_ppm_stuff.new_timer(1000)
        self._timer_ppm.add_callback(self._update_canvas_ppm)
        self._timer_ppm.start()
        #endregion

        #region Oxygen Pressure Screen on Right Side Monitor
        self.scrollArea_oxy_press = QScrollArea(self.oxy_press_monitor)
        self.scrollArea_oxy_press.setGeometry(QRect(0,0,960,900))
        self.current_oxy_press_label = QLabel(self.scrollArea_oxy_press)
        self.current_oxy_press_label.setGeometry(QRect(20, 30, 250, 40))
        self.current_oxy_press_label.setObjectName("current_oxy_press_label")
        font.setPointSize(25)
        self.current_oxy_press_label.setFont(font)
        self.current_oxy_press_label.setText("Oxygen Pressure")
        self.current_oxy_press_value = QLabel(self.scrollArea_oxy_press)
        self.current_oxy_press_value.setGeometry(QRect(25,480,150,100))
        self.current_oxy_press_value.setObjectName("current_oxy_press_value")
        self.current_oxy_press_value.setFont(font)
        def getCurrentOxygenValue():
            self.current_oxy_press_value.setText(str(round(current_oxy_press,2)) + " PSIa")

        #Text for Current Oxygen Pressure Label
        self.current_oxy_press_value_top = QLabel(self.scrollArea_oxy_press)
        font.setPointSize(12)
        self.current_oxy_press_value_top.setFont(font)
        self.current_oxy_press_value_top.setGeometry(QRect(25,475,175,25))
        self.current_oxy_press_value_top.setObjectName("current_oxy_press_value_top")
        self.current_oxy_press_value_top.setText("Current Oxygen Pressure")
        schedule.every().seconds.do(getCurrentOxygenValue)

        #Text for Min Oxygen Pressure Label
        self.min_oxy_press_label = QLabel(self.scrollArea_oxy_press)
        font.setPointSize(12)
        self.min_oxy_press_label.setFont(font)
        self.min_oxy_press_label.setGeometry(QRect(550,475,175,25))
        self.min_oxy_press_label.setObjectName("min_oxy_press_label")
        self.min_oxy_press_label.setText("Min. Oxygen Pressure")

        #Text for Min Oxygen Pressure Value
        self.min_oxy_press_value = QLabel(self.scrollArea_oxy_press)
        font.setPointSize(12)
        self.min_oxy_press_value.setFont(font)
        self.min_oxy_press_value.setGeometry(QRect(700,475,175,25))
        self.min_oxy_press_value.setObjectName("min_oxy_press_value")
        def setMinOxy():
            global min_oxy_press
            if(current_oxy_press < min_oxy_press):
                min_oxy_press = current_oxy_press
            self.min_oxy_press_value.setText(str(min_oxy_press) + " PSIa")
        schedule.every().seconds.do(setMinOxy)

        #Text for Max Oxygen Pressure Label
        self.max_oxy_press_label = QLabel(self.scrollArea_oxy_press)
        font.setPointSize(12)
        self.max_oxy_press_label.setFont(font)
        self.max_oxy_press_label.setGeometry(QRect(550,500,175,25))
        self.max_oxy_press_label.setObjectName("max_oxy_press_label")
        self.max_oxy_press_label.setText("Max Oxygen Pressure")

        #Text for Max Oxygen Pressure Value
        self.max_oxy_press_value = QLabel(self.scrollArea_oxy_press)
        font.setPointSize(12)
        self.max_oxy_press_value.setFont(font)
        self.max_oxy_press_value.setGeometry(QRect(700,500,175,25))
        self.max_oxy_press_value.setObjectName("max_oxy_press_value")
        def setMaxOxy():
            global max_oxy_press
            if(current_oxy_press > max_oxy_press):
                max_oxy_press = current_oxy_press
            self.max_oxy_press_value.setText(str(max_oxy_press) + " PSIa")
        schedule.every().seconds.do(setMaxOxy)

        #Text for Average Oxygen Pressure Label
        self.avg_oxy_press_label = QLabel(self.scrollArea_oxy_press)
        font.setPointSize(12)
        self.avg_oxy_press_label.setFont(font)
        self.avg_oxy_press_label.setGeometry(QRect(550,525,175,25))
        self.avg_oxy_press_label.setObjectName("avg_oxy_press_label")
        self.avg_oxy_press_label.setText("Avg. Oxygen Pressure")

        #Text for Avg Oxygen Pressure Value
        self.avg_oxy_press_value = QLabel(self.scrollArea_oxy_press)
        font.setPointSize(12)
        self.avg_oxy_press_value.setFont(font)
        self.avg_oxy_press_value.setGeometry(QRect(700,525,175,25))
        self.avg_oxy_press_value.setObjectName("avg_oxy_press_value")
        def setAvgOxy():
            global avg_oxy_press, iterations
            avg_oxy_press = (avg_oxy_press*iterations + current_oxy_press)/(iterations+1)
            iterations = iterations + 1
            self.avg_oxy_press_value.setText(str(round(avg_oxy_press,2)) + " PSIa")
        schedule.every().seconds.do(setAvgOxy)

        #Plot Figure for Oxygen Pressure
        self.plot_oxy_press_widget = QWidget(self.scrollArea_oxy_press)
        self.plot_oxy_press_widget.setGeometry(QRect(-50,75,600, 400))
        self.plot_oxy_press_stuff = FigureCanvas(Figure(figsize=(2,2)))
        self.plot_widget_oxy_press = QVBoxLayout(self.plot_oxy_press_widget)
        self.plot_widget_oxy_press.addWidget(self.plot_oxy_press_stuff)
        self.plot_oxy_press_stuff_ax = self.plot_oxy_press_stuff.figure.subplots()

        #Plot attributes for Oxygen Pressure
        #self.plot_temp_stuff_ax.set_yticklabels([])
        self.plot_oxy_press_stuff_ax.set_xticklabels([])
        self.plot_oxy_press_stuff_ax.tick_params(length=0)
        self.plot_oxy_press_stuff_ax.grid()
        self.plot_oxy_press_stuff_ax.set_xlim([-60,0])
        #Set up a Line2D.
        self._line_oxy_press, = self.plot_oxy_press_stuff_ax.plot(t, ys_np_oxy_press)
        self._timer_oxy_press = self.plot_oxy_press_stuff.new_timer(1000)
        self._timer_oxy_press.add_callback(self._update_canvas_oxy_press)
        self._timer_oxy_press.start()
        #endregion
        
        #region Nitrogen Pressure Screen on Right Side Monitor
        self.scrollArea_nit_press = QScrollArea(self.nit_press_monitor)
        self.scrollArea_nit_press.setGeometry(QRect(0,0,960,900))
        self.current_nit_press_label = QLabel(self.scrollArea_nit_press)
        self.current_nit_press_label.setGeometry(QRect(20, 30, 250, 40))
        self.current_nit_press_label.setObjectName("current_nit_press_label")
        font.setPointSize(25)
        self.current_nit_press_label.setFont(font)
        self.current_nit_press_label.setText("Nitrogen Pressure (PSIa)")
        self.current_nit_press_value = QLabel(self.scrollArea_nit_press)
        self.current_nit_press_value.setGeometry(QRect(25,480,150,100))
        self.current_nit_press_value.setObjectName("current_nit_press_value")
        self.current_nit_press_value.setFont(font)
        def getCurrentNitrogenValue():
            self.current_nit_press_value.setText(str(round(current_nit_press,2)) + " PSIa")

        #Text for Current Nitrogen Pressure Label
        self.current_nit_press_value_top = QLabel(self.scrollArea_nit_press)
        font.setPointSize(12)
        self.current_nit_press_value_top.setFont(font)
        self.current_nit_press_value_top.setGeometry(QRect(25,475,175,25))
        self.current_nit_press_value_top.setObjectName("current_nit_press_value_top")
        self.current_nit_press_value_top.setText("Current Nitrogen Pressure")
        schedule.every().seconds.do(getCurrentNitrogenValue)

        #Text for Min Nitrogen Pressure Label
        self.min_nit_press_label = QLabel(self.scrollArea_nit_press)
        font.setPointSize(12)
        self.min_nit_press_label.setFont(font)
        self.min_nit_press_label.setGeometry(QRect(550,475,175,25))
        self.min_nit_press_label.setObjectName("min_nit_press_label")
        self.min_nit_press_label.setText("Min. Nitrogen Pressure")

        #Text for Min Nitrogen Pressure Value
        self.min_nit_press_value = QLabel(self.scrollArea_nit_press)
        font.setPointSize(12)
        self.min_nit_press_value.setFont(font)
        self.min_nit_press_value.setGeometry(QRect(725,475,175,25))
        self.min_nit_press_value.setObjectName("min_nit_press_value")
        def setMinNit():
            global min_nit_press
            if(current_nit_press < min_nit_press):
                min_nit_press = current_nit_press
            self.min_nit_press_value.setText(str(min_nit_press) + " PSIa")
        schedule.every().seconds.do(setMinNit)

        #Text for Max Nitrogen Pressure Label
        self.max_nit_press_label = QLabel(self.scrollArea_nit_press)
        font.setPointSize(12)
        self.max_nit_press_label.setFont(font)
        self.max_nit_press_label.setGeometry(QRect(550,500,175,25))
        self.max_nit_press_label.setObjectName("max_nit_press_label")
        self.max_nit_press_label.setText("Max Nitrogen Pressure")

        #Text for Max Nitrogen Pressure Value
        self.max_nit_press_value = QLabel(self.scrollArea_nit_press)
        font.setPointSize(12)
        self.max_nit_press_value.setFont(font)
        self.max_nit_press_value.setGeometry(QRect(725,500,175,25))
        self.max_nit_press_value.setObjectName("max_nit_press_value")
        def setMaxNit():
            global max_nit_press
            if(current_nit_press > max_nit_press):
                max_nit_press = current_nit_press
            self.max_nit_press_value.setText(str(max_nit_press) + " PSIa")
        schedule.every().seconds.do(setMaxNit)

        #Text for Average Nitrogen Pressure Label
        self.avg_nit_press_label = QLabel(self.scrollArea_nit_press)
        font.setPointSize(12)
        self.avg_nit_press_label.setFont(font)
        self.avg_nit_press_label.setGeometry(QRect(550,525,175,25))
        self.avg_nit_press_label.setObjectName("avg_nit_press_label")
        self.avg_nit_press_label.setText("Avg. Nitrogen Pressure")

        #Text for Avg Nitrogen Pressure Value
        self.avg_nit_press_value = QLabel(self.scrollArea_nit_press)
        font.setPointSize(12)
        self.avg_nit_press_value.setFont(font)
        self.avg_nit_press_value.setGeometry(QRect(725,525,175,25))
        self.avg_nit_press_value.setObjectName("avg_nit_press_value")
        def setAvgNit():
            global avg_nit_press, iterations
            avg_nit_press = (avg_nit_press*iterations + current_nit_press)/(iterations+1)
            iterations = iterations + 1
            self.avg_nit_press_value.setText(str(round(avg_nit_press,2)) + " PSIa")
        schedule.every().seconds.do(setAvgNit)


        #Plot Figure for Nitrogen Pressure
        self.plot_nit_press_widget = QWidget(self.scrollArea_nit_press)
        self.plot_nit_press_widget.setGeometry(QRect(-50,75,600, 400))
        self.plot_nit_press_stuff = FigureCanvas(Figure(figsize=(2,2)))
        self.plot_widget_nit_press = QVBoxLayout(self.plot_nit_press_widget)
        self.plot_widget_nit_press.addWidget(self.plot_nit_press_stuff)
        self.plot_nit_press_stuff_ax = self.plot_nit_press_stuff.figure.subplots()

        #Plot attributes for Nitrogen Pressure
        #self.plot_temp_stuff_ax.set_yticklabels([])
        self.plot_nit_press_stuff_ax.set_xticklabels([])
        self.plot_nit_press_stuff_ax.tick_params(length=0)
        self.plot_nit_press_stuff_ax.grid()
        self.plot_nit_press_stuff_ax.set_xlim([-60,0])
        #Set up a Line2D.
        self._line_nit_press, = self.plot_nit_press_stuff_ax.plot(t, ys_np_nit_press)
        self._timer_nit_press = self.plot_nit_press_stuff.new_timer(1000)
        self._timer_nit_press.add_callback(self._update_canvas_nit_press)
        self._timer_nit_press.start()
        #endregion

        #region Stress Screen on Right Side Monitor
        self.scrollArea_stress = QScrollArea(self.stress_monitor)
        self.scrollArea_stress.setGeometry(QRect(0,0,960,900))
        self.current_stress_label = QLabel(self.scrollArea_stress)
        self.current_stress_label.setGeometry(QRect(20, 30, 250, 40))
        self.current_stress_label.setObjectName("current_stress_label")
        font.setPointSize(25)
        self.current_stress_label.setFont(font)
        self.current_stress_label.setText("Stress (%)")
        self.current_stress_value = QLabel(self.scrollArea_stress)
        self.current_stress_value.setGeometry(QRect(25,480,100,100))
        self.current_stress_value.setObjectName("current_stress_value")
        self.current_stress_value.setFont(font)
        def getCurrentStressValue():
            self.current_stress_value.setText(str(round(current_stress,2)) + "%")

        #Text for Current Stress Pressure Label
        self.current_stress_value_top = QLabel(self.scrollArea_stress)
        font.setPointSize(12)
        self.current_stress_value_top.setFont(font)
        self.current_stress_value_top.setGeometry(QRect(25,475,175,25))
        self.current_stress_value_top.setObjectName("current_stress_value_top")
        self.current_stress_value_top.setText("Current Stress")
        schedule.every().seconds.do(getCurrentStressValue)

        #Text for Min Stress Pressure Label
        self.min_stress_label = QLabel(self.scrollArea_stress)
        font.setPointSize(12)
        self.min_stress_label.setFont(font)
        self.min_stress_label.setGeometry(QRect(550,475,175,25))
        self.min_stress_label.setObjectName("min_stress_label")
        self.min_stress_label.setText("Min. Stress")

        #Text for Min Stress Pressure Value
        self.min_stress_value = QLabel(self.scrollArea_stress)
        font.setPointSize(12)
        self.min_stress_value.setFont(font)
        self.min_stress_value.setGeometry(QRect(650,475,175,25))
        self.min_stress_value.setObjectName("min_stress_value")
        def setMinStress():
            global min_stress
            if(current_stress < min_stress):
                min_stress = current_stress
            self.min_stress_value.setText(str(min_stress) + "%")
        schedule.every().seconds.do(setMinStress)

        #Text for Max Stress Pressure Label
        self.max_stress_label = QLabel(self.scrollArea_stress)
        font.setPointSize(12)
        self.max_stress_label.setFont(font)
        self.max_stress_label.setGeometry(QRect(550,500,175,25))
        self.max_stress_label.setObjectName("max_stress_label")
        self.max_stress_label.setText("Max Stress")

        #Text for Max Stress Pressure Value
        self.max_stress_value = QLabel(self.scrollArea_stress)
        font.setPointSize(12)
        self.max_stress_value.setFont(font)
        self.max_stress_value.setGeometry(QRect(650,500,175,25))
        self.max_stress_value.setObjectName("max_stress_value")
        def setMaxStress():
            global max_stress
            if(current_stress > max_stress):
                max_stress = current_stress
            self.max_stress_value.setText(str(max_stress) + "%")
        schedule.every().seconds.do(setMaxStress)

        #Text for Average Stress Pressure Label
        self.avg_stress_label = QLabel(self.scrollArea_stress)
        font.setPointSize(12)
        self.avg_stress_label.setFont(font)
        self.avg_stress_label.setGeometry(QRect(550,525,175,25))
        self.avg_stress_label.setObjectName("avg_stress_label")
        self.avg_stress_label.setText("Avg. Stress")

        #Text for Avg Stress Pressure Value
        self.avg_stress_value = QLabel(self.scrollArea_stress)
        font.setPointSize(12)
        self.avg_stress_value.setFont(font)
        self.avg_stress_value.setGeometry(QRect(650,525,175,25))
        self.avg_stress_value.setObjectName("avg_stress_value")
        def setAvgStress():
            global avg_stress, iterations
            avg_stress = (avg_stress*iterations + current_stress)/(iterations+1)
            iterations = iterations + 1
            self.avg_stress_value.setText(str(round(avg_stress,2)) + "%")
        schedule.every().seconds.do(setAvgStress)


        #Plot Figure for Stress
        self.plot_stress_widget = QWidget(self.scrollArea_stress)
        self.plot_stress_widget.setGeometry(QRect(-50,75,600, 400))
        self.plot_stress_stuff = FigureCanvas(Figure(figsize=(2,2)))
        self.plot_widget_stress = QVBoxLayout(self.plot_stress_widget)
        self.plot_widget_stress.addWidget(self.plot_stress_stuff)
        self.plot_stress_stuff_ax = self.plot_stress_stuff.figure.subplots()

        #Plot attributes for Stress
        #self.plot_temp_stuff_ax.set_yticklabels([])
        self.plot_stress_stuff_ax.set_xticklabels([])
        self.plot_stress_stuff_ax.tick_params(length=0)
        self.plot_stress_stuff_ax.grid()
        self.plot_stress_stuff_ax.set_xlim([-60,0])
        #Set up a Line2D.
        self._line_stress, = self.plot_stress_stuff_ax.plot(t, ys_np_stress)
        self._timer_stress = self.plot_stress_stuff.new_timer(1000)
        self._timer_stress.add_callback(self._update_canvas_stress)
        self._timer_stress.start()
        #endregion
        #endregion
        
        #endregion
        #region ======================Web Browsing=============================
        #Set Base Layer for Web Browsing
        self.web_base = QWidget(self.web_screen)
        self.web_base.setGeometry(QRect(0,0,1280,645))

        self.web_base.setStyleSheet("background-color: rgb(255,255,255);")
        self.web_text = QLabel(self.web_base)
        self.web_text.setText("Web")
        #endregion
        #region ========================Systems================================
        #Set Base Layer for Web Browsing
        self.systems_base = QWidget(self.systems_screen)
        self.systems_base.setGeometry(QRect(0,0,1280,645))

        self.systems_base.setStyle(QStyleFactory.create("Fusion"))
        self.systems_base.setStyleSheet("background-color: rgb(255,255,255);")
        self.systems_base.setGeometry(QRect(0,0,1280,645))

        self.tabWidget = QTabWidget(self.systems_base)
        font.setPointSize(11)
        self.tabWidget.setFont(font)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("QTabBar:tab{\n""height: 80px;\n""width:182.33px;\n""}")
        self.tabWidget.setGeometry(QRect(0,0,1280,645))
        self.tabWidget.setElideMode(Qt.ElideMiddle)
        self.tabWidget.setTabPosition(QTabWidget.North)
        self.tabWidget.setTabShape(QTabWidget.Rounded)
        self.tabWidget.setUsesScrollButtons(True)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")

        self.Power_tab = QWidget()
        self.Air_tab = QWidget()
        self.Heating_tab = QWidget()
        self.AC_tab = QWidget()
        self.Connections_tab = QWidget()
        self.Oxygen_tab = QWidget()
        self.Nitrogen_tab = QWidget()

        self.tabWidget.addTab(self.Power_tab,"Power")
        self.tabWidget.addTab(self.Air_tab,"Air Filtration")
        self.tabWidget.addTab(self.Heating_tab,"Heating")
        self.tabWidget.addTab(self.AC_tab,"A/C")
        self.tabWidget.addTab(self.Connections_tab, "Connections")
        self.tabWidget.addTab(self.Oxygen_tab,"Oxygen Production")
        self.tabWidget.addTab(self.Nitrogen_tab,"Nitrogen Production")
        #region Labeling and Adding Tabs
        
        #endregion

        #endregion
        #region ===========================Settings============================
        self.settings_base = QWidget(self.settings_screen)
        self.settings_base.setGeometry(QRect(0,0,1280,645))
        self.settings_base.setStyleSheet("background-color: rgb(255,255,255);")
        self.settings_text = QLabel(self.settings_base)
        self.settings_text.setText("User Settings")
        #endregion
        #region =======================Stress Relieve==========================
        self.stress_relief_base = QWidget(self.stress_relief_screen)
        self.stress_relief_base.setGeometry(QRect(0,0,1280,645))
        self.stress_relief_base.setStyleSheet("background-color: rgb(255,255,255);")
        self.stress_relief_text = QLabel(self.stress_relief_base)
        self.stress_relief_text.setText("Stress Relieve")
        #endregion
        #region ========================Communications=========================
        self.communications_base = QWidget(self.communications_screen)
        self.communications_base.setGeometry(QRect(0,0,1280,645))
        self.communications_base.setStyleSheet("background-color: rgb(255,255,255);")
        self.communications_text = QLabel(self.communications_base)
        self.communications_text.setText("Communications")
        #endregion
        #region =============================Maps==============================
        self.maps_base = QWidget(self.maps_screen)
        self.maps_base.setGeometry(QRect(0,0,1280,645))
        self.maps_base.setStyleSheet("background-color: rgb(255,255,255);")
        self.maps_text = QLabel(self.maps_base)
        self.maps_text.setText("Maps")
        #endregion
        #region =============================Eject=============================
        self.eject_base = QWidget(self.eject_screen)
        self.eject_base.setGeometry(QRect(0,0,1280,645))
        self.eject_base.setStyleSheet("background-color: rgb(255,255,255);")
        self.eject_text = QLabel(self.eject_base)
        self.eject_text.setText("Eject")
        #endregion
        def swap_to_monitor():
            self.stacked_widget.setCurrentIndex(2)
            self.home_label.setText("System Monitoring")
        
        def swap_to_web():
            self.stacked_widget.setCurrentIndex(3)
            self.home_label.setText("Web Browser")

        def swap_to_systems():
            self.stacked_widget.setCurrentIndex(4)
            self.home_label.setText("Systems")

        def swap_to_settings():
            self.stacked_widget.setCurrentIndex(5)
            self.home_label.setText("Settings")

        def swap_to_stress_relief():
            self.stacked_widget.setCurrentIndex(6)
            self.home_label.setText("Stress Relieve")

        def swap_to_communications():
            self.stacked_widget.setCurrentIndex(7)
            self.home_label.setText("Communications")
        
        def swap_to_maps():
            self.stacked_widget.setCurrentIndex(8)
            self.home_label.setText("Maps")

        def swap_to_eject():
            self.stacked_widget.setCurrentIndex(9)
            self.home_label.setText("Prepare for Ejection")

        #region Button Stuff in All Applications
        self.app_1.setText("Web Browser")
        self.app_1.clicked.connect(swap_to_web)
        self.app_5.setText("Stress Relieve")
        self.app_5.clicked.connect(swap_to_stress_relief)
        self.app_9.setText("PushButton9")
        self.app_9.setEnabled(False)
        self.app_2.setText("Monitoring")
        self.app_2.clicked.connect(swap_to_monitor)
        self.app_6.setText("Communications")
        self.app_6.clicked.connect(swap_to_communications)
        self.app_3.setText("Systems")
        self.app_3.clicked.connect(swap_to_systems)
        self.app_7.setText("Maps")
        self.app_7.clicked.connect(swap_to_maps)
        self.app_4.setText("Settings")
        self.app_4.clicked.connect(swap_to_settings)
        self.app_8.setText("Eject")
        self.app_8.clicked.connect(swap_to_eject)
        #endregion



        def update_Suggested_Apps():

            global predict_Grouping, suggested_app_2_value

            if(predict_Grouping != -1):
                if(predict_Grouping[0] == 0):
                    print("apps 0")
                    self.suggested_app_1.setEnabled(False)
                    self.suggested_app_2.setText("Eject")
                    suggested_app_2_value = 8
                    self.suggested_app_3.setEnabled(False)
                elif(predict_Grouping[0] == 1):
                    print("apps 1")
                elif(predict_Grouping[0] == 2):
                    print("apps 2")
                elif(predict_Grouping[0] == 3):
                    print("apps 3")
                elif(predict_Grouping[0] == 4):
                    print("apps 4")
                elif(predict_Grouping[0] == 5):
                    print("apps 5")
                    suggested_app_2_value = 2


        def swap_to_what():
            global suggested_app_2_value
            if(suggested_app_2_value == 2):
                self.stacked_widget.setCurrentIndex(2)
                self.home_label.setText("System Monitoring")
            elif(suggested_app_2_value == 8):
                self.stacked_widget.setCurrentIndex(9)
                self.home_label.setText("Prepare for Ejection")
        self.suggested_app_2.clicked.connect(swap_to_what)

            
        schedule.every().seconds.do(update_Suggested_Apps)
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()




    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 360, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    


def on_press(key):
    a = threading.Thread(target = increase_temp_module, daemon = True)
    b = threading.Thread(target = decrease_temp_module, daemon = True)
    c = threading.Thread(target = return_to_normal_temp_module, daemon = True)
    d = threading.Thread(target = increase_co2_module, daemon = True)
    e = threading.Thread(target = return_to_normal_co2_module, daemon = True)
    f = threading.Thread(target = increase_ppm_module, daemon = True)
    g = threading.Thread(target = return_to_normal_ppm_module, daemon = True)
    h = threading.Thread(target = increase_stress_module, daemon = True)
    i = threading.Thread(target = return_to_normal_stress_module, daemon = True)
    j = threading.Thread(target = increase_oxy_press_module, daemon = True)
    k = threading.Thread(target = decrease_oxy_press_module, daemon = True)
    l = threading.Thread(target = return_to_normal_oxy_press_module, daemon = True)
    m = threading.Thread(target = increase_nit_press_module, daemon = True)
    n = threading.Thread(target = decrease_nit_press_module, daemon = True)
    o = threading.Thread(target = return_to_normal_nit_press_module, daemon = True)
    if(key == keyboard.Key.esc):
        print('esc key pressed')
        os._exit(0)
    elif(key.char == '1'):
        a.start()
    elif(key.char == '2'):
        b.start()
    elif(key.char == '3'):
        d.start()
    elif(key.char == '4'):
        f.start()
    elif(key.char == '5'):
        h.start()
    elif(key.char == '6'):
        j.start()
    elif(key.char == '7'):
        k.start()
    elif(key.char == '8'):
        m.start()
    elif(key.char == '9'):
        n.start()
    elif(key.char == '0'):
       c.start()
       e.start()
       g.start()
       i.start()
       l.start()
       o.start()
    else:
        print('Ignored')


if __name__=="__main__":

    #Read in Data
    df = pd.read_csv("C:/Users/Raymond Fey/Desktop/MHIS_PY/ok_datasets/Working_Dataset.csv")

    #Shuffling Data
    df = shuffle(df)

    #Reset Index in Pandas
    df.reset_index(inplace = True, drop = True)

    #Split Data into Train-Test Split
    train, test = train_test_split(df, test_size = 0.25)
    print(len(train), 'train examples')
    print(len(test), 'test examples')

    train_features = train.iloc[:, :-1]
    train_label = train['cluster']

    test_features = test.iloc[:,:-1]
    test_label = test['cluster']

    model = train_DT(train_features, train_label)
    schedule.every(5).seconds.do(predict_Module, model)

    model_predicted_train_Y = model.predict(train_features)
    print(classification_report(train_label,model_predicted_train_Y))

    model_predicted_test_Y = model.predict(test_features)
    print(classification_report(test_label,model_predicted_test_Y))

    app = QApplication(sys.argv)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    a = App()
    a.show()

    sys.exit(app.exec_())
