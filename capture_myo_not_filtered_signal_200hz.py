from __future__ import print_function
from serial.tools.list_ports import comports
from common import *
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_main_window import Ui_MainWindow
from datetime import date
import time
import cv2
import numpy as np
import pose_module as pm
import re
import sys
import threading
import pandas as pd
import serial
import os
import enum

###################################################################################
# Global Variables
actual_angle = None
FIRST_TIME = time.time()
Emg_total = []
end_of_training = False
write_EMG_signals = False
problem_ = False
saved_data = True
end_ = False

# list_of_angles = [170, 90, 165, 150, 135, 120, 105, 75, 60, 45]
list_of_angles = [170, 90, 60, 45]
##############################################################################################
# Myo Modules


##########################################################################################################
# Starts the main code:
"""
    Original by Alan Mendes:
    https://github.com/alans96/arm_robotics.git
    
    Edited by Caio Lima and Alan Mendes:
    https://github.com/Calima94/Capture_EMG_Data.git
    
"""


def multichr(ords):
    if sys.version_info[0] >= 3:
        return bytes(ords)
    else:
        return ''.join(map(chr, ords))


def multiord(b):
    if sys.version_info[0] >= 3:
        return list(b)
    else:
        return map(ord, b)


class Arm(enum.Enum):
    UNKNOWN = 0
    RIGHT = 1
    LEFT = 2


class XDirection(enum.Enum):
    UNKNOWN = 0
    X_TOWARD_WRIST = 1
    X_TOWARD_ELBOW = 2


class Pose(enum.Enum):
    REST = 0
    FIST = 1
    WAVE_IN = 2
    WAVE_OUT = 3
    FINGERS_SPREAD = 4
    THUMB_TO_PINKY = 5
    UNKNOWN = 255


class Packet(object):
    def __init__(self, ords):
        self.typ = ords[0]
        self.cls = ords[2]
        self.cmd = ords[3]
        self.payload = multichr(ords[4:])

    def __repr__(self):
        return 'Packet(%02X, %02X, %02X, [%s])' % \
               (self.typ, self.cls, self.cmd,
                ' '.join('%02X' % b for b in multiord(self.payload)))


class BT(object):
    '''Implements the non-Myo-specific details of the Bluetooth protocol.'''

    def __init__(self, tty):
        self.ser = serial.Serial(port=tty, baudrate=9600, dsrdtr=1)
        self.buf = []
        self.lock = threading.Lock()
        self.handlers = []

    # internal data-handling methods
    def recv_packet(self, timeout=None):
        t0 = time.time()
        self.ser.timeout = None
        while timeout is None or time.time() < t0 + timeout:
            if timeout is not None:
                self.ser.timeout = t0 + timeout - time.time()
            c = self.ser.read()
            if not c:
                return None

            ret = self.proc_byte(ord(c))
            if ret:
                if ret.typ == 0x80:
                    self.handle_event(ret)
                return ret

    def recv_packets(self, timeout=.5):
        res = []
        t0 = time.time()
        while time.time() < t0 + timeout:
            p = self.recv_packet(t0 + timeout - time.time())
            if not p:
                return res
            res.append(p)
        return res

    def proc_byte(self, c):
        if not self.buf:
            if c in [0x00, 0x80, 0x08, 0x88]:  # [BLE response pkt, BLE event pkt, wifi response pkt, wifi event pkt]
                self.buf.append(c)
            return None
        elif len(self.buf) == 1:
            self.buf.append(c)
            self.packet_len = 4 + (self.buf[0] & 0x07) + self.buf[1]
            return None
        else:
            self.buf.append(c)

        if self.packet_len and len(self.buf) == self.packet_len:
            p = Packet(self.buf)
            self.buf = []
            return p
        return None

    def handle_event(self, p):
        for h in self.handlers:
            h(p)

    def add_handler(self, h):
        self.handlers.append(h)

    def remove_handler(self, h):
        try:
            self.handlers.remove(h)
        except ValueError:
            pass

    def wait_event(self, cls, cmd):
        res = [None]

        def h(p):
            if p.cls == cls and p.cmd == cmd:
                res[0] = p

        self.add_handler(h)
        while res[0] is None:
            self.recv_packet()
        self.remove_handler(h)
        return res[0]

    # specific BLE commands
    def connect(self, addr):
        return self.send_command(6, 3, pack('6sBHHHH', multichr(addr), 0, 6, 6, 64, 0))

    def get_connections(self):
        return self.send_command(0, 6)

    def discover(self):
        return self.send_command(6, 2, b'\x01')

    def end_scan(self):
        return self.send_command(6, 4)

    def disconnect(self, h):
        return self.send_command(3, 0, pack('B', h))

    def read_attr(self, con, attr):
        self.send_command(4, 4, pack('BH', con, attr))
        return self.wait_event(4, 5)

    def write_attr(self, con, attr, val):
        self.send_command(4, 5, pack('BHB', con, attr, len(val)) + val)
        return self.wait_event(4, 1)

    def send_command(self, cls, cmd, payload=b'', wait_resp=True):
        s = pack('4B', 0, len(payload), cls, cmd) + payload
        self.ser.write(s)

        while True:
            p = self.recv_packet()
            # no timeout, so p won't be None
            if p.typ == 0:
                return p
            # not a response: must be an event
            self.handle_event(p)


class MyoRaw(object):
    """
    Implements the Myo-specific communication protocol.
    """

    def __init__(self, tty=None):
        if tty is None:
            tty = self.detect_tty()
        if tty is None:
            raise ValueError('Myo dongle not found!')

        self.bt = BT(tty)
        self.conn = None
        self.emg_handlers = []
        self.imu_handlers = []
        self.arm_handlers = []
        self.pose_handlers = []
        self.battery_handlers = []

    def detect_tty(self):
        for p in comports():
            if re.search(r'PID=2458:0*1', p[2]):
                print('using device:', p[0])
                return p[0]

        return None

    def run(self, timeout=None):
        self.bt.recv_packet(timeout)

    def connect(self):
        # stop everything from before
        self.bt.end_scan()
        self.bt.disconnect(0)
        self.bt.disconnect(1)
        self.bt.disconnect(2)

        # start scanning
        print('scanning...')
        self.bt.discover()
        while True:
            p = self.bt.recv_packet()
            print('scan response:', p)

            if p.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                addr = list(multiord(p.payload[2:8]))
                break
        self.bt.end_scan()

        # connect and wait for status event
        conn_pkt = self.bt.connect(addr)
        self.conn = multiord(conn_pkt.payload)[-1]
        self.bt.wait_event(3, 0)

        # get firmware version
        fw = self.read_attr(0x17)
        _, _, _, _, v0, v1, v2, v3 = unpack('BHBBHHHH', fw.payload)
        print('firmware version: %d.%d.%d.%d' % (v0, v1, v2, v3))

        self.old = (v0 == 0)

        if self.old:
            # don't know what these do; Myo Connect sends them, though we get data
            # fine without them
            self.write_attr(0x19, b'\x01\x02\x00\x00')
            # Subscribe for notifications from 4 EMG data channels
            self.write_attr(0x2f, b'\x01\x00')
            self.write_attr(0x2c, b'\x01\x00')
            self.write_attr(0x32, b'\x01\x00')
            self.write_attr(0x35, b'\x01\x00')

            # enable EMG data
            self.write_attr(0x28, b'\x01\x00')
            # enable IMU data
            self.write_attr(0x1d, b'\x01\x00')

            # Sampling rate of the underlying EMG sensor, capped to 1000. If it's
            # less than 1000, emg_hz is correct. If it is greater, the actual
            # framerate starts dropping inversely. Also, if this is much less than
            # 1000, EMG data becomes slower to respond to changes. In conclusion,
            # 1000 is probably a good value.
            C = 1000
            emg_hz = 50
            # strength of low-pass filtering of EMG data
            emg_smooth = 100

            imu_hz = 50

            # send sensor parameters, or we don't get any data
            self.write_attr(0x19, pack('BBBBHBBBBB', 2, 9, 2, 1, C, emg_smooth, C // emg_hz, imu_hz, 0, 0))

        else:
            name = self.read_attr(0x03)
            print('device name: %s' % name.payload)

            # enable IMU data
            self.write_attr(0x1d, b'\x01\x00')
            # enable on/off arm notifications
            self.write_attr(0x24, b'\x02\x00')
            # enable EMG notifications
            self.start_raw()
            # enable battery notifications
            self.write_attr(0x12, b'\x01\x10')

        # add data handlers
        def handle_data(p):
            if (p.cls, p.cmd) != (4, 5):
                return

            c, attr, typ = unpack('BHB', p.payload[:4])
            pay = p.payload[5:]

            if attr == 0x27:
                # Unpack a 17 byte array, first 16 are 8 unsigned shorts, last one an unsigned char
                vals = unpack('8HB', pay)
                # not entirely sure what the last byte is, but it's a bitmask that
                # seems to indicate which sensors think they're being moved around or
                # something
                emg = vals[:8]
                moving = vals[8]
                self.on_emg(emg, moving)
            # Read notification handles corresponding to the for EMG characteristics
            elif attr == 0x2b or attr == 0x2e or attr == 0x31 or attr == 0x34:
                '''According to http://developerblog.myo.com/myocraft-emg-in-the-bluetooth-protocol/
                each characteristic sends two sequential readings in each update,
                so the received payload is split in two samples. According to the
                Myo BLE specification, the data type of the EMG samples is int8_t.
                '''
                emg1 = struct.unpack('<8b', pay[:8])
                emg2 = struct.unpack('<8b', pay[8:])
                self.on_emg(emg1, 0)
                self.on_emg(emg2, 0)
            # Read IMU characteristic handle
            elif attr == 0x1c:
                vals = unpack('10h', pay)
                quat = vals[:4]
                acc = vals[4:7]
                gyro = vals[7:10]
                self.on_imu(quat, acc, gyro)
            # Read classifier characteristic handle
            elif attr == 0x23:
                typ, val, xdir, _, _, _ = unpack('6B', pay)

                if typ == 1:  # on arm
                    self.on_arm(Arm(val), XDirection(xdir))
                elif typ == 2:  # removed from arm
                    self.on_arm(Arm.UNKNOWN, XDirection.UNKNOWN)
                elif typ == 3:  # pose
                    self.on_pose(Pose(val))
            # Read battery characteristic handle
            elif attr == 0x11:
                battery_level = ord(pay)
                self.on_battery(battery_level)
            else:
                print('data with unknown attr: %02X %s' % (attr, p))

        self.bt.add_handler(handle_data)

    def write_attr(self, attr, val):
        if self.conn is not None:
            self.bt.write_attr(self.conn, attr, val)

    def read_attr(self, attr):
        if self.conn is not None:
            return self.bt.read_attr(self.conn, attr)
        return None

    def disconnect(self):
        if self.conn is not None:
            self.bt.disconnect(self.conn)

    def sleep_mode(self, mode):
        self.write_attr(0x19, pack('3B', 9, 1, mode))

    def power_off(self):
        self.write_attr(0x19, b'\x04\x00')

    def start_raw(self):

        ''' To get raw EMG signals, we subscribe to the four EMG notification
        characteristics by writing a 0x0100 command to the corresponding handles.
        '''
        self.write_attr(0x2c, b'\x01\x00')  # Suscribe to EmgData0Characteristic
        self.write_attr(0x2f, b'\x01\x00')  # Suscribe to EmgData1Characteristic
        self.write_attr(0x32, b'\x01\x00')  # Suscribe to EmgData2Characteristic
        self.write_attr(0x35, b'\x01\x00')  # Suscribe to EmgData3Characteristic

        '''Bytes sent to handle 0x19 (command characteristic) have the following
        format: [command, payload_size, EMG mode, IMU mode, classifier mode]
        According to the Myo BLE specification, the commands are:
            0x01 -> set EMG and IMU
            0x03 -> 3 bytes of payload
            0x02 -> send 50Hz filtered signals
            0x01 -> send IMU data streams
            0x01 -> send classifier events
        '''
        self.write_attr(0x19, b'\x01\x03\x02\x01\x01')

        '''Sending this sequence for v1.0 firmware seems to enable both raw data and
        pose notifications.
        '''

        '''By writting a 0x0100 command to handle 0x28, some kind of "hidden" EMG
        notification characteristic is activated. This characteristic is not
        listed on the Myo services of the offical BLE specification from Thalmic
        Labs. Also, in the second line where we tell the Myo to enable EMG and
        IMU data streams and classifier events, the 0x01 command wich corresponds
        to the EMG mode is not listed on the myohw_emg_mode_t struct of the Myo
        BLE specification.
        These two lines, besides enabling the IMU and the classifier, enable the
        transmission of a stream of low-pass filtered EMG signals from the eight
        sensor pods of the Myo armband (the "hidden" mode I mentioned above).
        Instead of getting the raw EMG signals, we get rectified and smoothed
        signals, a measure of the amplitude of the EMG (which is useful to have
        a measure of muscle strength, but are not as useful as a truly raw signal).
        '''

        # self.write_attr(0x28, b'\x01\x00')  # Not needed for raw signals
        # self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def mc_start_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when starting data
        collection for v1.0 firmware; this enables raw data but disables arm and
        pose notifications.
        '''

        self.write_attr(0x28, b'\x01\x00')  # Suscribe to EMG notifications
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x24, b'\x02\x00')  # Suscribe to classifier indications
        self.write_attr(0x19,
                        b'\x01\x03\x01\x01\x01')  # Set EMG and IMU, payload size = 3, EMG on, IMU on, classifier on
        self.write_attr(0x28, b'\x01\x00')  # Suscribe to EMG notifications
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x19,
                        b'\x09\x01\x01\x00\x00')  # Set sleep mode, payload size = 1, never go to sleep, don't know, don't know
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x19,
                        b'\x01\x03\x00\x01\x00')  # Set EMG and IMU, payload size = 3, EMG off, IMU on, classifier off
        self.write_attr(0x28, b'\x01\x00')  # Suscribe to EMG notifications
        self.write_attr(0x1d, b'\x01\x00')  # Suscribe to IMU notifications
        self.write_attr(0x19,
                        b'\x01\x03\x01\x01\x00')  # Set EMG and IMU, payload size = 3, EMG on, IMU on, classifier off

    def mc_end_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when ending data collection
        for v1.0 firmware; this reenables arm and pose notifications, but
        doesn't disable raw data.
        '''

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x19, b'\x09\x01\x00\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def vibrate(self, length):
        # first byte tells it to vibrate; purpose of second byte is unknown (payload size?)
        self.write_attr(0x19, pack('3B', 3, 1, length))

    def set_leds(self, logo, line):
        self.write_attr(0x19, pack('8B', 6, 6, *(logo + line)))

    # def get_battery_level(self):
    #     battery_level = self.read_attr(0x11)
    #     return ord(battery_level.payload[5])

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    def add_pose_handler(self, h):
        self.pose_handlers.append(h)

    def add_arm_handler(self, h):
        self.arm_handlers.append(h)

    def add_battery_handler(self, h):
        self.battery_handlers.append(h)

    def on_emg(self, emg, moving):
        for h in self.emg_handlers:
            h(emg, moving)

    def on_imu(self, quat, acc, gyro):
        for h in self.imu_handlers:
            h(quat, acc, gyro)

    def on_pose(self, p):
        for h in self.pose_handlers:
            h(p)

    def on_arm(self, arm, xdir):
        for h in self.arm_handlers:
            h(arm, xdir)

    def on_battery(self, battery_level):
        for h in self.battery_handlers:
            h(battery_level)


##########################################################################################
def main(args=None):
    class MainWindow:
        def __init__(self, ):
            self.main_win = QMainWindow()
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self.main_win)

            # Define which screen will be initialized
            self.ui.Pages.setCurrentWidget(self.ui.page_home)

            # Menu Buttons
            self.ui.btn_contatos.clicked.connect(self.show_contatos)
            self.ui.btn_funcoes.clicked.connect(self.show_funcao)
            self.ui.btn_home.clicked.connect(self.show_home)
            self.ui.btn_sobre.clicked.connect(self.show_sobre)

            # Right or Left Button
            self.ui.btn_right.clicked.connect(self.show_right)
            # self.ui.btn_left.clicked.connect(self.show_left)

            # Start Button
            self.ui.start1.clicked.connect(self.show_btn_1)
            # self.ui.start2.clicked.connect(self.show_btn_2)

        def show(self):
            self.main_win.show()
            ##############################
            # Functions of do Menu Button
            ##############################

        def show_contatos(self):
            self.ui.Pages.setCurrentWidget(self.ui.page_contatos)

        def show_home(self):
            self.ui.Pages.setCurrentWidget(self.ui.page_home)

        def show_funcao(self):
            self.ui.Pages.setCurrentWidget(self.ui.page_funcoes)

        def show_sobre(self):
            self.ui.Pages.setCurrentWidget(self.ui.page_sobre)

        ##########################################
        # Functions of the left and right of the Menu
        ##########################################
        # def show_left(self):
        # self.ui.Pages_start.setCurrentWidget(self.ui.page1)

        def show_right(self):
            self.ui.Pages_start.setCurrentWidget(self.ui.page2)

        #######################################
        # Functions of the Start Button
        #######################################
        def show_btn_1(self):
            threading.Thread(target=write_file).start()
            self.write_image()
            # threading.Thread(target=self.write_image).start()

        def write_image(self):
            global write_EMG_signals
            global actual_angle
            global Emg_total
            global end_of_training

            # Choose the quantity of test cases
            teste = int(self.ui.comboBox_1.currentText())
            teste_init = teste

            # Input the variation of each class
            var = int(self.ui.lineEdit_var_1.text())
            # Input the number of samples in each test
            number_of_samples_in_each_class = int(self.ui.lineEdit_amostra_1.text())

            write_EMG_signals = False
            actual_list_of_angles = list_of_angles
            cap = cv2.VideoCapture(0)  # Image Source
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Video File
            out = cv2.VideoWriter('Videos/test_2_05.avi', fourcc, 20.0, (640, 480))  # Video format
            count = 0
            dir_ = 0
            select_arm = "right"
            angle = None
            detector = pm.PoseDetector(teste=teste, var=var)
            categories_already_trained = []
            while cap.isOpened():
                if detector.teste == 0:
                    break
                # Capture the mirroring of the video
                ret, img = cap.read()
                img = cv2.flip(img, 180)

                if ret is True:
                    # Detect the body and the marks without drawing
                    img = detector.find_pose(img, False)
                    lm_list = detector.find_position(img, False)

                    # List the Markers
                    if len(lm_list) != 0:
                        try:
                            # Right Arm
                            if select_arm == "right":
                                angle = detector.find_angle(img=img,
                                                            p1=11,
                                                            p2=13,
                                                            p3=15,
                                                            test=teste_init,
                                                            list_of_angles=actual_list_of_angles,
                                                            var=detector.var,
                                                            categories_already_trained=categories_already_trained)
                            # Left Arm
                            else:
                                angle = detector.find_angle(img=img,
                                                            p1=12,
                                                            p2=14,
                                                            p3=16,
                                                            test=teste_init,
                                                            list_of_angles=actual_list_of_angles,
                                                            var=detector.var,
                                                            categories_already_trained=categories_already_trained)

                            # Convert the angle between 0° and 100°
                            per = np.interp(angle, (40, 171), (0, 100))
                            # interpol the bar scale
                            bar = np.interp(angle, (40, 171), (400, 100))

                            if per == 0:
                                if dir_ == 0:
                                    count += 0.5
                                    dir_ = 1
                            if per == 100:
                                if dir_ == 1:
                                    count += 0.5
                                    dir_ = 0

                        finally:
                            # Rectangular bar
                            cv2.rectangle(img, (50, 400), (50, 400), (0, 255, 0), 3)
                            # Moving bar
                            cv2.rectangle(img, (50, int(bar)), (60, 400), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, f'{int(per)}%', (55, 300), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 4)
                            # Number of repetitions
                            cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

                            write_EMG_signals, actual_angle = detector.test_cases_to_use(tests_init=teste_init,
                                                                                         angle=angle,
                                                                                         var=detector.var,
                                                                                         angles_to_test=actual_list_of_angles,
                                                                                         categories_already_trained=categories_already_trained)

                            detector.teste, categories_already_trained = detector.check_if_num_samples_is_complete(
                                emg_table=Emg_total,
                                list_of_angles=actual_list_of_angles,
                                num_of_samples_in_each_class=number_of_samples_in_each_class,
                                test=detector.teste,
                                categories_already_trained=categories_already_trained
                            )
                    out.write(img)
                    cv2.imshow("Image", img)
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

                else:
                    print("Webcam not found!")
                    break
            global end_of_training
            end_of_training = True
            cap.release()
            out.release()
            cv2.destroyAllWindows()

    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())


def return_name_file(file_name: str) -> str:
    """
    Return new name of file if the name chosen already existed

    input:
    param_ file_name: Name of the file -> str
    return:
    new_name: New name of the file
    """
    u = 0
    while os.path.exists(f"{file_name}{u}.csv"):
        u += 1
    new_name = f"{file_name}{u}.csv"
    return new_name


def name_columns_of_table(n_columns: int):
    """
    Return name of columns of each column in the EMG data

    input:
    Name of the file -> int

    return:
    columns_name: names of each column stored in a vector

    """
    columns_name = None
    for i in range(n_columns):
        if columns_name is None:
            columns_name = ["time"]
        else:
            if i < n_columns - 1:
                columns_name.append(f"channel{i}")
            else:
                columns_name.append("position")
    return columns_name


def write_file(args=None):
    global write_EMG_signals
    global Emg_total
    global end_of_training
    global problem_
    global saved_data
    global actual_angle
    global end_
    end_of_training = False
    write_EMG_signals = False

    # Start new program here
    ########################################################################################

    last_vals = None
    m = MyoRaw(None)

    def return_name_file(file_name: str) -> str:
        """
        return new name of the file if the names already exists

        input:
        file_name: str -> name of the file

        return:
        new_name: str -> name of new file

        """
        u = 0
        new_name = f"{file_name}{u}.csv"
        while os.path.exists(f"{file_name}{u}.csv"):
            u += 1
            new_name = f"{file_name}{u}.csv"
        return new_name

    def name_columns_of_table(n_columns: int):
        """
        Name of columns of the database

        input:
        n_of_columns: int -> number of the columns of the database

        output:
        columns_name: name of columns save in a vector

        """
        columns_name = None
        for i in range(n_columns):
            if columns_name is None:
                columns_name = ["time"]
            else:
                if i < n_columns - 1:
                    columns_name.append(f"channel{i}")
                else:
                    columns_name.append(f"position")
        return columns_name

    def proc_emg(emg, moving, times=[]):
        """
        Function to save EMG data if the signal is in the correct category
        """

        emg_list_ = list(emg)
        if write_EMG_signals:
            if len(emg_list_) == 8:
                time_now = time.time()
                diff_time = time_now - FIRST_TIME
                emg_list_.insert(0, diff_time)
                emg_list_.insert(9, actual_angle)
                Emg_total.append(emg_list_)

            # print framerate of received data
            # times.append(time.time())
            # if len(times) > 20:
            # print((len(times) - 1) / (times[-1] - times[0]))
            #    times.pop(0)

    def proc_battery(battery_level):
        print("Battery level: %d" % battery_level)
        if battery_level < 5:
            m.set_leds([255, 0, 0], [255, 0, 0])
        else:
            m.set_leds([128, 128, 255], [128, 128, 255])

    m.add_emg_handler(proc_emg)
    # m.add_battery_handler(proc_battery)
    m.connect()

    # m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
    # m.add_pose_handler(lambda p: print('pose', p))
    # m.add_imu_handler(lambda quat, acc, gyro: print('quaternion', quat))
    # m.sleep_mode(1)
    # m.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs
    # m.vibrate(1)

    try:
        while True:
            m.run(1)
            print(len(Emg_total))
            if end_of_training:
                df = pd.DataFrame(Emg_total)
                n_columns = len(Emg_total[0])
                name_of_columns = name_columns_of_table(n_columns=n_columns)
                df.columns = name_of_columns
                today = str(date.today())
                name_ = return_name_file(f"EMG_Data/test_{today}")
                name_2 = return_name_file(f"../Train_Myo_Signals/Raw_EMG_Data/test_{today}")
                df.to_csv(name_, index=False)
                df.to_csv(name_2, index=False)
                print("end")
                break


    except KeyboardInterrupt:
        pass
    finally:
        # m.power_off()
        # print("Power off")
        m.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
