import os, sys, shutil, math
import cv2
import threading
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QPushButton, QWidget, QDialog
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QTableWidget, QTableWidgetItem, QFileDialog, QProgressBar
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5.QtGui import QImage, QPixmap, QFont, QPen, QMovie
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from controller import input_image_handle, load_result, many_to_save

# class EditScreen(QWidget):
class EditScreen(QDialog):
    def __init__(self, parent, load_img, edge, diam):
        super().__init__(parent)
        # self.parent = parent
        self.active_point = None
        self.s_diam = diam
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.view) 
        
        image_data = np.ascontiguousarray(load_img)
        height, width, channels = image_data.shape
        qimage = QImage(image_data.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(-10, -4, 780, 520)
        
        # upper
        self.up_points = [QPointF(y, x) for x, y in edge[0]][::5]
        self.up_point_items = []  
        for point in self.up_points:
            point_item = QGraphicsEllipseItem(-5, -5, 10, 10)
            point_item.setPos(point)
            point_item.setBrush(Qt.red)  # 빨간색으로 설정
            self.scene.addItem(point_item)
            self.up_point_items.append(point_item)
        # lower
        self.down_points = [QPointF(y, x) for x, y in edge[1]][::5]
        self.down_point_items = []  
        for point in self.down_points:
            point_item = QGraphicsEllipseItem(-5, -5, 10, 10)
            point_item.setPos(point)
            point_item.setBrush(Qt.blue)  # 파란색으로 설정
            self.scene.addItem(point_item)
            self.down_point_items.append(point_item)

        self.lines = []  # 추가 점에 대한 선을 저장할 리스트

        self.view.mousePressEvent = self.on_mouse_press
        self.view.mouseMoveEvent = self.on_mouse_move
        self.view.mouseReleaseEvent = self.on_mouse_release
        
        self.coord_label = QLabel(self)
        # self.coord_label.setGeometry(10, 400, 200, 60)
        # self.coord_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        # "Back" 버튼 추가
        self.prev_positions = []
        self.back_button = QPushButton("Back", self)
        self.back_button.setGeometry(550, 260, 150, 30)
        # 550, 170, 300, 80
        self.back_button.clicked.connect(self.restore_previous_positions)
        self.layout().addWidget(self.back_button)
        
    def on_mouse_press(self, event):
        items = self.view.items(event.pos())
        for item in items:
            if item in self.up_point_items or item in self.down_point_items:
                self.active_point = item
                self.save_current_position()
                
    def save_current_position(self):
        if self.active_point:
            pos = self.active_point.scenePos()
            self.prev_positions.append((self.active_point, pos))
            
    def restore_previous_positions(self):
        if self.prev_positions:
            item, pos = self.prev_positions.pop()
            item.setPos(pos)
        self.update_lines()
            
    def on_mouse_move(self, event):
        if self.active_point:
            self.active_point.setPos(self.view.mapToScene(event.pos()))
            self.update_lines()
            self.update_coord_label()

    def on_mouse_release(self, event):
        self.active_point = None

    def update_lines(self):
        for line_item in self.lines:
            self.scene.removeItem(line_item)
        self.lines = []
        for i in range(len(self.up_point_items) - 1):
            start_point = self.up_point_items[i].scenePos()
            end_point = self.up_point_items[i + 1].scenePos()

            if i < len(self.lines):
                self.lines[i].setLine(start_point.x(), start_point.y(), end_point.x(), end_point.y())
            else:
                line_item = QGraphicsLineItem()
                self.scene.addItem(line_item)
                self.lines.append(line_item)

                pen = QPen(Qt.green)  # 초록색으로 설정
                line_item.setPen(pen)
                line_item.setLine(start_point.x(), start_point.y(), end_point.x(), end_point.y())

        for i in range(len(self.down_point_items) - 1):
            start_point = self.down_point_items[i].scenePos()
            end_point = self.down_point_items[i + 1].scenePos()

            if i + len(self.up_point_items) < len(self.lines):
                self.lines[i + len(self.up_point_items)].setLine(start_point.x(), start_point.y(), end_point.x(), end_point.y())
            else:
                line_item = QGraphicsLineItem()
                self.scene.addItem(line_item)
                self.lines.append(line_item)

                pen = QPen(Qt.red)  # 빨간색으로 설정
                line_item.setPen(pen)
                line_item.setLine(start_point.x(), start_point.y(), end_point.x(), end_point.y())

        # up_point 내부의 점들을 선으로 연결
        for i in range(len(self.up_point_items) - 1):
            start_point = self.up_point_items[i].scenePos()
            end_point = self.up_point_items[i + 1].scenePos()
            
            line_item = QGraphicsLineItem()
            self.scene.addItem(line_item)
            self.lines.append(line_item)
            
            pen = QPen(Qt.red)
            line_item.setPen(pen)
            line_item.setLine(start_point.x(), start_point.y(), end_point.x(), end_point.y())
            
        # down_point 내부의 점들을 선으로 연결
        for i in range(len(self.down_point_items) - 1):
            start_point = self.down_point_items[i].scenePos()
            end_point = self.down_point_items[i + 1].scenePos()
            
            line_item = QGraphicsLineItem()
            self.scene.addItem(line_item)
            self.lines.append(line_item)
            
            pen = QPen(Qt.blue)
            line_item.setPen(pen)
            line_item.setLine(start_point.x(), start_point.y(), end_point.x(), end_point.y())

    def update_coord_label(self):
        if self.active_point:
            pos = self.active_point.scenePos()
            coord_text = f"X: {pos.x():.2f}, Y: {pos.y():.2f}"
            
            if self.active_point in self.up_point_items:
                up_total_length = 0.0
                for i in range(len(self.up_point_items) - 1):
                    start_point = self.up_point_items[i].scenePos()
                    end_point = self.up_point_items[i + 1].scenePos()
                    distance = math.sqrt((end_point.x() - start_point.x()) ** 2 + (end_point.y() - start_point.y()) ** 2)
                    up_total_length += distance

                up_lid_length = round(9 * up_total_length / self.s_diam, 2)
                coord_text += f"\n• Upper_lid Length: {up_lid_length:.2f}"
            elif self.active_point in self.down_point_items:
                down_total_length = 0.0
                for i in range(len(self.down_point_items) - 1):
                    start_point = self.down_point_items[i].scenePos()
                    end_point = self.down_point_items[i + 1].scenePos()
                    distance = math.sqrt((end_point.x() - start_point.x()) ** 2 + (end_point.y() - start_point.y()) ** 2)
                    down_total_length += distance

                down_lid_length = round(9 * down_total_length / self.s_diam, 2)
                coord_text += f"\n• Lower_lid Length: {down_lid_length:.2f}"
            
            if self.active_point in self.up_point_items:
                coord_text += f"\n  - moving Upper_lid"
            elif self.active_point in self.down_point_items:
                coord_text += f"\n  - moving Lower_lid"

            self.coord_label.setText(coord_text)
            self.coord_label.setFont(QFont("Arial", 12))
            self.coord_label.setGeometry(550, 170, 300, 80)  # 좌표와 크기를 조정

            print(coord_text)
        else:
            self.coord_label.clear()

# Select many upload
class SelectMany(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle('Many Settings Window')
        self.parent = parent

        file_upload_button = QPushButton('Upload Files', self)
        # self.setGeometry(80, 80, 700, 900)
        file_upload_button.setGeometry(520, 500, 150, 50)
        file_upload_button.clicked.connect(self.upload_files)

        back_button = QPushButton('Back to Main', self)
        back_button.clicked.connect(self.go_back)
        back_button.setGeometry(520, 580, 150, 50)
        
        # self.gif_label = QLabel(self)
        # self.gif_label.setGeometry(700, 400, 200, 200)
        # self.gif_label.setVisible(False)

    def upload_files(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.bmp *.jpeg *.gif)")
        selected_files, _ = file_dialog.getOpenFileNames(self, "Select Image Files", "", "Image Files (*.png *.jpg *.bmp *.jpeg *.gif)", options=options)
        many_to_save(selected_files)
        # if selected_files:
        #     print("Selected Files:", selected_files)

    def go_back(self):
        self.parent.show_buttons()
        self.close()
        
    def closeEvent(self, event):
        event.accept()

# Select only one
class SelectOne(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle('Settings Window')
        self.parent = parent
        
        for_img = 'fi_bg.png'
        self.fi_label = QLabel(self)
        fi_pixmap = QPixmap(for_img)
        fi_pixmap = fi_pixmap.scaled(700, 550)
        self.fi_label.setPixmap(fi_pixmap)
        self.fi_label.setGeometry(10, 10, 700, 550) 
        
        m_img = 'fi_bg.png'
        self.ma_label = QLabel(self)
        ma_label = QPixmap(m_img)
        ma_label = ma_label.scaled(360, 200)
        self.ma_label.setPixmap(ma_label)
        self.ma_label.setGeometry(800, 50, 360, 200) 
        
        l_img = 'fi_bg.png'
        self.lll_label = QLabel(self)
        lll_label = QPixmap(l_img)
        lll_label = lll_label.scaled(360, 300)
        self.lll_label.setPixmap(lll_label)
        self.lll_label.setGeometry(800, 220, 360, 300) 
        
        button_font = QFont("Arial", 16)
        
        select_button = QPushButton('Select File', self)
        select_button.setGeometry(260, 580, 200, 50)
        select_button.clicked.connect(self.select_file)
        # select_button.clicked.connect(self.show_text)

        help_button = QPushButton('Help', self)
        help_button.clicked.connect(self.go_back)
        help_button.setGeometry(830, 580, 140, 50)
        
        back_button = QPushButton('Back to Main', self)
        back_button.clicked.connect(self.go_back)
        back_button.setGeometry(1000, 580, 140, 50)
        
        self.or_label1 = QLabel(self); self.or_label1.setGeometry(20, 60, 420, 420)
        self.or_label2 = QLabel(self); self.or_label2.setGeometry(459, 60, 420, 420)
        
        self.label1 = QLabel(self); self.label1.setGeometry(20, 60, 420, 420)
        self.label2 = QLabel(self); self.label2.setGeometry(459, 60, 420, 420)
        
        self.o_label1 = QLabel(self); self.o_label1.setGeometry(20, 60, 420, 420)
        self.o_label2 = QLabel(self); self.o_label2.setGeometry(459, 60, 420, 420)
        
        self.origin_label = QLabel(self)
        self.origin_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.origin_label.move(35, 20); self.origin_label.setVisible(False)
        
        self.left_label = QLabel(self)
        self.left_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.left_label.move(367, 310); self.left_label.setVisible(False)
        
        # self.o_left_label = QLabel(self)
        # self.o_left_label.setFont(QFont("Arial", 18, QFont.Bold))
        # self.o_left_label.move(1368, 342); self.o_left_label.setVisible(False)
        
        self.right_label = QLabel(self)
        self.right_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.right_label.move(35, 310); self.right_label.setVisible(False)
        
        # length table
        self.len_table = QTableWidget(self)
        self.len_table.setRowCount(5)  # 행의 수
        self.len_table.setColumnCount(3)  # 열의 수
        self.len_table.move(830, 350)
        
        # manual
        self.manual_label = QLabel('Manual Edit', self)
        self.manual_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.manual_label.move(935, 75)
        self.manual_label.adjustSize()
        
        # manual  eyelid
        self.Eyelid_label = QLabel('Eyelid', self)
        self.Eyelid_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.Eyelid_label.move(830, 126)
        self.Eyelid_label.adjustSize()
        
        # left 버튼 추가
        self.to_left_button = QPushButton('Left', self)
        self.to_left_button.setFont(button_font)
        self.to_left_button.setGeometry(910, 120, 100, 40)
        self.to_left_button.clicked.connect(lambda: self.button_clicked("L"))
        
        # right 버튼 추가
        self.to_right_button = QPushButton('Right', self)
        self.to_right_button.setFont(button_font)
        self.to_right_button.setGeometry(1020, 120, 100, 40)
        self.to_right_button.clicked.connect(lambda: self.button_clicked("R"))
        
        # manual  reflex
        self.Reflex_label  = QLabel('Reflex', self)
        self.Reflex_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.Reflex_label.move(830, 201)
        self.Reflex_label.adjustSize()
        
        # left 버튼 추가
        self.rto_left_button = QPushButton('Left', self)
        self.rto_left_button.setFont(button_font)
        self.rto_left_button.setGeometry(910, 195, 100, 40)
        
        # right 버튼 추가
        self.rto_right_button = QPushButton('Right', self)
        self.rto_right_button.setFont(button_font)
        self.rto_right_button.setGeometry(1020, 195, 100, 40)
        
        self.lenre_label = QLabel('Length', self)
        self.lenre_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.lenre_label.move(935, 310)
        self.lenre_label.adjustSize()
        
        self.lenre_label.setVisible(False) 
        self.len_table.setVisible(False) 
        self.manual_label.setVisible(False)  
        self.Eyelid_label.setVisible(False)  
        self.Reflex_label.setVisible(False)  
        self.to_left_button.setVisible(False)  
        self.to_right_button.setVisible(False)
        self.rto_left_button.setVisible(False)  
        self.rto_right_button.setVisible(False)
        
        self.gif_label = QLabel(self)
        self.gif_label.setGeometry(520, 400, 200, 200)
        self.gif_label.setVisible(False)

    def select_file(self):
        gif_file_path = "giphy.gif"
        gif_movie = QMovie(gif_file_path)
        self.gif_label.setMovie(gif_movie)
        gif_movie.start()
        self.gif_label.setGeometry(250, 250, 300, 200)
        self.gif_label.setVisible(True)
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select File")
        
        def process_image():
            if file_path:
                origin_dir = './original'
                if not os.path.exists(origin_dir):
                    os.makedirs(origin_dir)
                
                destination_path = "./original/"
                shutil.copy(file_path, destination_path)
                file_name = file_path.split('/')[-1]
                o_l, o_r, L_length, R_length, L_img, R_img,\
                    overlay_L, overlay_R, L_edge, R_edge, f_name, diam = input_image_handle(destination_path, file_name)
                o_l, o_r = o_l[122:390], o_r[122:390]
                L_img, R_img = L_img[122:390], R_img[122:390]
                overlay_L, overlay_R = overlay_L[122:390], overlay_R[122:390]
                
                # reshape
                o_l = cv2.resize(o_l, dsize=(332,187))
                o_r = cv2.resize(o_r, dsize=(332,187))
                L_img = cv2.resize(L_img, dsize=(332,187))
                R_img = cv2.resize(R_img, dsize=(332,187))
                overlay_L = cv2.resize(overlay_L, dsize=(332,187))
                overlay_R = cv2.resize(overlay_R, dsize=(332,187))
                
                self.file_name = f_name
                self.Left_edge = L_edge
                self.Right_edge = R_edge
                self.s_diam = diam            
                
                #original image
                height, width, channels = o_l.shape
                ori_l = QImage(o_l.data, width, height, channels * width, QImage.Format_RGB888)
                ori_r = QImage(o_r.data, width, height, channels * width, QImage.Format_RGB888)
                or_pixmap1 = QPixmap.fromImage(ori_r)
                or_pixmap2 = QPixmap.fromImage(ori_l)
                self.or_label1.setPixmap(or_pixmap1)
                self.or_label1.setGeometry(30, 70, or_pixmap1.width(), or_pixmap1.height())
                self.or_label2.setPixmap(or_pixmap2)
                self.or_label2.setGeometry(362, 70, or_pixmap2.width(), or_pixmap2.height())
                
                # segment image
                height, width, channels = L_img.shape
                # l_image = QImage(L_img.data, width, height, channels * width, QImage.Format_RGB888)
                # r_image = QImage(R_img.data, width, height, channels * width, QImage.Format_RGB888)
                # pixmap1 = QPixmap.fromImage(r_image)
                # pixmap2 = QPixmap.fromImage(l_image)
                # self.label1.setPixmap(pixmap1)
                # self.label1.setGeometry(15, 255, pixmap1.width(), pixmap1.height())
                # self.label2.setPixmap(pixmap2)
                # self.label2.setGeometry(15, 495, pixmap2.width(), pixmap2.height())

                # overlay image
                height, width, channels = overlay_L.shape
                ov_l = QImage(overlay_L.data, width, height, channels * width, QImage.Format_RGB888)
                ov_r = QImage(overlay_R.data, width, height, channels * width, QImage.Format_RGB888)
                o_pixmap1 = QPixmap.fromImage(ov_l)
                o_pixmap2 = QPixmap.fromImage(ov_r)
                self.o_label1.setPixmap(o_pixmap1)
                self.o_label1.setGeometry(362, 350, or_pixmap1.width(), or_pixmap1.height())
                self.o_label2.setPixmap(o_pixmap2)
                self.o_label2.setGeometry(30, 350, or_pixmap1.width(), or_pixmap1.height())
                
                self.len_table.setItem(0, 1, QTableWidgetItem('R'))
                self.len_table.setItem(0, 2, QTableWidgetItem('L'))
                
                self.len_table.setItem(1, 0, QTableWidgetItem('Upper Lid'))
                self.len_table.setItem(2, 0, QTableWidgetItem('Lower Lid'))
                self.len_table.setItem(3, 0, QTableWidgetItem('MRD1'))
                self.len_table.setItem(4, 0, QTableWidgetItem('MRD2'))
                
                self.len_table.setItem(1, 1, QTableWidgetItem(str(R_length[0])))
                self.len_table.setItem(2, 1, QTableWidgetItem(str(R_length[1])))
                self.len_table.setItem(3, 1, QTableWidgetItem(str(R_length[2])))
                self.len_table.setItem(4, 1, QTableWidgetItem(str(R_length[3])))
                
                self.len_table.setItem(1, 2, QTableWidgetItem(str(L_length[0])))
                self.len_table.setItem(2, 2, QTableWidgetItem(str(L_length[1])))
                self.len_table.setItem(3, 2, QTableWidgetItem(str(L_length[2])))
                self.len_table.setItem(4, 2, QTableWidgetItem(str(L_length[3])))
                
                for row in range(0, 5):
                    for col in range(0, 3):
                        item = self.len_table.item(row, col)
                        if item:
                            item.setTextAlignment(Qt.AlignCenter)
                            
                self.len_table.verticalHeader().setVisible(False)
                self.len_table.horizontalHeader().setVisible(False)
                self.len_table.setFixedSize(305, 155)
                
                self.lenre_label.setVisible(True) 
                self.len_table.setVisible(True)
                self.to_left_button.setVisible(True)
                self.to_right_button.setVisible(True)
                self.rto_left_button.setVisible(True)
                self.rto_right_button.setVisible(True)
                self.manual_label.setVisible(True)
                self.gif_label.setVisible(False)
                self.origin_label.setText("Original")
                self.origin_label.setVisible(True)
                self.right_label.setText("Right")
                self.right_label.setVisible(True)
                self.left_label.setText("Left")
                self.left_label.setVisible(True)
                self.len_table.setVisible(True)
                self.Eyelid_label.setVisible(True)  
                self.Reflex_label.setVisible(True)  
                # self.o_right_label.setText("Right")
                # self.o_right_label.setVisible(True)
                # self.o_left_label.setText("Left")
                # self.o_left_label.setVisible(True)
                
        image_processing_thread = threading.Thread(target=process_image)
        image_processing_thread.start()
            
    def go_back(self):
        self.parent.show_buttons()
        self.close()
        
    def closeEvent(self, event):
        event.accept()
        
    def button_clicked(self, s):
        if s == 'L':
            load_img, edge, diam = load_result(self.file_name, self.Left_edge, self.s_diam, 'L')
        elif s == 'R':
            load_img, edge, diam = load_result(self.file_name, self.Right_edge, self.s_diam, 'R')
        edit_screen = EditScreen(self, load_img, edge, diam)
        edit_screen.exec_()

# Main window        
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setStyleSheet("background-color: white;")
        self.setWindowTitle('Semantic Eyelid Measurement')
        self.setGeometry(80, 80, 1200, 700)
        button_font = QFont("Arial", 16)
        
        # menubar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Menu')
        
        home_action = QAction('Home', self)
        home_action.triggered.connect(self.go_home)
        file_menu.addAction(home_action)
        
        select_many_action = QAction('Many', self)
        select_many_action.triggered.connect(self.open_select_many)
        file_menu.addAction(select_many_action)
        
        select_one_action = QAction('One', self)
        # select_one_action.triggered.connect(self.open_select_one)
        file_menu.addAction(select_one_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.exit_application)
        file_menu.addAction(exit_action)
        
        # main image
        image_path = 'main.png'
        self.label = QLabel(self)
        logo_pixmap = QPixmap(image_path)
        logo_pixmap = logo_pixmap.scaled(1100, 400, Qt.KeepAspectRatio)
        self.label.setPixmap(logo_pixmap)
        self.label.setGeometry(50, 80, logo_pixmap.width(), logo_pixmap.height()) 
        
        bgg = 'bg.png'
        self.bg_label = QLabel(self)
        bg_pixmap = QPixmap(bgg)
        bg_pixmap = bg_pixmap.scaled(500, 400, Qt.KeepAspectRatio)
        self.bg_label.setPixmap(bg_pixmap)
        self.bg_label.setGeometry(350, 410, logo_pixmap.width(), logo_pixmap.height()) 
        
        self.select_label  = QLabel('Select Mode', self)
        self.select_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.select_label.move(530, 540)
        self.select_label.adjustSize()
     
        # "Many" 버튼 생성 및 설정
        self.many_button = QPushButton('Many', self)
        self.many_button.setFont(button_font)
        self.many_button.setGeometry(400, 580, 160, 80)
        self.many_button.clicked.connect(self.open_select_many)

        # "One" 버튼 생성 및 설정
        self.one_button = QPushButton('One', self)
        self.one_button.setFont(button_font)
        self.one_button.setGeometry(640, 580, 160, 80)
        self.one_button.clicked.connect(self.open_select_one)
    
      
    def hide_buttons(self):
        self.many_button.hide()
        self.one_button.hide()
        self.label.hide()
        self.bg_label.hide()
        self.select_label.hide()

    def show_buttons(self):
        self.many_button.show()
        self.one_button.show()
        self.label.show()
        self.bg_label.show()
        self.select_label.show()
        
    def open_select_many(self):
        self.hide_buttons()
        self.select_many_window = SelectMany(parent=self)
        self.setCentralWidget(self.select_many_window)
        
    def open_select_one(self):
        self.hide_buttons()        
        self.select_one_window = SelectOne(parent=self)
        self.setCentralWidget(self.select_one_window)
        
    def show_edit_screen(self, load_img, edge, diam):
        edit_screen = EditScreen(self, load_img, edge, diam)
        edit_screen.exec_()
        
    def show_select_one(self):
        self.select_one_window = SelectOne(parent=self)
        self.setCentralWidget(self.select_one_window)

    def go_home(self):
        if self.select_many_window:
            self.select_many_window.close()
        if self.select_one_window:
            self.select_one_window.close()
        self.show_buttons()
        
    def exit_application(self):
        self.close()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())