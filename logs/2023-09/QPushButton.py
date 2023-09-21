import sys
import os
from PySide6.QtWidgets import (QApplication, QWidget, 
    QPushButton, QStyleFactory,QVBoxLayout)
from PySide6.QtGui import QIcon

class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(300, 200)
        self.setWindowTitle("为按钮添加图标")

        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建图标文件的相对路径
        icon_path = os.path.join(script_dir, "vtune.ico")
        icon = QIcon(icon_path)

        self.button = QPushButton("Button")
        self.button.setIcon(icon)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = MainWidget()
    window.show()
    sys.exit(app.exec())