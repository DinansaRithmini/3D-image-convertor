from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from image_processing import estimate_depth, generate_point_cloud, noise_reduction, color_processing
import matplotlib.pyplot as plt


class ImageTo3DApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.loadButton = QPushButton("Load Image", self)
        self.loadButton.clicked.connect(self.load_image)
        self.layout.addWidget(self.loadButton)

        self.imageLabel = QLabel(self)
        self.layout.addWidget(self.imageLabel)

        self.depthButton = QPushButton("Generate 3D Image", self)
        self.depthButton.clicked.connect(self.generate_3d_image)
        self.layout.addWidget(self.depthButton)

        self.noiseButton = QPushButton("Reduce Noise", self)
        self.noiseButton.clicked.connect(self.reduce_noise)
        self.layout.addWidget(self.noiseButton)

        self.colorButton = QPushButton("Color Process", self)
        self.colorButton.clicked.connect(self.color_process)
        self.layout.addWidget(self.colorButton)

        self.setLayout(self.layout)
        self.setWindowTitle('2D to 3D Image Conversion')
        self.show()

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "All Files (*);;Image Files (*.jpg; *.png)", options=options)
        if fileName:
            self.img_path = fileName
            pixmap = QPixmap(fileName)
            self.imageLabel.setPixmap(pixmap)

    def generate_3d_image(self):
        depth_map = estimate_depth(self.img_path)
        points, colors = generate_point_cloud(self.img_path, depth_map)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=0.1)
        plt.show()

    def reduce_noise(self):
        denoised_img = noise_reduction(self.img_path)
        cv2.imshow("Denoised Image", denoised_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def color_process(self):
        processed_img = color_processing(self.img_path)
        cv2.imshow("Color Processed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication([])
    ex = ImageTo3DApp()
    app.exec_()
