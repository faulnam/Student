import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar dari file
image_path = "yeye.jpg"  # Pastikan file ini ada di folder yang sama
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Gambar tidak ditemukan!")
    exit()

# 1. Citra Negatif
negative = 255 - img

# 2. Transformasi Log
c = 255 / np.log(1 + np.max(img))
log_transform = c * (np.log(1 + img.astype(np.float32)))
log_transform = np.array(log_transform, dtype=np.uint8)

# 3. Transformasi Power Law (Gamma Correction)
gamma = 2.0
gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

# 4. Histogram Equalization
hist_eq = cv2.equalizeHist(img)

# 5. Histogram Normalization
norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# 6. Konversi RGB ke HSI (Menggunakan HSV karena OpenCV tidak punya HSI)
img_rgb = cv2.imread(image_path)  # Load gambar berwarna
if img_rgb is None:
    print("Error: Gambar tidak ditemukan atau tidak bisa dibaca sebagai RGB!")
    exit()

hsi = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

# Menampilkan hasil
titles = ["Original", "Negative", "Log Transform", "Power Law", "Histogram Eq", "Histogram Norm"]
images = [img, negative, log_transform, gamma_corrected, hist_eq, norm_img]

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.show()
