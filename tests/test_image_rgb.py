import cv2
import matplotlib.pyplot as plt



img = cv2.imread("collect_data/2025_06_20-15_04_59/sensors/top_cam/6.png")
plt.imshow(img)
plt.title("If image looks blue-ish, it's BGR")
plt.savefig("/home/ka/ka_stud/ka_ulgrl/policies/Isaac-GR00T/tests/bgr_image.png")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("rgb")
plt.savefig("/home/ka/ka_stud/ka_ulgrl/policies/Isaac-GR00T/tests/rgb_image.png")
