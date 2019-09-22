import numpy as np
import cv2

pic_path = './mid_result_vis/epoch0/plot_dif_0.png'
pic = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)

n = 10
pic = pic * n

r, c = pic.shape
for x in range(r):
	for y in range(c):
		if(pic[x][y] > 255):
			pic[x][y] = 255

save_path = './light' + str(n) + 'times.png'
cv2.imwrite(save_path, pic)