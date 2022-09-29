import cv2

frameKamera = [30, 400, 330, 430]

frame = cv2.imread('image/Camera/cam'+str(iCam-1)+'.jpg')

height_final_begin = frameKamera[0] + 3
width_final_begin = frameKamera[1] + 3
height_final_end = frameKamera[2] - 3
width_final_end = frameKamera[3] - 3

frame = frame[height_final_begin:height_final_end,
              width_final_begin: width_final_end]

# cv2.imwrite('image/Mic/mic ' +
#             str(datetime.datetime.now().strftime('%y %m %d %H:%M:%S'))+'.jpg', frame)

savedImageMic = frame

cv2.imshow('frame', frame)

histR = cv2.calcHist([frame], [2], None, [256], [0, 256])
histG = cv2.calcHist([frame], [1], None, [256], [0, 256])
histB = cv2.calcHist([frame], [0], None, [256], [0, 256])

sumAllR = 0
sumMulR = 0
sumAllG = 0
sumMulG = 0
sumAllB = 0
sumMulB = 0

for i in range(0, 256):
     if(histR[i] == np.max(histR)):
         max_cam_R = i
         # inEMaR.set(i)
     sumAllR += histR[i]
     sumMulR += i*histR[i]
     if(histG[i] == np.max(histG)):
         max_cam_G = i
         # inEMaG.set(i)
     sumAllG += histG[i]
     sumMulG += i*histG[i]
     if(histB[i] == np.max(histB)):
         max_cam_B = i
         # inEMaB.set(i)
     sumAllB += histB[i]
     sumMulB += i*histB[i]

    mean_cam_R = int(sumMulR/sumAllR)
    mean_cam_G = int(sumMulG/sumAllG)
    mean_cam_B = int(sumMulB/sumAllB)

    Max_R_Cam_1.setText(str(max_cam_R))
    Max_G_Cam_1.setText(str(max_cam_G))
    Max_B_Cam_1.setText(str(max_cam_B))

    Mean_R_Cam_1.setText(str(mean_cam_R))
    Mean_G_Cam_1.setText(str(mean_cam_G))
    Mean_B_Cam_1.setText(str(mean_cam_B))
