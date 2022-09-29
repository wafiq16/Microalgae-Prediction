import random
import csv

consentration = [0, 20, 40, 60, 80, 100]
header = ['class', 'consentration', 'camera max red', 'camera max green', 'camera max blue', 'camera mean red', 'camera mean green',
          'camera mean blue', 'mic max red', 'mic max green', 'mic max blue', 'mic mean red', 'mic mean green', 'mic mean blue', 'sensor red', 'sensor green', 'sensor blue']
header_c = ['camera mean red', 'camera mean green', 'camera mean blue', 'camera max red', 'camera max green', 'camera max blue', 'mic mean red',
            'mic mean green', 'mic mean blue', 'mic max red', 'mic max green', 'mic max blue', 'sensor red', 'sensor green', 'sensor blue', 'class']

header_r = ['camera mean red', 'camera mean green', 'camera mean blue', 'camera max red', 'camera max green', 'camera max blue', 'mic mean red',
            'mic mean green', 'mic mean blue', 'mic max red', 'mic max green', 'mic max blue', 'sensor red', 'sensor green', 'sensor blue', 'concentration']

# =======================
# ==> class encoder   <==
# ==> 1 = red         <==
# ==> 2 = green       <==
# ==> 3 = blue        <==
# =======================

f_reg = open('data/dummy_data_reg.csv', 'w')
# writer = csv.writer(f)
writer_reg = csv.writer(f_reg)
writer_reg.writerow(header_r)

f_class = open('data/dummy_data_cla.csv', 'w')
# writer = csv.writer(f)
writer_class = csv.writer(f_class)
writer_class.writerow(header_c)

# writer.writerow(header)

i = 0

while(i < 500):
    # class dominant red
    classes = "red"
    cam_mean_r = round(random.uniform(0.4, 0.7), 3)
    cam_mean_g = round(random.uniform(0, 1-cam_mean_r-0.3), 3)
    cam_mean_b = 1 - cam_mean_r - cam_mean_g

    cam_max_r = round(random.uniform(cam_mean_r, 0.7), 3)
    cam_max_g = round(random.uniform(0, 1-cam_max_r-0.3), 3)
    cam_max_b = 1 - cam_max_r - cam_max_g

    mic_mean_r = round(random.uniform(0.4, 0.7), 3)
    mic_mean_g = round(random.uniform(0, 1-mic_mean_r-0.3), 3)
    mic_mean_b = 1 - mic_mean_r - mic_mean_g

    mic_max_r = round(random.uniform(mic_mean_r, 0.7), 3)
    mic_max_g = round(random.uniform(0, 1-mic_max_r-0.3), 3)
    mic_max_b = 1 - mic_max_r - mic_max_g

    sens_r = round(random.uniform(0.4, 0.7), 3)
    sens_g = round(random.uniform(0, 1-sens_r-0.3), 3)
    sens_b = 1 - sens_r - sens_g

    flag = (1, 0)[sens_g > sens_b]

    if(sens_r <= 0.3):
        amountR = random.randint(consentration[0], consentration[1])
        if(flag):
            amountG = consentration[random.randint(2, 3)]
            amountB = consentration[random.randint(4, 5)]
        else:
            amountB = consentration[random.randint(2, 3)]
            amountG = consentration[random.randint(4, 5)]

    elif(sens_r <= 0.6 and sens_r >= 0.3):
        amountR = consentration[random.randint(2, 3)]

        if(flag):
            amountG = consentration[random.randint(0, 1)]
            amountB = consentration[random.randint(4, 5)]
        else:
            amountB = consentration[random.randint(0, 1)]
            amountG = consentration[random.randint(4, 5)]

    elif(sens_r >= 0.6):
        amountR = consentration[random.randint(4, 5)]
        if(flag):
            amountG = consentration[random.randint(0, 1)]
            amountB = consentration[random.randint(2, 3)]
        else:
            amountB = consentration[random.randint(0, 1)]
            amountG = consentration[random.randint(2, 3)]

    print("red class : ")
    print("cam mean r = %f, cam mean g = %f, cam mean b = %f, cam max r = %f, cam max g = %f, cam max b = %f \n mic mean r = %f, mic mean g = %f, mic mean b = %f, mic max r = %f, mic max g = %f, mic max b = %f \n, sensor r = %f , sensor g = %f, sensor b = %f  || amountR = %f | amountG = %f | amountB = %f \n" % (
        cam_mean_r, cam_mean_g, cam_mean_b, cam_max_r, cam_max_g, cam_max_b, mic_mean_r, mic_mean_g, mic_mean_b, mic_max_r, mic_max_g, mic_max_b, sens_r, sens_g, sens_b, amountR, amountG, amountB))
    row = [round(cam_mean_r, 3), round(cam_mean_g, 3), round(cam_mean_b, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(mic_mean_r, 3),
           round(mic_mean_g, 3), round(mic_mean_b, 3), round(mic_max_r, 3), round(mic_max_g, 3), round(mic_max_b, 3), round(sens_r, 3), round(sens_g, 3), round(sens_b, 3), classes]
    writer_class.writerow(row)
    row = [round(cam_mean_r, 3), round(cam_mean_g, 3), round(cam_mean_b, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(mic_mean_r, 3),
           round(mic_mean_g, 3), round(mic_mean_b, 3), round(mic_max_r, 3), round(mic_max_g, 3), round(mic_max_b, 3), round(sens_r, 3), round(sens_g, 3), round(sens_b, 3), amountR, amountG, amountB]
    writer_reg.writerow(row)
    i = i + 1

i = 0

while(i < 500):

    # class dominant green
    classes = "green"
    cam_mean_g = round(random.uniform(0.4, 0.7), 3)
    cam_mean_b = round(random.uniform(0, 1-cam_mean_g-0.3), 3)
    cam_mean_r = 1 - cam_mean_b - cam_mean_g

    cam_max_g = round(random.uniform(cam_mean_g, 0.7), 3)
    cam_max_b = round(random.uniform(0, 1-cam_max_g-0.3), 3)
    cam_max_r = 1 - cam_max_b - cam_max_g

    mic_mean_g = round(random.uniform(0.4, 0.7), 3)
    mic_mean_b = round(random.uniform(0, 1-mic_mean_g-0.3), 3)
    mic_mean_r = 1 - mic_mean_b - mic_mean_g

    mic_max_g = round(random.uniform(mic_mean_g, 0.7), 3)
    mic_max_b = round(random.uniform(0, 1-mic_max_g-0.3), 3)
    mic_max_r = 1 - mic_max_b - mic_max_g

    sens_g = round(random.uniform(0.4, 0.7), 3)
    sens_b = round(random.uniform(0, 1-sens_g-0.3), 3)
    sens_r = 1 - sens_b - sens_g

    flag = (1, 0)[sens_r > sens_b]

    if(sens_g <= 0.3):
        amountG = consentration[random.randint(0, 1)]
        if(flag):
            amountR = consentration[random.randint(2, 3)]
            amountB = consentration[random.randint(4, 5)]
        else:
            amountB = consentration[random.randint(2, 3)]
            amountR = consentration[random.randint(4, 5)]

    elif(sens_g <= 0.6 and sens_g >= 0.3):
        amountG = consentration[random.randint(2, 3)]

        if(flag):
            amountR = consentration[random.randint(0, 1)]
            amountB = consentration[random.randint(4, 5)]
        else:
            amountB = consentration[random.randint(0, 1)]
            amountR = consentration[random.randint(4, 5)]

    elif(sens_g >= 0.6):
        amountG = consentration[random.randint(4, 5)]
        if(flag):
            amountR = consentration[random.randint(0, 1)]
            amountB = consentration[random.randint(2, 3)]
        else:
            amountB = consentration[random.randint(0, 1)]
            amountR = consentration[random.randint(2, 3)]

    print("blue class : ")
    print("cam mean r = %f, cam mean g = %f, cam mean b = %f, cam max r = %f, cam max g = %f, cam max b = %f \n mic mean r = %f, mic mean g = %f, mic mean b = %f, mic max r = %f, mic max g = %f, mic max b = %f \n, sensor r = %f , sensor g = %f, sensor b = %f  || amountR = %f | amountG = %f | amountB = %f \n" % (
        cam_mean_r, cam_mean_g, cam_mean_b, cam_max_r, cam_max_g, cam_max_b, mic_mean_r, mic_mean_g, mic_mean_b, mic_max_r, mic_max_g, mic_max_b, sens_r, sens_g, sens_b, amountR, amountG, amountB))
    row = [round(cam_mean_r, 3), round(cam_mean_g, 3), round(cam_mean_b, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(mic_mean_r, 3),
           round(mic_mean_g, 3), round(mic_mean_b, 3), round(mic_max_r, 3), round(mic_max_g, 3), round(mic_max_b, 3), round(sens_r, 3), round(sens_g, 3), round(sens_b, 3), classes]
    writer_class.writerow(row)
    row = [round(cam_mean_r, 3), round(cam_mean_g, 3), round(cam_mean_b, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(mic_mean_r, 3),
           round(mic_mean_g, 3), round(mic_mean_b, 3), round(mic_max_r, 3), round(mic_max_g, 3), round(mic_max_b, 3), round(sens_r, 3), round(sens_g, 3), round(sens_b, 3), amountR, amountG, amountB]
    writer_reg.writerow(row)
    i = i + 1

i = 0

while(i < 500):
    # class dominant blue
    classes = "blue"

    cam_mean_b = round(random.uniform(0.4, 0.7), 3)
    cam_mean_g = round(random.uniform(0, 1-cam_mean_b-0.3), 3)
    cam_mean_r = 1 - cam_mean_b - cam_mean_g

    cam_max_b = round(random.uniform(cam_mean_b, 0.7), 3)
    cam_max_g = round(random.uniform(0, 1-cam_max_b-0.3), 3)
    cam_max_r = 1 - cam_max_b - cam_max_g

    mic_mean_b = round(random.uniform(0.4, 0.7), 3)
    mic_mean_g = round(random.uniform(0, 1-mic_mean_b-0.3), 3)
    mic_mean_r = 1 - mic_mean_b - mic_mean_g

    mic_max_b = round(random.uniform(mic_mean_b, 0.7), 3)
    mic_max_g = round(random.uniform(0, 1-mic_max_b-0.3), 3)
    mic_max_r = 1 - mic_max_b - mic_max_g

    sens_b = round(random.uniform(0.4, 0.7), 3)
    sens_g = round(random.uniform(0, 1-sens_b-0.3), 3)
    sens_r = 1 - sens_b - sens_g

    flag = (1, 0)[sens_g > sens_r]

    if(sens_b <= 0.3):
        amountB = consentration[random.randint(0, 1)]
        if(flag):
            amountG = consentration[random.randint(2, 3)]
            amountR = consentration[random.randint(4, 5)]
        else:
            amountR = consentration[random.randint(2, 3)]
            amountG = consentration[random.randint(4, 5)]

    elif(sens_b <= 0.6 and sens_b >= 0.3):
        amountB = consentration[random.randint(2, 3)]

        if(flag):
            amountG = consentration[random.randint(0, 1)]
            amountR = consentration[random.randint(4, 5)]
        else:
            amountR = consentration[random.randint(0, 1)]
            amountG = consentration[random.randint(4, 5)]

    elif(sens_b >= 0.6):
        amountB = consentration[random.randint(4, 5)]
        if(flag):
            amountG = consentration[random.randint(0, 1)]
            amountR = consentration[random.randint(2, 3)]
        else:
            amountR = consentration[random.randint(0, 1)]
            amountG = consentration[random.randint(2, 3)]

    print("blue class : ")
    print("cam mean r = %f, cam mean g = %f, cam mean b = %f, cam max r = %f, cam max g = %f, cam max b = %f \n mic mean r = %f, mic mean g = %f, mic mean b = %f, mic max r = %f, mic max g = %f, mic max b = %f \n, sensor r = %f , sensor g = %f, sensor b = %f  || amountR = %f | amountG = %f | amountB = %f \n" % (
        cam_mean_r, cam_mean_g, cam_mean_b, cam_max_r, cam_max_g, cam_max_b, mic_mean_r, mic_mean_g, mic_mean_b, mic_max_r, mic_max_g, mic_max_b, sens_r, sens_g, sens_b, amountR, amountG, amountB))
    row = [round(cam_mean_r, 3), round(cam_mean_g, 3), round(cam_mean_b, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(mic_mean_r, 3),
           round(mic_mean_g, 3), round(mic_mean_b, 3), round(mic_max_r, 3), round(mic_max_g, 3), round(mic_max_b, 3), round(sens_r, 3), round(sens_g, 3), round(sens_b, 3), classes]
    writer_class.writerow(row)
    row = [round(cam_mean_r, 3), round(cam_mean_g, 3), round(cam_mean_b, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(cam_max_r, 3), round(mic_mean_r, 3),
           round(mic_mean_g, 3), round(mic_mean_b, 3), round(mic_max_r, 3), round(mic_max_g, 3), round(mic_max_b, 3), round(sens_r, 3), round(sens_g, 3), round(sens_b, 3), amountR, amountG, amountB]
    writer_reg.writerow(row)
    i = i + 1

i = 0
