import paho.mqtt.client as mqtt
import time
import json
import math

import Adafruit_ADS1x15

myBroker = "broker.hivemq.com"
myPort = 1883
myTimeOut = 60

adc_1 = Adafruit_ADS1x15.ADS1115()
time.sleep(1)
adc_2 = Adafruit_ADS1x15.ADS1115(address=0x4A)
time.sleep(1)

GAIN = 1

temp_current = []
rawValue1 = []
rawValue2 = []
rawValue3 = []

i = 0
currentku = 0
temp = 0
final = 0

lastValue1 = 0
lastValue2 = 0
lastValue3 = 0

voltage = 0.0
current = 0.0
power = 0.0
motor = 0


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("/motor_monitoring/control")

def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload}")
    motor = msg.payload
    print(f"send {voltage} {current} {power} {motor}to /motor_monitoring/logger")

def findMax1():
    sensorValue = 0
    sensorMax = 0
    start_time = 0
    while start_time < 30 :
        sensorValue = adc_2.read_adc(0, gain=GAIN)
        if sensorValue > sensorMax:
            sensorMax = sensorValue
        start_time = start_time + 1
    start_time = 0
    return sensorMax

def findMax2():
    sensorValue = 0
    sensorMax = 0
    start_time = 0
    while start_time < 30 :
        sensorValue = adc_2.read_adc(1, gain=GAIN)
        if sensorValue > sensorMax:
            sensorMax = sensorValue
        start_time = start_time + 1
    start_time = 0
    return sensorMax

def findMax3():
    sensorValue = 0
    sensorMax = 0
    start_time = 0
    while start_time < 30 :
        sensorValue = adc_2.read_adc(2, gain=GAIN)
        if sensorValue > sensorMax:
            sensorMax = sensorValue
        start_time = start_time + 1
    start_time = 0
    return sensorMax


    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(myBroker, myPort, myTimeOut)

# send a message to the raspberry/topic every 1 second, 5 times in a row
while 1 :
    client.loop_start()
    values_current = [0]*4
    values_voltage = [0]*4
    final_current = [0]*4
    final_voltage = [0]*4
    amc_values = [0]*4
    
    
    rawCurrent = [0]*4
    
    for i in range(50):
        value1 = adc_1.read_adc(0, gain=GAIN)
        value2 = adc_1.read_adc(1, gain=GAIN)
        value3 = adc_1.read_adc(2, gain=GAIN)
        
        #print(value1)
        
        rawValue1.append(value1)
        rawValue2.append(value2)
        rawValue3.append(value3)
    
    rawValue1.sort()
    rawValue2.sort()
    rawValue3.sort()
    
    rawCurrent[0] = rawValue1[0]
    rawCurrent[1] = rawValue2[0]
    rawCurrent[2] = rawValue3[0]
    
    #print(rawValue1[0])
    #print(rawValue2[0])
    #print(rawValue3[0])
    
    #print((rawCurrent[0]/37632) * 4770)
    
    rawValue1.clear()
    rawValue2.clear()
    rawValue3.clear()
    
    final_current[0] = (rawCurrent[0]/37632) * 4770
    amplitude_current = ((final_current[0] - 2387) / 100);
    final_current[0] = 3.055555555555556*10*amplitude_current / math.sqrt(2);
    
    final_current[1] = (rawCurrent[1]/37632) * 4770
    amplitude_current = ((final_current[1] - 2387) / 100);
    final_current[1] = 3.055555555555556*10*amplitude_current / math.sqrt(2);
    
    final_current[2] = (rawCurrent[2]/37632) * 4770
    amplitude_current = ((final_current[2] - 2387) / 100);
    final_current[2] = 3.055555555555556*10*amplitude_current / math.sqrt(2);
    #amc_values[0] = findMax1()
    #final_voltage[0] = abs(amc_values[0]-20274)/(22281-20274) * 220
    
    #amc_values[1] = findMax1()
    #final_voltage[1] = abs(amc_values[1]-20274)/(22281-20274) * 220
    
    #amc_values[2] = findMax1()
    #final_voltage[2] = abs(amc_values[2]-20274)/(22281-20274) * 220
    
    amc_values[0] = findMax1()
    final_voltage[0] = 0.07448*amc_values[0] - 1499.32466 - 102
    final_voltage[0] = final_voltage[0] * 10 /13    

    amc_values[1] = findMax2()
    final_voltage[1] = 0.07448*amc_values[1] - 1499.32466 - 102
    final_voltage[1] = final_voltage[1] * 10 /13    

    amc_values[2] = findMax3()
    final_voltage[2] = 0.07448*amc_values[2] - 1499.32466 - 102
    final_voltage[2] = final_voltage[2] * 10 /13

    print(amc_values[0])
    
    # 0 -> 20274
    # 220 -> 22281
    
    print("\n ADS 1 = Arus")
    print("channel 0 = ",abs(final_current[0]))
    print("channel 1 = ",abs(final_current[1]))
    print("channel 2 = ",abs(final_current[2]))
    
    print("\n ADS 2 = Voltage")
    print("channel 0 = ",abs(final_voltage[0]))
    print("channel 1 = ",abs(final_voltage[1]))
    print("channel 2 = ",abs(final_voltage[2]))
    
    #v_average = (abs(final_current[0]) + abs(final_current[1]) + abs(final_current[2])) / 3
    #i_average = (abs(final_voltage[0]) + abs(final_voltage[1]) + abs(final_voltage[2])) / 3
    
    #spek motor 3 fasa 380 v, 1.6 A, 0.65 KW
    
    #power = i_average * v_average * math.sqrt(3) * 0.62
    p = abs(final_current[0]) * abs(final_voltage[0]) * 0.6 * 1.73
    
    print("\ndaya motor = ",p , "volt")
    
    total_voltage = (final_voltage[0] + final_voltage[1] + final_voltage[2])/3
    total_current = (final_current[0] + final_current[1] + final_current[2])/3

    client.publish('/motor_monitoring/logger', payload=json.dumps({"tegangan_fasa_1" : abs(final_voltage[0]), "tegangan_fasa_2" : abs(final_voltage[1]), "tegangan_fasa_3" : abs(final_voltage[2]), "arus_fasa_1" : abs(final_current[0]), "arus_fasa_2" : abs(final_current[1]), "arus_fasa_3" : abs(final_current[2]), "tegangan_total": total_voltage, "arus_total": total_current ,"daya" : p}), qos=0, retain=False)