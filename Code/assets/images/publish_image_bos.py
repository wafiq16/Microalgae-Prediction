import paho.mqtt.publish as publish
import json
import base64

# f = open("mqtt.png", "rb")
filecontent = ""
# base64.b64encode(filecontent)
with open("mqtt.png", "rb") as image_file:
    filecontent = base64.b64encode(image_file.read())
byteArr = bytearray(filecontent)
# byteArr = "haha"
message = {"sensor1": 25.5, "sensor2": 10}
string = {"MQ3": "19.25", "TGS": "29.05", "GolA1": "0", "GolB1": "1", "GolC1": "0",
          "status1": "0", "GolA2": "0", "GolB2": "0", "GolC2": "1", "status2": "0"}
data_out = json.dumps(string)

publish.single('/alcoholMonitor', data_out, qos=0,
               hostname='broker.hivemq.com')
# publish.single('/esp32camSensor', data_out, qos=0,
#                hostname='192.168.117.202')
