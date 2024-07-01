import requests

url = "http://localhost:5000/predict"

payload = {}
files=[
  ('image_l',('left_img.png', open('/home/ubuntu/CarlaFLCAV/FLDatasetTool/dataset/record_2024_0405_1702/vehicle.tesla.model3_1/kitti_object/training/image_2/000094.png','rb'),'image/png')),
  ('image_r',('right_img.png', open('/home/ubuntu/CarlaFLCAV/FLDatasetTool/dataset/record_2024_0405_1702/vehicle.tesla.model3_1/kitti_object/training/image_3/000094.png','rb'),'image/png')),
  ('calib',('calib.txt', open('/home/ubuntu/poc/LIGA-Stereo/calib.txt','rb'),'text/plain'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)