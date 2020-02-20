import requests
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image

x = Image.open('./cat.jpg')
preprocess = Compose([Resize((224, 224)),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

x = preprocess(x).numpy().tolist()

df = pd.DataFrame(data=x)

v = df.values
data = df.to_json(orient='split')

res = requests.post('http://127.0.0.1:5000/invocations', json={
    "columns": ['x'],
    "data": [[x]]
})
# print(res.content)