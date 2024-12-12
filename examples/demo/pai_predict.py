"""
调用阿里云上的深度学习模型，
sh:curl 1828488879222***.cn-shanghai.pai-eas.aliyuncs.com/api/predict/mnist_saved_model_example -H 'Authorization: YTg2ZjE0ZjM4ZmE3OTc0NzYxZDMyNmYzMTJjZTQ1***'
"""

#!/usr/bin/env python

from eas_prediction import PredictClient
from eas_prediction import TFRequest

if __name__ == '__main__':
    client = PredictClient('http://1828488879222***.cn-shanghai.pai-eas.aliyuncs.com', 'mnist_saved_model_example')
    client.set_token('YTg2ZjE0ZjM4ZmE3OTc0NzYxZDMyNmYzMTJjZTQ1****')
    client.init()

    req = TFRequest('predict_images')
    req.add_feed('images', [1, 784], TFRequest.DT_FLOAT, [1] * 784)
    for x in range(0, 1000000):
        resp = client.predict(req)
        print(resp)