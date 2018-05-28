from keras.models import load_model
from keras.preprocessing import image
from keras.models import Model
from scipy.misc import imresize as imresize
import numpy as np
import cv2

CLASS_NUM = 9
names = ['is_male', 'have_long_hair', 'have_glasses', 'have_hat', 'have_T-shirt',
         'have_long_sleeves', 'have_shorts', 'have_jeans', 'have_long_pants']
model = load_model('first_blood.h5')

def returnCAM(dense_output, weights, class_idx):  # (7,7,512), (512,9), [需要显示的种类数]
    # generate the class activation maps upsample to 224*224
    size_upsample = (224, 224)
    h, w, nc = dense_output.shape  # (7,7,512)
    output_cam = []
    for idx in class_idx:
        cam = (dense_output.reshape((7*7, 512))).dot(weights[:,idx])  # [7*7, 512].dot([512,1])=(49,)
        cam = cam.reshape(h, w)  # [7,7]
        cam1 = cam - np.min(cam)
        cam_img = cam1 / (np.max(cam)-np.min(cam))  # 这个地方修改为除以最大值和最小值的差值
        cam_img = np.uint8(255 * cam_img)  # 把结果归一化到[0,255]之间
        output_cam.append(imresize(cam_img, size_upsample))  # [7,7] 直接resize成[224, 224]
    return output_cam

# 获取权重
weights = model.layers[-1].get_weights()[0]
print(weights.shape)  # [512, 9]


img_path = '1.jpg'  # male, glasses, hat
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
dense_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('block5_pool').output)
# 以这个model的预测值作为输出
dense_output = dense_layer_model.predict(x)
dense_output = np.squeeze(dense_output)
# print(dense_output.shape)  # [7, 7, 512]
# print((dense_output.reshape((7*7, 512))).dot(weights[:,0]).shape)

preds = model.predict(x)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
print(preds)
pre_list = preds[0].tolist()
class_idx = []
for i,v in enumerate(pre_list):
    if v == 1:
        class_idx.append(i)  # print(i)  # names[i]
print(class_idx)


CAMs = returnCAM(dense_output, weights, class_idx)  # (512, 14, 14), (365, 512), [52] -> list(array([256,256]))
# render the CAM and output
img = cv2.imread(img_path)
height, width, _ = img.shape
font=cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
# print(CAMs[0].shape)
for i, v in enumerate(CAMs):
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[i],(width, height)), cv2.COLORMAP_JET)  # 色度图的一种模式,中间红,最外层蓝
    result = heatmap * 0.4 + img * 0.5
    cv2.putText(result, names[class_idx[i]] , (10, 40), font, 1.2, (255, 255, 255), 2)
    cv2.imwrite('{}.jpg'.format(names[class_idx[i]]), result)  # names[v]

