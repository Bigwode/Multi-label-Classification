# coding:utf-8
import keras
from keras.applications import vgg16, xception
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Embedding, Dropout, Flatten, Dense
from keras.models import Model, Sequential


# 未使用预训练模型
def build_normal(img_width, img_height):
    input_image = Input(shape=(img_width, img_height, 3))
    x1 = (Conv2D(32, (3, 3), activation='relu'))(input_image)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    x2 = Conv2D(64, (3, 3), activation='relu')(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x31 = Conv2D(128, (3, 3), activation='relu')(x2)
    x31 = MaxPooling2D(pool_size=(2, 2))(x31)

    x41 = GlobalAveragePooling2D()(x31)
    # Flatten()(x31)
    x51 = Dense(64, activation='relu')(x41)
    # x61 = Dropout(0.5)(x51)
    # prediction1 = Dense(6, activation='softmax')(x61) # 6分类


    x32 = Conv2D(256, (3, 3), activation='relu')(x2)
    x32 = MaxPooling2D(pool_size=(2, 2))(x32)

    x42 = Conv2D(256, (3, 3), activation='relu')(x32)
    x42 = MaxPooling2D(pool_size=(2, 2))(x42)

    x52 = GlobalAveragePooling2D()(x42)
    x62 = Dense(64, activation='relu')(x52)
    merged_vector = keras.layers.concatenate([x51, x62], axis=-1)  # (None, 64), (None, 64) -> (none, 128)
    x72 = Dropout(0.5)(merged_vector)
    prediction = Dense(6, activation='softmax')(x72) # 6分类

    model = Model(inputs=input_image, outputs=prediction)
    # print(model.summary())
    return model

# build(128, 128)


# 使用预训练模型
def build_vgg_mod(img_width, img_height):

    # image_input = Input(shape=(224, 224, 3))
    vgg_model = vgg16.VGG16(input_tensor=None, weights=None,
                            include_top=False, input_shape=(img_width, img_height, 3))  # 'imagenet'

    model_mid = Model(inputs =vgg_model.input, outputs= vgg_model.get_layer('block3_pool').output)
    x1 = model_mid.get_layer('block3_pool').output  # (None, 14, 14, 512)

    # 任意中间层中抽取特征
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='blo4_conv1')(x1)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='blo4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='blo4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2) ,name='blo4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='blo5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='blo5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='blo5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='blo5_pool')(x)

    x2 = MaxPooling2D((4, 4), strides=(4, 4))(x1)  # # max pooling 到 (N, 7, 7, 256)
    merged_vector = keras.layers.concatenate([x, x2], axis=-1)

    x = GlobalAveragePooling2D()(merged_vector)
    # x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    # x = Dropout(0.5)(x)
    predictions = Dense(6, activation='softmax')(x)
    model_all = Model(inputs=model_mid.input, outputs=predictions)
    # print(model_all.summary())
    model_mid.load_weights('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    return model_mid, model_all


def build_vgg_raw(img_width, img_height):

    vgg_model = vgg16.VGG16(input_tensor=None, weights='imagenet',
                            include_top=False, input_shape=(img_width, img_height, 3))
    # print(vgg_model.summary())
    # for i, layer in enumerate(vgg_model.layers):
    #     print(i, layer.name)
    x = vgg_model.output

    x = GlobalAveragePooling2D()(x)  # [7,7,512]
    # x = Dropout(0.5)(x)
    # x = Dense(100, activation='relu')(x)  # ACM 去掉这一层
    # x = Dropout(0.5)(x)
    predict_class = Dense(9, activation='sigmoid')(x)
    # predict_attri = Dense(10, activation='sigmoid')(x)
    model_all = Model(inputs=vgg_model.input, outputs=predict_class)


    return vgg_model, model_all


'''
(0, 'input_1')
(1, 'block1_conv1')
(2, 'block1_conv2')
(3, 'block1_pool')
(4, 'block2_conv1')
(5, 'block2_conv2')
(6, 'block2_pool')
(7, 'block3_conv1')
(8, 'block3_conv2')
(9, 'block3_conv3')
(10, 'block3_pool')
(11, 'block4_conv1')
(12, 'block4_conv2')
(13, 'block4_conv3')
(14, 'block4_pool')
(15, 'block5_conv1')
(16, 'block5_conv2')
(17, 'block5_conv3')
(18, 'block5_pool')


Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
=================================================================

'''
