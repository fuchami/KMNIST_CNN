# coding-utf8
"""
CNN + XGBoost
"""

#%%
import os,sys
import numpy as np
import pandas as pd
from protos import load
import keras
from keras.models import Model
import xgboost as xgb
def expand(x):
    return np.expand_dims(np.expand_dims(x, axis=0), axis=3)


#%%
""" dataset load """
kmnist_dl = load.KMNISTDataLoader(validation_size=0.2, img_resize=38)
train_x, train_y, valid_x, valid_y = kmnist_dl.load('./input/')
train_x, train_y, valid_x, valid_y = load.Preprocessor().transform(train_x, train_y, valid_x, valid_y)
print('train_x:', train_x.shape)
""" testdata load """
test_x = load.test_load(imgsize=38)

#%%
""" load model"""
model_path = "./train_log/wrn-4-8-SE-True-NeXt-True_imgsize38_batchsize128_adabound_fullTrain_custom_expandConv/model.h5"
model = keras.models.load_model(model_path, compile=False)
# model.summary()

#%%
""" extra feature """
feature_extra_model = Model(inputs=model.input,
    outputs=model.get_layer("global_average_pooling2d_148").output)

#%%
train_feature = feature_extra_model.predict(train_x)
print("train_feature.shape: ", train_feature.shape)
valid_feature = feature_extra_model.predict(valid_x)
print("valid_feature.shape", valid_feature.shape)
test_feature = feature_extra_model.predict(test_x)
print("test_feature.shape", test_feature.shape)

#%%
""" Applying XGBOOST """
xb = xgb.XGBClassifier()
print("--- train XGBOOST ---")
xb.fit(train_feature, np.argmax(train_y, axis=1))
print("--- fitting done !!! ---")

#%%
""" prediction score """
print(np.argmax(train_y, axis=1))
score_train = xb.score(train_feature, np.argmax(train_y, axis=1))
print("score_train:", score_train)

score_valid = xb.score(valid_feature, np.argmax(valid_y, axis=1))
print("score_valid:", score_valid)

predicts = xb.predict(test_feature)

#%%
""" submit file """
print('predicts.shape: ', predicts.shape) # hope shape(10000, )
print("predicts", predicts)
submit = pd.DataFrame(data={"ImageId": [], "Label": []})
submit.ImageId = list(range(1, predicts.shape[0]+1))
submit.Label = predicts

submit.to_csv("./output/CNNFeature_xgboost.csv", index=False)

print('--- end train... ---')




#%%
