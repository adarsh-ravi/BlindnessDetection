import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train['diagnosis'].value_counts().plot(kind='bar')
plt.show()

image_list = []
for idx, row in df_train.iterrows():
    img = Image.open('train_images/{}.png'.format(row['id_code']))
    image_list.append(img)
    height, width = img.size
    df_train.loc[idx, 'height'] = height
    df_train.loc[idx, 'width'] = width

df_train['image'] = image_list

plt.figure(2)
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(df_train.iloc[i]['image'])
    plt.title(df_train.iloc[i]['id_code'] + ' ' + str(df_train.iloc[i]['diagnosis']))

plt.show()
