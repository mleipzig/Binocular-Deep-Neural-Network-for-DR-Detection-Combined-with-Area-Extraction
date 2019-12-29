
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import keras
#from keras.applications.vgg19 import VGG19
#from keras.models import Model
#from keras.layers import Dense, Dropout, Flatten

import os
from tqdm import trange
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

def scaleRadius(img,scale):
    x=img[img.shape[0]//2,:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def plot_images(images):
    plt.figure(figsize=(10, 10))
    columns = 9
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)


def get_image_names(df_train):
    image_names=[]
    for i in trange(df_train.values.shape[0]):
        t=df_train.values[i][1].split('/')
        image_names.append(t[len(t)-1].split('.')[0])
        #print(image_names)
        return image_names



def count_average_dimension(df_train,scale):
    image_shape_x=[]
    image_shape_y=[]
    origin_x=[]
    origin_y=[]
    for i in trange(0,df_train.values.shape[0]):
            #img = cv2.imread('C:/Users/User/Desktop/Digital Image Processing/project/images/39_left')
            path = 'C:/Users/User/Desktop/Digital Image Processing/project/images/'+df_train.values[i][1].split('/')[len(df_train.values[i][1].split('/'))-1].split('.')[0]
            #print(path)
            img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            img2 = scaleRadius(img, scale)
            image_shape_x.append(img2.shape[0])
            image_shape_y.append(img2.shape[1])
            origin_x.append(img.shape[0])
            origin_y.append(img.shape[1])
    origin_x_mean=np.mean(np.array(origin_x))
    origin_y_mean=np.mean(np.array(origin_y))
    x_mean=np.mean(np.array(image_shape_x))
    y_mean=np.mean(np.array(image_shape_y))
    x_var=np.var(np.array(image_shape_x))
    y_var=np.var(np.array(image_shape_y))
    return x_mean,y_mean,x_var,y_var,origin_x_mean,origin_y_mean

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
            #print(img.shape)
        return img

def crop_and_Gaussian(path, path_or_img,img,sigmaX=20):
    if (path_or_img==1):
        image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    else:
        image=img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (im_size1,im_size2))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    return image

def circle_crop(path, sigmaX=20):
    """
    Create circular crop around image centre
    """

    img =  cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    #print(img.shape)
    return img




def CLAHE(img):
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    r,g,b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    img = cv2.merge([r,g,b])

    return img


def new_data_augmentation(img):
    #img = crop_and_Gaussian(None,0,img,15,im_size1,im_size2)
    #img = CLAHE(img)
    im = Image.fromarray(np.uint8(img))
    img2 = im.rotate(90)
    img3 = im.rotate(180)
    img4 = im.rotate(270)
    img2 = np.asarray(img2)
    #img2 = rotate(img, angle=90)*256
    #print("img2----------------",img2.max())
    img3 = np.asarray(img3)
    img4 = np.asarray(img4)
    img5 = np.fliplr(img)
    img6 = np.flipud(img)
    img7 = np.fliplr(img2)
    img8 = np.flipud(img2)
    img9 = np.fliplr(img3)
    img10 = np.flipud(img3)
    img11 = np.fliplr(img4)
    img12 = np.flipud(img4)
    img = img.reshape((1,) + img.shape).astype('uint8')
    img2 = img2.reshape((1,) + img2.shape)
    img3 = img3.reshape((1,) + img3.shape)
    img4 = img4.reshape((1,) + img4.shape)
    img5 = img5.reshape((1,) + img5.shape)
    img6 = img6.reshape((1,) + img6.shape)
    img7 = img7.reshape((1,) + img7.shape)
    img8 = img8.reshape((1,) + img8.shape)
    img9 = img9.reshape((1,) + img9.shape)
    img10 = img10.reshape((1,) + img10.shape)
    img11 = img11.reshape((1,) + img11.shape)
    img12 = img12.reshape((1,) + img12.shape)
    #print(img.shape,img2.shape,img3.shape,img4.shape,img5.shape,img6.shape)
    img = np.concatenate((img,img2,img3,img4,img5,img6,img7,img8,img9,img10,img11,img12)).astype('uint8')
    #print("IMG0:----------------",img[0],"IMG1:-----------------------",img[1])
    return img

def create_npy(df_train,df_train2,im_size1,im_size2):
    targets_series = pd.Series(df_train['level'])
    targets_series2 = pd.Series(df_train2['level'])
    #one_hot = pd.get_dummies(targets_series,sparse=True)
    #one_hot_labels = np.asarray(one_hot)
    #print(one_hot_labels.shape)
    x_train=[]
    y_train=[]
    for i in trange(df_train.values.shape[0]):
        if (targets_series[i]==4):
            #img = cv2.imread('C:/Users/User/Desktop/Digital Image Processing/project/images/39_left')
            path = 'C:/Users/User/Desktop/Digital Image Processing/project/images/'+df_train.values[i][1].split('/')[len(df_train.values[i][1].split('/'))-1].split('.')[0]
            #print(path)
            img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = crop_and_Gaussian(path,20)
            #print(img.shape)
            img=new_data_augmentation(img)
            #print(img.shape)
            for j in range(img.shape[0]):
                x_train.append(img[j])
            #x_train.append(img)
            #label = one_hot_labels[i]

    #kaggle数据
    for i in trange(df_train2.values.shape[0]):
        if (targets_series2[i]==4):
            #img = cv2.imread('C:/Users/User/Desktop/Digital Image Processing/project/images/39_left')
            path = 'G:/DIP/image_kaggle/'+df_train2.values[i][0]+'.jpeg'
            #print(path)
            img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=new_data_augmentation(img)
            for j in range(img.shape[0]):
                x_train.append(img[j])
            #label = one_hot_labels[i]

    np.save('G:\\DIP\\NEW\\origin\\x_4',x_train)
    #np.save('y_clahe',y_train)
    print('Done')

def show(df_train):
    targets_series = pd.Series(df_train['level'])
    one_hot = pd.get_dummies(targets_series,sparse=True)
    one_hot_labels = np.asarray(one_hot)
    print(one_hot_labels.shape)
    I=[]
    for i in trange(df_train.values.shape[0]):
        if (df_train['level'][i]==0 and len(I)<=30):
            path = 'C:/Users/User/Desktop/Digital Image Processing/project/images/'+df_train.values[i][1].split('/')[len(df_train.values[i][1].split('/'))-1].split('.')[0]
            print(path)
            img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2=crop_and_Gaussian(path,20)
            img3=cv2.resize(circle_crop(path,20),(im_size1,im_size2))

            I.append(img)
            I.append(img2)
            I.append(img3)

    plot_images(I)
    plt.show()

def datagen(x,y):
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=True,
            fill_mode='constant',
            cval=0)


    first=1
    x1=x[0].reshape((1,) + x[0].shape)
    y1=y[0].reshape((1,) + y[0].shape)
    for i in trange(x.shape[0]):
        j = 0
        '''if(np.argmax(y[i])==0):
            continue
        elif (np.argmax(y[i])==1):
            time=2
        elif (np.argmax(y[i])==2):
            time=1
        elif (np.argmax(y[i])==3):
            time=9
        elif (np.argmax(y[i])==4):
            time=15
        '''
        if (np.argmax(y[i])==1):
            time=2
        else:
            continue
        for x_batch,y_batch in datagen.flow(x[i].reshape((1,) + x[0].shape) ,y[i].reshape((1,) + y[0].shape) , batch_size=1 ):
            if (first==1):
                first=0
                x1=x_batch
                y1=y_batch
            else:
                x1=np.concatenate((x1,x_batch))
                y1=np.concatenate((y1,y_batch))
            j += 1
            if j > time:
                break  # 否则生成器会退出循环

    #x=np.concatenate((x,x1))
    #y=np.concatenate((y,y1))

    print("done")
    np.save('x_rotate1',x1)
    np.save('y_rotate1',y1)
    print("save done")

#df_train = pd.read_excel('C:\\Users\\User\\Desktop\\Digital Image Processing\\project\\label.xlsx')
#df_train2 = pd.read_excel('G:\\DIP\\image_kaggle\\trainLabels.xlsx')
scale=300

im_size1=512
im_size2=512

#show(df_train)
#create_npy(df_train,im_size1,im_size2)


'''
0: 6449+25810=32259 
1: 852+2443=3295  *4 = 13180
2: 1216+5292=6508 *2 = 13016
3: 212+873=1085 *12 = 13020
4: 131+708=839  *12 = 10068
'''
