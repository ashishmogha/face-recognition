import cv2
import numpy as np
# from keras.models import load_model

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

d_faces = np.load('./dataset/face_data.npy')
l_faces = np.load('./dataset/face_lab.npy')
# model = load_model('cnn/face.h5')


def dist(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum())

def knn(X_train, x, y_train, k=5):
    vals = []
    
    for ix in range(X_train.shape[0]):
        v = [dist(x, X_train[ix].flatten()), y_train[ix]]
        vals.append(v)
    
    updated_vals = sorted(vals, key=lambda x: x[0])
    pred_arr = np.asarray(updated_vals[:k])
    pred_arr = np.unique(pred_arr[:, 1], return_counts=True)
    print pred_arr
    pred = pred_arr[1].argmax()
    return pred_arr[0][pred]


def get_name(im):
    res = knn(d_faces, im, l_faces, k=5)
    # im = im[np.newaxis, np.newaxis, :, :]
    # res = model.predict_classes(im,verbose=0)
    # print res
    # if res == 1:
    #     res = 'Laksh'
    # elif res == 0:
    #     res = 'Ashish'
    return res


def recognize_face(im):
    im = cv2.resize(im, (100, 100))
    im = im.flatten()
    return get_name(im)

while True:
    _, fr = rgb.read()
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        fc = fr[y:y+h, x:x+w, :]
        gfc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', fc) 
        out = recognize_face(gfc)
        cv2.putText(fr, out, (int(x), int(y)), font, 1, (255, 255, 0), 2)
    	cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow('rgb', fr)
    # cv2.imshow('gray', gray)
    k = cv2.waitKey(1)
    if k==27:   
        break
    elif k==-1:
        continue
    else:
        # print k
        continue

cv2.destroyAllWindows()
