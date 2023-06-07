import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import pickle

st.title("COVID-19 Prediction Using Ensemble Learning")
st.caption("This is a simple image classification web app to predict rock-paper-scissor hand sign")
file = st.file_uploader(label="Please upload CT image file", type=["jpg", "png"])

if file is None:
    st.text("")
else:
    cov = 0
    noncov = 0
    image = Image.open(file)
    st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image_np = np.asarray(image)
    img_pil = Image.fromarray(image_np)
    img_25x25 = np.array(img_pil.resize((25,25), Image.Resampling.LANCZOS))
    img_25x25 = (img_25x25.flatten())
    img_25x25  = img_25x25.reshape(-1,1).T

    his_equ = cv2.equalizeHist(image_np)

    ret,th = cv2.threshold(his_equ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_pil = Image.fromarray(th)
    img_heseg_25x25 = np.array(img_pil.resize((25,25), Image.Resampling.LANCZOS))
    img_heseg_25x25 = (img_heseg_25x25.flatten())
    img_heseg_25x25  = img_heseg_25x25.reshape(-1,1).T

    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize=(8,8))
    ah_equ = clahe.apply(image_np)
    img_pil = Image.fromarray(ah_equ)
    img_ah_25x25 = np.array(img_pil.resize((25,25), Image.Resampling.LANCZOS))
    img_ah_25x25 = (img_ah_25x25.flatten())
    img_ah_25x25  = img_ah_25x25.reshape(-1,1).T

    ret,th = cv2.threshold(ah_equ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_pil = Image.fromarray(th)
    img_ahseg_25x25 = np.array(img_pil.resize((25,25), Image.Resampling.LANCZOS))
    img_ahseg_25x25 = (img_ahseg_25x25.flatten())
    img_ahseg_25x25  = img_ahseg_25x25.reshape(-1,1).T


    #Normal LDA - NB
    with open('sc_nor.pk', 'rb') as pickle_file:
        sc_X = pickle.load(pickle_file)
    img_sc=sc_X.transform(img_25x25)
    with open('lda_nor.pk', 'rb') as pickle_file:
        lda_model = pickle.load(pickle_file)
    img_lda=lda_model.transform(img_sc)
    with open('nb_nor.pk', 'rb') as pickle_file:
        nb = pickle.load(pickle_file)
    y_pred=nb.predict(img_lda)
    st.write(y_pred[0])
    if(y_pred[0]=='Covid'):
        cov = cov + 1
    else:
        noncov = noncov + 1

    #HE Seg LDA - Rdge
    with open('sc_heseg.pk', 'rb') as pickle_file:
        sc_X = pickle.load(pickle_file)
    img_sc=sc_X.transform(img_heseg_25x25)
    with open('lda_heseg.pk', 'rb') as pickle_file:
        lda_model = pickle.load(pickle_file)
    img_lda=lda_model.transform(img_sc)
    with open('rdge_heseg.pk', 'rb') as pickle_file:
        rdgclassifier = pickle.load(pickle_file)
    y_pred=rdgclassifier.predict(img_lda)
    st.write(y_pred[0])
    if(y_pred[0]=='Covid'):
        cov = cov + 1
    else:
        noncov = noncov + 1

    #HE Seg PCA - DT
    with open('pca_heseg.pk', 'rb') as pickle_file:
        pca_model = pickle.load(pickle_file)
    img_pca=pca_model.transform(img_sc)
    with open('dt_heseg.pk', 'rb') as pickle_file:
        dt = pickle.load(pickle_file)
    y_pred=dt.predict(img_pca)
    st.write(y_pred[0])
    if(y_pred[0]=='Covid'):
        cov = cov + 1
    else:
        noncov = noncov + 1

    #AH Seg LDA - SVM
    with open('sc_ahseg.pk', 'rb') as pickle_file:
        sc_X = pickle.load(pickle_file)
    img_sc=sc_X.transform(img_ahseg_25x25)
    with open('lda_ahseg.pk', 'rb') as pickle_file:
        lda_model = pickle.load(pickle_file)
    img_lda=lda_model.transform(img_sc)
    with open('svm_ahseg.pk', 'rb') as pickle_file:
        svm = pickle.load(pickle_file)
    y_pred=svm.predict(img_lda)
    st.write(y_pred[0])
    if(y_pred[0]=='Covid'):
        cov = cov + 1
    else:
        noncov = noncov + 1
    
    #AH PCA - DT
    with open('sc_ahnor.pk', 'rb') as pickle_file:
        sc_X = pickle.load(pickle_file)
    img_sc=sc_X.transform(img_ah_25x25)
    with open('pca_ahnor.pk', 'rb') as pickle_file:
        pca_model = pickle.load(pickle_file)
    img_pca=pca_model.transform(img_sc)
    with open('dt_ahnor.pk', 'rb') as pickle_file:
        dt = pickle.load(pickle_file)
    y_pred=dt.predict(img_pca)
    st.write(y_pred[0])
    if(y_pred[0]=='Covid'):
        cov = cov + 1
    else:
        noncov = noncov + 1

    #st.write('Cov '+ str(cov) +'  Noncov '+ str(noncov))

    if(cov>noncov):
        st.write("Prediction: Covid")
    else:
        st.write("Prediction: Non-Covid")
    
    if(file.name[0] == 'C' or file.name[0] == 'p'):
        st.write("Truth: Covid")
    else:
        st.write("Truth: Non-Covid")
    


