import numpy as np
import PIL
import streamlit as st
import tensorflow as tf
import tensorflow_hub as tf_hub


@st.cache(show_spinner=False)
def load_img(img_path):
    img=PIL.Image.open(img_path)
    #img=tf.keras.preprocessing.image.load_img(img_path)
    img=np.asarray(img)
    img=np.expand_dims(img,axis=0)/255
    ### converting np array to eager_tensor to make it
    ##compatible with tf models
    img=tf.convert_to_tensor(img,dtype=tf.float32)
    return img

st.title('Neural Style Transfer')
st.subheader('Generate Art without an Artist..')

model_path='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1'

@st.cache(show_spinner=False,allow_output_mutation=True)
def load_my_model():
    model=tf_hub.load(model_path)
    return model

model=load_my_model()

### prevent displaying file_uploader warning
st.set_option('deprecation.showfileUploaderEncoding', False)

content_img=st.file_uploader('Choose Content Image:')
style_img=st.file_uploader('Choose Style Image:')

@st.cache(show_spinner=False)
def Transforming(c_image,s_image):
    #Indicates text until processing..
    with st.spinner('Hold On Transforming!!!...'):
        prediction=model(tf.constant(c_image),tf.constant(s_image))[0]
        pred_img=np.squeeze(prediction)
    return prediction,pred_img
if content_img and style_img is not None:
    content_image=load_img(content_img)
    style_image=load_img(style_img)

    result,res_img=Transforming(content_image,style_image)

    if result is not None:
        st.subheader('Here Is Your Art')
        'Whoaa Look At It.....'
        st.image(res_img,use_column_width=True)

if st.sidebar.checkbox('Show Content Image'):
    st.sidebar.image(content_img,caption='Content Image...',use_column_width=True)
if st.sidebar.checkbox('Show Style Image'):
    st.sidebar.image(style_img,caption='Style Image...',use_column_width=True)



    
