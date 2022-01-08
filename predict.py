import numpy as np
from PIL import Image
import os
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json

#savedModel_directory='./best_model.h5'
#model = tf.keras.models.load_model((savedModel_directory),custom_objects={'KerasLayer':hub.KerasLayer})

#test_image_path = os.path.abspath(os.getcwd()) +'/test_images'


image_size=224
def process_image(test_image):
  test_image=tf.convert_to_tensor(test_image, np.float32)
  test_image=tf.image.resize(test_image,size=[image_size,image_size]).numpy()
  test_image/=255
  return(test_image)


#image_path = test_image_path + '/hard-leaved_pocket_orchid.jpg'#'./test_images/hard-leaved_pocket_orchid.jpg'
#im = Image.open(image_path)
#test_image = np.asarray(im)

#processed_test_image = process_image(test_image)

def predict(image_path,model,top_k=4):
  probs=[]
  classes=[]
  im = Image.open(image_path)
  test_image = np.asarray(im)
  processed_test_image = process_image(test_image)
  img=np.expand_dims(processed_test_image,axis=0)
  ps = model.predict(img).reshape(-1)
  max_arr=np.argsort(-1*ps)[0:top_k]+1
  max_arr=max_arr.reshape(-1,)
  max_arr=list(map(str,max_arr.tolist()))
  for i in max_arr:
    #classes.append(class_names[i])
    classes.append(i)
    probs.append(ps[int(i)-1])

  return(probs,classes)

'''
if __name__=="__main__":
    image_path=input("Enter image path:")
    model=input("Enter model:")
    top_k=input("Enter top k:")

    test_image_path = os.path.abspath(os.getcwd()) + image_path
    savedModel_directory='./' + model
    mod = tf.keras.models.load_model((savedModel_directory),custom_objects={'KerasLayer':hub.KerasLayer})
    n_flowers=int(top_k)

    probs, classes = predict(test_image_path, mod, n_flowers)
    #print(probs)
    print(classes)
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",help="image")
    parser.add_argument("model", help="model")
    parser.add_argument("--top_k",help="top_predictions")
    parser.add_argument("--category_names",help="flowers")

    args = parser.parse_args()

    test_image_path = os.path.abspath(os.getcwd()) + args.image_path
    savedModel_directory='./' + args.model
    mod = tf.keras.models.load_model((savedModel_directory),custom_objects={'KerasLayer':hub.KerasLayer})
    n_flowers=int(args.top_k)

    probs, classes = predict(test_image_path, mod, n_flowers)
    #print(probs)
    #print(classes)
    #args.category_names
    

    if args.category_names:
        class_names=None
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)

        # corresponding class names
        clname=[]
        for i in classes:
            clname.append(class_names[i])
        print(clname)
        print(probs)
    else:
        print(classes)
        print(probs)
