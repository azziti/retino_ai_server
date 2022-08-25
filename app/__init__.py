





from .model import load_model,prepare_image,predict
#    ^ here do relative import
# or
from app.model import load_model,prepare_image,predict
# because your package named 'core' and importing looks in root folder


"""
import os    
import pandas as pd
import matplotlib.pyplot as plt
    
    
data = "app//kaggle_retinopathy_diabetic"
print('number of images in total - ',len(os.listdir(data)))

index = pd.read_csv("app//trainLabels.csv") 
print('number of images in total - ',len(index))

#index.info()

temp = data + '//10_left' + '.jpeg'

plt.imshow(plt.imread(temp))
 """  