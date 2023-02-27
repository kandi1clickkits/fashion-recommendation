# Fashion Recommendation Engine

## Overview

  The repository has the source code for training as well as providing recommendations based on the input product text or input product image. 

## Getting Recommendations
  
  The repository has 3 notebooks for getting recommendations:
  
      1. Fashion recommendation image.ipynb -> Jupyter notebook which has the source code of providing recommendations for an input Product image from the Fashion world.
      2. Fashion recommendation text.ipynb -> Jupyter notebook which has the source code of providing recommendations for an input Product text from the Fashion world.
      3. Fashion Recommendation Service.ipynb -> Jupyter notebook which has the source code of that triggers a Flask service for providing recommendations. 
      
      The Flask service runs at http://127.0.0.1:5000 at with two separate endpoints (One for image based recommendation and one for text based recommendation).
      
### Training with Custom data
    
    The repository has pre-trained models on Images and Product texts pertaining to the Fashion world. 
    The user can create new custom embeddings by training custom data, either using the Jupyter notebooks or Python files.
    
There a two training Jupyter notebooks:
    1. Image Embedding Training.ipynb -> For creating Image embeddings using custom Product Image data.
    2. Text embedding training.ipynb -> For creating text embeddings using custom Product Text attributes data.
    
There are two training Python files:
    1. Image Embedding Training.py -> For creating Image embeddings using custom Product Image data.
    2. Text embedding training.py -> For creating text embeddings using custom Product Text attributes data.
