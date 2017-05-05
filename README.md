# Coriander-vs-Parsley

Classification of coriander (قزبر  in Moroccan dialect) and parsley (معدنوس in Moroccan dialect): A modest try of data augmentation with Keras (TensorFlow backend)

@author: Redouane Lguensat

> Data is collected and provided by Ali Lakrakbi https://github.com/alilakrakbi/Coriander-vs-Parsley

Train: 464 images (137 for coriander and 327 for parsley)
Validation: 170 images (77 coriander and 93 parsley)

**Important**
For now, data is not big (Keras example used 1000 examples from both classes) and classes are unbalanced, this is a simple try with a shallow net, I am sure there is a large margin of improvement through the use of class balancing, fine-tuning with deeper nets or just sampling more from the coriander class

This is the directory structure:
```
data/
    train/
        coriander/
            001.jpg
            002.jpg
            ...
        parsley/
           001.jpg
           002.jpg
            ...
    validation/
        coriander/
           001.jpg
           002.jpg
           ...
       parsley/
           001.jpg
           002.jpg
           ...
```
