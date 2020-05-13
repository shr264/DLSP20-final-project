# DLSP20-final-project
====================

#### Syed Rahman, Andrew Hopen, Eric Kosgey 

#### 5/12/2020

How to run the code:
--------------------

Change directory (`cd`) into the folder, then type:

```
    conda env create -f pdl.yml

    source activate pDL
```

Then we can simply run the scripts (for example UnetLite for Bounding
Box prediction) as follows:

```
    python CNN-UNET-BB-200.py
```

The python scripts of interest are:

1.  CNN-VAE-BB.py (Bounding Box prediction)
2.  CNN-VAE-Road.py (Roadmap prediction)
3.  CNN-UNET-Road-200.py (Roadmap prediction)
4.  CNN-UNET-BB-200.py (Bounding Box prediction)
5.  LEARNROT-CNNVAE.py (Rotation learning for VAE)
6.  LearnRot-Unet-200.py (Rotation learning for Unet)


## Self Supervised Road Map Prediction and Vehicle Detection


We tried two approaches for detecting the binary road-maps using Semantic Segmentation: 

1. CNN-VAE
2. UNetLite

These can easily be extended to bounding box detection as well. For bounding box detection we chose to fit a YOLO model.


