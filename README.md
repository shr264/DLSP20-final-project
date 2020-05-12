# DLSP20-final-project


Change directory (`cd`) into the folder, then type:



```

conda env create -f pdl.yml

source activate pDL

```



Then we can simply run the scripts (for example UnetLite for Bounding Box prediction) as follows:

```

python CNN-UNET-BB-200.py

```

The python scripts of interest are: 
1. CNN-VAE-BB.py (Bounding Box prediction)
2. CNN-VAE-Road.py (Roadmap prediction)
3. CNN-UNET-Road-200.py (Roadmap prediction)
4. CNN-UNET-BB-200.py (Bounding Box prediction)
5. LEARNROT-CNNVAE.py (Rotation learning for VAE)
6. LearnRot-Unet-200.py (Rotation learning for Unet)