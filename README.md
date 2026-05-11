This open-source repository PureFER is directly associated with the manuscript submitted to The Visual Computer.If you use this code for your research or projects, please cite the corresponding paper. Thank you!
1.structure：

PureFER/

├── checkpoints/             # Pre-trained model weights and evaluation visualization charts (CAM / Confusion Matrix)

│   ├── rafdb_acc0.8985.pth  

│   ├── FERPlus_acc0.9126.pth

│   ├── affectnet7...        

├── networks/                # Core Model Architecture Code

│   ├── DDCM.py              

│   └── MixedFeatureNet.py   

├── pretrained/              # Directory for Pre-trained Base Weights

│   └── MFN_msceleb.pth      # Backbone pre-trained weights on Ms-Celeb-1M

├── affectnet_train.py       

├── affectnet_test.py        

├── ferPlus_train.py        

├── ferplus_confusion.py    

├── rafdb_train.py           

├── rafdb_test.py           

└── sam.py                   # Sharpness-Aware Minimization (SAM) 

2.Installation：

git clone https://github.com/yourusername/PureFER.git

cd PureFER

#Install dependencies (PyTorch 2.5.1 & compatible CUDA)

pip install torch torchvision numpy opencv-python pandas

3.Data Preparation：
RAF-DB   

FERPlus   

AffectNet-7 / AffectNet-8

4.Pre-progress：

All input images are required to extract facial regions using RetinaFace and be uniformly resized to 112 × 112 pixels.

5.Quick Start：

5.1： Before training starts, initialize the backbone network. Please place the pre-trained backbone weight file MFN_msceleb.pth (trained on the Ms-Celeb-1M dataset) into the pretrained/ folder to        guarantee the feature extraction capability of model initialization.

5.2： The project provides independent training scripts for different datasets. Taking RAF-DB as an example:
      python rafdb_train.py
      Hyperparameter Description:The SAM optimizer is adopted by default. We jointly optimize the standard cross-entropy loss (Lcls) and the region-focused loss (LRF) with a weight λ=0.5. The initial           learning rate is set to 0.00001 and the batch size
      
5.3： Testing & Evaluation
      We have provided the pre-trained weights that achieve SOTA performance on all datasets in the checkpoints/ directory.To evaluate the accuracy on the RAF-DB test set, please run:
      python rafdb_test.py --checkpoint checkpoints/rafdb_acc0.8985.pth
