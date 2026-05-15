# PureFER

> **Note**: The full source code of this project is located in the `master` branch.
> The source code and datasets utilized in this study are publicly available at GitHub with the following DOI:https://doi.org/10.5281/zenodo.20174388

## Dataset
The related datasets used in this paper are stored on Baidu Netdisk for easy access:
- Link: https://pan.baidu.com/s/1oFSC92itVViW9cHBrLVGCw
- Extract code: m2s7

The datasets include:
- RAF-DB
- FERPlus
- AffectNet

# Requirements & Dependencies
- PyTorch 2.5.1

# Deep Learning Framework
torch==2.5.1
torchvision>=0.20.1  

# Face Detection & Preprocessing
retina-face
opencv-python>=4.5.0
Pillow>=8.0.0

# Visualization & Metrics
scikit-learn>=1.0.2 
matplotlib>=3.5.0
seaborn>=0.11.2      
grad-cam>=1.4.6      

# Utilities
numpy>=1.21.0
tqdm>=4.62.0
pandas>=1.3.0

## 🛠️ Requirements & Dependencies

The experiments were conducted using **PyTorch 2.5.1** on a single NVIDIA RTX 4060 (8GB) GPU. The environment can be easily set up using the provided `requirements.txt`.

### 1. Prerequisites
- Python 3.8 or higher
- CUDA 11.8 / 12.1 or higher (matching your PyTorch version)

### 2. Installation
We recommend using [Anaconda](https://www.anaconda.com/) to create a virtual environment:

```bash
# Create and activate a new conda environment
conda create -n purefer python=3.9 -y
conda activate purefer

# Install PyTorch (Please adjust the CUDA version according to your local machine)
# Example for CUDA 12.1:
pip install torch==2.5.1 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install the rest of the required dependencies
pip install -r requirements.txt

## Citation
If you find this project useful for your research, please cite our paper:

### Plain Text
Lightweight Facial Expression Recognition via Dual-Domain Compression and Orthogonal Competitive Attention for Edge Deployment. The Visual Computer.

### BibTeX
@article{PureFER2026,
  title={Lightweight Facial Expression Recognition via Dual-Domain Compression and Orthogonal Competitive Attention for Edge Deployment},
  journal={The Visual Computer},
  author={shuo Feng, Ruisheng Jia, Hongmei Sun, shaung Yang, qing Sun},
  year={2026}
  doi={https://doi.org/10.5281/zenodo.20174388}
}
