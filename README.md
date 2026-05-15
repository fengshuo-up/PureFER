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
