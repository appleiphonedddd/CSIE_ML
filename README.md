# CSIE


## Contents

- [CSIE](#csie)
  - [Contents](#contents)
    - [Getting Started](#getting-started)
          - [Requirements](#requirements)
          - [Installation](#installation)
    - [Deployment](#deployment)
    - [Author](#author)

### Getting Started

###### Requirements

- **Operating System**: Ubuntu 24.04.03 LTS (Linux-based)
- **GPU**: NVIDIA GeForce RTX 3060 (or higher, CUDA-enabled)
- **CUDA Toolkit**: 12.x (compatible with your GPU driver)

###### Installation

1. Install Conda (If you have already installed this command or Anaconda , you can skip this step!!!!)

```sh
./install_miniconda.sh
```

### Deployment

1. Create a virtual environment and install the Python libraries

```sh
conda env create -f env.yaml
conda activate CSIE
```

2. Run evaluation

```sh
python main.py --dataset mnist --cv 3

python main.py --dataset fashion --cv 3

```

### Author

611221201@gms.ndhu.edu.tw

Egor Alekseyevich Morozov
