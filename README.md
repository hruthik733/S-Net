# S-Net
S-Net: A Novel Shallow Network for Enhanced Detail Retention in Medical Image Segmentation

### Environment install
```
conda create -n your_env_name python=3.10.13
conda activate your_env_name
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
```

### Install mamba
```
pip install causal-conv1d==1.1.1  
pip install mamba-ssm==1.1.3.post1 
```

## Usage
运行mian.py文件完成训练后自动进行测试

### train_test ISIS2018
```
python main.py --data ISIC2018_png_224 --epochs 120 --batch_size 16
```
### train_test Kvasir
```
python main.py --data Kvasir_png_224 --epochs 120 --batch_size 16
```
### train_test BUSI
```
python main.py --data BUSI_png_224 --epochs 120 --batch_size 16
```
