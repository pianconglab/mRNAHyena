# H20安装说明
```
git clone https://github.com/pianconglab/mRNAHyena.git
conda create -n mRNA-hyena python=3.8
conda activate mRNA-hyena
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
conda install cuda -c nvidia/label/cuda-12.4.1
conda install -c conda-forge gcc gxx libxcrypt
cd flash-attention
python setup.py install
cd ..
python -m train wandb=null experiment=prot14m/prot14m_hyena trainer.devices=1
python -m train wandb=null experiment=mRNA/mRNA_hyena trainer.devices=1
```