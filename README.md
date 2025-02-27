## Single-Image-Reflection-Separation-by-Reflection-and-Refraction-Estimate


# Environment
absl-py==1.4.0
antlr4-python3-runtime==4.9.3
astunparse==1.6.3
backcall==0.2.0
beautifulsoup4==4.12.2
cachetools==5.3.1
certifi==2022.12.7
charset-normalizer==3.0.1
colorama==0.4.6
cycler==0.11.0
decorator==5.1.1
dominate==2.9.1
einops==0.6.0
et-xmlfile==1.1.0
ffmpeg-normalize==1.25.3
ffmpeg-progress-yield==0.5.0
fid-score==0.1.3
filelock==3.12.2
flatbuffers==23.5.26
fonttools==4.38.0
gast==0.4.0
gdown==4.7.1
gitdb==4.0.10
GitPython==3.1.31
google-auth==2.19.0
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.54.2
h5py==3.8.0
hydra-core==1.3.2
idna==3.4
imageio==2.25.0
importlib-metadata==6.6.0
importlib-resources==5.12.0
install==1.3.5
ipdb==0.13.13
ipython==7.34.0
jedi==0.18.2
joblib==1.3.2
keras==2.11.0
kiwisolver==1.4.4
libclang==16.0.6
lpips==0.1.4
Markdown==3.4.3
MarkupSafe==2.1.2
matplotlib==3.5.3
matplotlib-inline==0.1.6
natsort==8.4.0
networkx==2.6.3
numpy==1.21.6
oauthlib==3.2.2
omegaconf==2.3.0
opencv-python==4.7.0.68
openpyxl==3.1.2
opt-einsum==3.3.0
packaging==23.0
pandas==1.3.5
parso==0.8.3
pickleshare==0.7.5
Pillow==9.5.0
pip==22.3.1
prompt-toolkit==3.0.38
protobuf==3.19.6
psutil==5.9.5
pyasn1==0.5.0
pyasn1-modules==0.3.0
pydub==0.25.1
Pygments==2.15.1
pyparsing==3.0.9
PySocks==1.7.1
python-dateutil==2.8.2
python-version==0.0.2
pytorch-msssim==1.0.0
pytz==2023.3
PyWavelets==1.3.0
PyYAML==6.0
requests==2.28.2
requests-oauthlib==1.3.1
rsa==4.9
scikit-image==0.19.3
scikit-learn==1.0.2
scipy==1.4.1
seaborn==0.12.2
setuptools==65.6.3
six==1.16.0
smmap==5.0.0
soupsieve==2.4.1
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorboardX==2.5.1
tensorflow==2.11.0
tensorflow-estimator==2.11.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==2.2.0
tf-slim==1.1.0
thop==0.1.1.post2209072238
threadpoolctl==3.1.0
tifffile==2021.11.2
tomli==2.0.1
torch==1.13.1
torch-summary==1.4.5
torchaudio==0.13.1+cu117
torchfile==0.1.0
torchvision==0.14.1
tqdm==4.64.1
traitlets==5.9.0
typing_extensions==4.7.1
urllib3==1.26.14
Werkzeug==2.2.3
wheel==0.37.1
wincertstore==0.2
wrapt==1.15.0
XlsxWriter==3.2.0
zipp==3.15.0


Please run the follow line to install enviroment
```python

pip install -r requirements.txt

```

# How to try


 
## dataset 
We use the synthetic method from [Zheng et. al] (https://github.com/q-zh/absorption)

[place 365] (https://github.com/CSAILVision/places365)

[RID] (https://github.com/USTCPCS/CVPR2018_attention)

[SIR2](https://www.dropbox.com/scl/fi/qgg1whla1jb3a9cgis18l/SIR2.zip?rlkey=kmhrc2uk63be2s9hzr43gc3hm&e=1&st=cfsh8sol&dl=0)

```python

Data_root/
         -train/
               -syn
               -t
               -r
               -estimate.txt
               ⋮
         -test/
               -syn
               -r
               -t
               ⋮
 ⋮

```

## Training
```python

python train.py --data_root 'data_root' --epochs 200

```

## Run testing
```python

python test.py --data_root 'data_root' --weight_path 'checkpoint_path'

```
