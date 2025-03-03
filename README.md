## Single-Image-Reflection-Separation-by-Reflection-and-Refraction-Estimate


# Environment

Pillow==9.5.0
python-version==0.0.2
scikit-image==0.19.3
tensorflow==2.11.0

Please run the following line to install enviroment
```python

pip install -r requirements.txt

```

 
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




## Citation
If you use this code for your research, please cite our paper :

```
@inproceedings{chan2025single,
  title={Single Image Reflection Separation by Reflection and Refraction Estimate},
  author={Chan, U-In and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  year={2024}
}
```
