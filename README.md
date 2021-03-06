# Learning Temporal Consistency for Video Super-Resolution
> official implementation

## GIF Visualization of BasicVSR and Ours

__*GIF Visualization of animated figure, Fig.1 and Fig.7-Fig9, in our paper.*__

<p align="left">
  <a href="https://github.com/mrluin/TCA">
    <img width=90% src="https://github.com/mrluin/TCA/blob/main/fig1_comparison.gif"/>
  </a>

  Fig. 1. Visual comparison of BasicVSR (top-right) and our TCA learning (bottom-right). The left if HR video. BasicVSR suffers from flickering artifacts, while our TCA learning can better keep temporal consistency of video results.
</p>



<p align="left">
  <a href="https://github.com/mrluin/TCA">
    <img width=90% src="https://github.com/mrluin/TCA/blob/main/fig7_comparison.gif"/>
  </a>
  
  Fig. 7. Visual comparison of BasicVSR (left, red rectangle) and our TCA learning (right, cyan-blue rectangle). In rectangle region, BasicVSR suffers from flickering artifacts, while in our TCA learning flickering artifacts are effectively alleviated.
</p>

<p align="left">
  <a href="https://github.com/mrluin/TCA">
    <img width=90% src="https://github.com/mrluin/TCA/blob/main/fig8_comparison_l.gif"/>
  </a>
  
  Fig. 8. Visual comparison of BasicVSR (left, top red rectangle) and our TCA learning (right, bottom cyan-blue rectangle). In rectangle region, BasicVSR suffers from flickering artifacts, while in our TCA learning flickering artifacts are effectively alleviated.
</p>

<p align="left">
  <a href="https://github.com/mrluin/TCA">
    <img width=90% src="https://github.com/mrluin/TCA/blob/main/fig9_comparison.gif"/>
  </a>
  
  Fig. 9. Visual comparison of BasicVSR (left, top red rectangle) and our TCA learning (right, bottom cyan-blue rectangle). In rectangle region, BasicVSR suffers from flickering artifacts, while in our TCA learning flickering artifacts are effectively alleviated.
</p>

## Dataset

Test data includes REDS4, Vid4, SPMC30, Vimeo90K-T and UDM10. Before evaluation, please download corresponding testing 
and put them in given paths.

| Type                     | Download URL                                | Path                                                      |
| ------------------------ |:-------------------------------------------:|:-----------------------------------------:|
| REDS4                    | [REDS4 download](https://seungjunnah.github.io/Datasets/reds.html)                      |./dataset/REDS4     |
| SPMC30                   | [SPMC30 download](https://github.com/jiangsutx/SPMC_VideoSR)                            |./dataset/SPMC30    |
| UDM10                    | [UDM10 download](https://github.com/psychopa4/PFNL)          |./dataset/UDM10           |
| Vid4                     | [Vid4 download](https://drive.google.com/file/d/1ZuvNNLgR85TV_whJoHM7uVb-XW1y70DW/view) |./dataset/Vid4      |
| Vimeo90K-T               | [Vimeo90K-T download](http://toflow.csail.mit.edu/)                                     |./dataset/Vimeo90K-T|

For BI degradation, we adopt matlab imresize function. 

For BD degradation, we adopt `BD_degradation.m`.

Note that LR of SPMC30 has pixel shift, you should use standard imresize function to obtain LR frames.

## Code

### installation
```
# create conda env
conda create -n tca python=3.7

# install pytorch, torchvision, cupy
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch
conda install -c conda-forge cupy cudatoolkit=10.2

# install other dependencies
pip install pyyaml, wandb, opencv-python, tqdm, lpips
``` 
### Checkpoints
Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1rQV0gfQWUrFt5hzNkklWcs_zVOftnPeN?usp=sharing), and put them in `./chekpoints`.
### Evaluation

Inference model on testing data:
```
python inference.py --opt ./options/tca_reds_bi4x_reds4_inference.yml

python inference.py --opt ./options/tca_vimeo90k_bi4x_vid4_vimeo90kt_spmc30_inference.yml

python inference.py --opt ./options/tca_vimeo90k_bd4x_udm10_vid4_vimeo90kt_spmc30.yml
```

### Metrics

Calculate PSNR, SSIM, LPIPS and flow warping error:
```
python metrics_calculator.py

python flow_warping_error_calculator.py
```

### Qualitative Comparison
Animated comparison results please refer to [Google Drive](https://drive.google.com/drive/folders/1bO6Sfl54QuO5PD3NNF4ZutZ-4Geg2Qnd?usp=sharing).


## License

This project is released under the Apache 2.0 license.


## Acknowledgement

This implementation largely depends on [BasicSR](https://github.com/xinntao/BasicSR). Thanks for the excellent codebase!
