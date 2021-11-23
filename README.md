# Learning Temporal Consistency for Video Super-Resolution
> Anonymous code repo

## Dataset

Test data includes REDS4, Vid4, SPMC30, Vimeo90K-T and UDM10. Before evaluation, please download corresponding testing 
and put them in given paths.

| Type                     | Download URL                                | Path               |
| ------------------------ |:-------------------------------------------:|:------------------:|
| REDS4                    | All ground truth sequences in RGB format    |./dataset/REDS4     |
| SPMC30                   | All low quality sequences in RGB format     |./dataset/SPMC30    |
| UDM10                    | All ground truth sequences in YCbCr format  |./dataset/UDM10     |
| Vid4                     | All low quality sequences in YCbCr format   |./dataset/Vid4      |
| Vimeo90K-T               | Ground truth test sequences in RGB format   |./dataset/Vimeo90K-T|


## Code

### Installation
```
# Create a new anaconda python environment (tca)
conda create -n tca python=3.7 

# Activate the created environment
conda activate tca

# Install dependencies
pip install -r requirements.txt
```

### Evaluation

Inference on datasets with BI and BD degradations:
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


## License

This project is released under the Apache 2.0 license.


## Acknowledgement

This implementation largely depends on [BasicSR](https://github.com/xinntao/BasicSR). Thanks for the excellent codebase!