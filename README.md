# DAPAO

Code for no-box watermark removal and forgery using pretrained classification
features. Attack optimization does not access the target watermark encoder,
decoder, gradients, or query feedback.

## Removal

`remove.py` generates perturbations for watermarked images

## Forgery

`forgery.py` uses watermarked donor images and clean carrier images to attempt
watermark forgery with two methods:

- `dual`: optimizes two perturbations for the donor and carrier.
- `transfer`: transfers a donor-derived perturbation to the carrier.


## Supporting Code

| File | Purpose |
|---|---|
| `launcher.py` | Validates inputs, pairs donors and carriers, and manages GPU execution and outputs. |
| `color.py` | Provides DenseNet feature extraction, channel clustering and perturbation projection. |
| `dp_trans.py` | Loads donor and carrier images with the shared preprocessing. |
| `pytorch_ssim.py` | Implements the SSIM objective used by the attacks. |
| `summarize_attack_validation.py` | Combines externally computed decoder results and attack records to summarize success rates and quality metrics. |
