# Swin Transformer on CIFAR-10

This Project deals with optimization of the last layer weights via Genetic Algorithms or Hybrid Algorithms such as Cuckoo Search + GA with SAA and CMA_ES + DE.
The following code was used to generate the log file of result and stored the result in the .txt files and plotting was done via visual.py
## Project Structure

```
├── main.py
├── model
│   └── swin_vit.py
├── requirements.txt
└── utils
    ├── autoaug.py
    ├── cutmix.py
    ├── dataloader.py
    ├── loss.py
    ├── optimizer.py
    ├── parser.py
    ├── random_erasing.py
    ├── sampler.py
    ├── scheduler.py
    ├── train_functions.py
    ├── transforms.py
    └── utils.py
```

## Usage

### Install Dependencies
```bash
# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

<hr>

## Usage
To replicate the reported results, run `main.py` with the following hyperparameters:

```bash
python main.py  --patch_size 2 \
                --weight_decay 0.1 \
                --batch_size 128 \
                --epochs 200 \
                --lr 0.001 \
                --warmup_epochs 10 \
                --min_lr 1e-6 \
                --clip_grad 3.0 
```
