# Variational Autoencoder with Arbitrary Conditioning in TensorFlow 2.0

## Installation
```bash
pip install virtualenv
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

## How to run
### Prepare CelebA dataset
```bash
python run.py --mode prepare
```

### Debug
```bash
python run.py --mode debug
```

### Training
```bash
python run.py --mode train
```

**References**
