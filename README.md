# Gen_Study: Generative Models Visualization

A comprehensive study and visualization project for various generative models using 1D and 2D synthetic data.

## Project Overview

This project implements and visualizes multiple generative models to understand their behavior, training dynamics, and generation capabilities. It supports both 1D and 2D synthetic data experiments with detailed visualizations and comparison tools.

## Implemented Models

### 1. Basic Models
1. **Non-identifiable Decoder**: A simple decoder that maps random noise to data (demonstrates mode collapse)
2. **Autoencoder**: Encodes data to latent space and reconstructs it
3. **VAE (Variational Autoencoder)**: Regularizes latent space using KL divergence

### 2. Flow-based Models
4. **Flow Matching (FM)**: Learns the probability flow ODE for generative modeling
5. **MeanFlow**: Uses mean velocity for faster sampling while maintaining flow matching
6. **FACM (Flow-Anchored Consistency Models)**: Combines flow matching anchor with consistency model accelerator
7. **BackFlow**: Implements backward flow matching with optimal transport perspective

## Project Structure

```
Gen_Study/
├── data/
│   ├── __init__.py
│   ├── synthetic.py          # 1D/2D synthetic data generation
│   └── synthetic_2d.py       # 2D data utilities
├── models/
│   ├── __init__.py
│   ├── base_mlp.py           # Base MLP architecture
│   ├── decoder.py            # Non-identifiable Decoder
│   ├── autoencoder.py        # Autoencoder
│   ├── vae.py                # Variational Autoencoder
│   ├── flow_matching.py      # Flow Matching model
│   ├── mean_flow.py          # MeanFlow model
│   ├── facm.py               # FACM model
│   └── backflow.py           # BackFlow model
├── training/
│   ├── __init__.py
│   ├── train_decoder.py      # Decoder training
│   ├── train_ae.py           # Autoencoder training
│   ├── train_vae.py          # VAE training
│   ├── train_fm.py           # Flow Matching training (1D/2D)
│   ├── train_fm_2d.py        # Flow Matching training (2D)
│   ├── train_meanflow.py     # MeanFlow training (1D/2D)
│   ├── train_meanflow_2d.py  # MeanFlow training (2D)
│   ├── train_facm.py         # FACM training
│   └── train_backflow.py     # BackFlow training
├── visualization/
│   ├── __init__.py
│   ├── config.py             # Color and style configuration
│   ├── viz_decoder.py        # Decoder visualization
│   ├── viz_ae.py             # Autoencoder visualization
│   ├── viz_vae.py            # VAE visualization
│   ├── viz_fm.py             # Flow Matching visualization (1D)
│   ├── viz_fm_2d.py          # Flow Matching visualization (2D)
│   ├── viz_meanflow.py       # MeanFlow visualization (1D)
│   ├── viz_meanflow_2d.py    # MeanFlow visualization (2D)
│   ├── viz_facm.py           # FACM visualization (1D)
│   ├── viz_facm_2d.py        # FACM visualization (2D)
│   ├── viz_backflow.py       # BackFlow visualization (1D)
│   └── viz_backflow_2d.py    # BackFlow visualization (2D)
├── outputs/                  # Trained models and visualization results
│   ├── comparison_1d/        # 1D comparison experiments
│   └── comparison_2d/        # 2D comparison experiments
├── run_comparison.py         # Comparison experiments script
├── main.py                   # Main experiment execution script
├── requirements.txt
└── README.md
```

## Installation

### 1. Create and Activate Conda Environment

```bash
# Navigate to Gen_Study directory
cd /home/user/Desktop/Gen_Study

# Create conda environment (Python 3.10)
conda create -n gen_study python=3.10 -y

# Activate environment
conda activate gen_study
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Deactivate Environment (when finished)

```bash
conda deactivate
```

## Usage

**Note**: All commands should be executed with the `gen_study` conda environment activated.

```bash
# Activate environment
conda activate gen_study
```

### Run All Experiments

```bash
# Run all models sequentially
python main.py

# Run specific model
python main.py --model facm
```

Available models: `decoder`, `ae`, `vae`, `vae_beta`, `fm`, `meanflow`, `facm`, `backflow`

### Run Comparison Experiments

```bash
# Run 1D comparison (all models)
python run_comparison.py --dim 1d --epochs 200

# Run 2D comparison (all models)
python run_comparison.py --dim 2d --epochs 200

# Run both 1D and 2D
python run_comparison.py --dim both --epochs 200

# Run specific models only
python run_comparison.py --dim 1d --models fm meanflow facm backflow --epochs 200
```

### Train Individual Models

```bash
# Decoder
python training/train_decoder.py

# Autoencoder
python training/train_ae.py

# VAE
python training/train_vae.py

# Flow Matching
python training/train_fm.py

# MeanFlow
python training/train_meanflow.py

# FACM
python training/train_facm.py

# BackFlow
python training/train_backflow.py
```

### Test Data Generation

```bash
# Test 1D data generation
python data/synthetic.py

# Test 2D data generation
python data/synthetic_2d.py
```

## Configuration

### Data Settings

**1D Data (2-Gaussian Mixture)**:
- Gaussian 1: mean=-2.0, std=0.8 (70%)
- Gaussian 2: mean=2.0, std=0.6 (30%)
- Sample size: 500 (default)

**2D Data**:
- Configurable 2D distributions
- Sample size: 500 (default)

### Model Settings

- **MLP Architecture**: [32, 64, 32] hidden layers (default)
- **Latent Dimension**: 1D or 2D (depending on experiment)
- **Input/Output Dimension**: Matches data dimension

### Training Settings

- **Optimizer**: Adam
- **Learning Rate**: 1e-3 (default)
- **Batch Size**: 64 (default)
- **Epochs**: 2000 (default for main.py), 200 (default for run_comparison.py)

### Visualization Color Settings

Colors are centrally managed in [`visualization/config.py`](visualization/config.py):

```python
COLORS = {
    'prior_z': '#1f77b4',        # blue - prior distribution z
    'data_x': '#ff7f0e',         # orange - data x
    'latent_z_hat': '#2ca02c',   # green - encoded ẑ
    'output_x_hat': '#d62728',   # red - reconstructed x̂
    'coupling_train': '#7f7f7f', # gray - training coupling
    'coupling_infer': '#9467bd', # purple - inference mapping
    'encode_line': '#8c564b',    # brown - encoding line
    'decode_line': '#e377c2',    # pink - decoding line
}
```

## Visualization Details

### Basic Models (Decoder, AE, VAE)

Each visualization shows two spaces in a single image:
- **Left Panel**: Latent/source space (z, ẑ)
- **Right Panel**: Data/target space (x, x̂)
- **Connecting Lines**: Mapping relationships

#### 1. Decoder Visualization
- Random coupling during training (gray lines)
- Inference mapping (purple lines)
- Mode collapse: outputs converge to data mean

#### 2. Autoencoder Visualization
- Encoding: x → ẑ (brown lines)
- Decoding: ẑ → x̂ (pink lines)
- Latent space ẑ may differ from prior distribution

#### 3. VAE Visualization (β=1.0)
- Same structure as Autoencoder
- KL regularization makes ẑ closer to prior N(0,1)
- Trade-off between reconstruction quality and regularization

#### 4. VAE with High Beta (β=10.0)
- Stronger KL regularization
- ẑ even closer to prior N(0,1)
- Stronger regularization effect compared to β=1.0
- Slight reconstruction quality decrease but improved generation quality

### Flow-based Models (FM, MeanFlow, FACM, BackFlow)

Visualizations show:
- **Trajectories**: ODE integration paths from noise to data
- **Training Data**: Source and target distributions
- **Inference Samples**: Generated samples from prior
- **One-step Predictions**: Direct mappings (for consistency models)

#### Flow Matching (FM)
- Shows probability flow trajectories
- Demonstrates smooth interpolation from noise to data

#### MeanFlow
- Shows mean velocity trajectories
- Faster sampling with multi-step mean velocity ODE

#### FACM
- Combines flow matching anchor with consistency model
- Shows both ODE trajectories and one-step consistency predictions
- Time-dependent weighting: CM loss stronger at t=0, FM loss at t=1

#### BackFlow
- Shows backward flow matching trajectories
- Optimal transport perspective
- One-step decode capability

## Output Files

After running experiments, the following files are generated in `outputs/`:

### Individual Model Outputs
- `*_model.pt` - Trained model checkpoints
- `*_visualization.png` - Model visualizations

### Comparison Experiments
- `comparison_1d/comparison_frames/epoch_*.png` - Epoch-by-epoch comparison frames
- `comparison_1d/comparison_1d.gif` - Animated comparison GIF
- `comparison_2d/comparison_frames/epoch_*.png` - Epoch-by-epoch comparison frames
- `comparison_2d/comparison_2d.gif` - Animated comparison GIF
- `*_epochs/epoch_*.png` - Individual model epoch visualizations

## Model-Specific Details

### FACM Loss Weighting

FACM uses a time-dependent weighting scheme:
- **FM Loss**: Constant weight (anchor)
- **CM Loss**: Weighted by `β = cos(t * π/2)`
  - t=0: β=1.0 (maximum CM weight)
  - t=1: β=0.0 (minimum CM weight)
- **Total Loss**: `L_total = L_CM.mean() + L_FM.mean()`

This design allows CM loss to accelerate learning early (t≈0) while FM loss stabilizes training later (t≈1).

## Requirements

- Python 3.10+
- PyTorch 2.9+
- NumPy 2.2+
- Matplotlib 3.10+
- Pillow 12.1+
- SciPy 1.15+

See `requirements.txt` for exact versions.

## License

This project is freely available for educational and research purposes.
