# ECG Arrhythmia Classification - Dendritic Neural Network Hackathon

## 🎯 Project Goal
Demonstrate how **Perforated AI's Artificial Dendrite Network** improves neural network performance **without changing the core architecture**.

## 🧠 What are Artificial Dendrites?

In biological neurons, dendrites perform local computations before signals reach the cell body. Perforated AI's open-source library brings this concept to artificial neural networks:

- **Local synaptic computation** - Processing happens at connection points
- **Automatic dendrite addition** - The library automatically adds dendrites when training plateaus
- **Improved performance** - Same architecture achieves better results

## 📊 Models Comparison

| Model | Description |
|-------|-------------|
| **Model 1: Baseline** | Standard PyTorch LSTM with early stopping |
| **Model 2: Dendritic** | **Same architecture** + Perforated AI dendrites |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Perforated AI (from cloned repo)

```bash
cd PAI_repo
pip install -e .
cd ..
```

### 3. Run Training

```bash
python train.py
```

## 📁 Project Structure

```
dendrites/
├── train.py                          # Main training script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── PAI_repo/                         # Perforated AI library (cloned)
│   ├── perforatedai/                 # Core PAI modules
│   ├── Examples/                     # PAI examples
│   └── API/                          # PAI documentation
└── MIT-BIH Arrhythmia Database.csv/  # ECG dataset
    └── MIT-BIH Arrhythmia Database.csv
```

## 📈 Expected Output

After training, you'll get:

- `model_baseline.pth` - Baseline model weights
- `model_dendritic.pth` - Dendritic model weights  
- `training_comparison.png` - Training curves comparison
- `confusion_matrices.png` - Model predictions visualization
- `accuracy_comparison.png` - Final accuracy comparison bar chart
- `results.json` - Numerical results

## 🔬 Technical Details

### Architecture (Same for Both Models)
- **Bidirectional LSTM** (2 layers, 128 hidden units)
- **Self-Attention mechanism** for temporal focus
- **Classification head** with dropout regularization

### Training Configuration
- Optimizer: AdamW with weight decay
- Learning rate: 0.001 with ReduceLROnPlateau scheduler
- Early stopping: Patience of 15 epochs
- Batch size: 64

### Dataset
- **MIT-BIH Arrhythmia Database**
- Multi-class ECG classification
- Features: RR intervals, peak amplitudes, QRS morphology

## 🔬 How Perforated AI Works

### Key API Functions:

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# 1. Configure PAI settings
GPA.pc.set_output_dimensions([-1, 0])  # [batch, neurons]
GPA.pc.set_n_epochs_to_switch(10)       # Epochs before adding dendrites

# 2. Initialize PAI on your model
model = UPA.initialize_pai(model, doing_pai=True, maximizing_score=True)

# 3. Setup optimizer through PAI
GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

# 4. In training loop - report scores
GPA.pai_tracker.add_extra_score(train_acc, 'Train')

# 5. In validation - PAI decides when to add dendrites
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
if restructured:
    # Dendrites were added! Reset optimizer
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

## 🎓 Key Takeaway

> **Same architecture + Same training + Dendrites = Better performance**

Perforated AI automatically adds dendrites to `nn.Linear` and `nn.Conv` layers when training plateaus, enabling the network to continue improving beyond standard early stopping.

## 📚 References

- [Perforated AI GitHub](https://github.com/PerforatedAI/PerforatedAI)
- [Perforated AI Website](https://perforatedai.com/)
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)

## 👥 Hackathon Team

Built for the Perforated AI Hackathon demonstrating dendritic neural network improvements.
