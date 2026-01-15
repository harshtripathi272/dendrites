# PerforatedAI / Perforated Backpropagation Reference Guide

**Comprehensive documentation of PerforatedAI library for dendritic neural network optimization**

---

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Installation & Setup](#installation--setup)
4. [Training Modes](#training-modes)
5. [Configuration Settings](#configuration-settings)
6. [API Reference](#api-reference)
7. [Integration with PEFT/LoRA](#integration-with-peftlora)
8. [Advanced Features](#advanced-features)
9. [Best Practices](#best-practices)
10. [Examples & Use Cases](#examples--use-cases)

---

## Overview

### What is PerforatedAI?

PerforatedAI is a neural network optimization library that implements **dendritic learning** - a biologically-inspired approach where networks dynamically add "dendrites" (specialized sub-networks) during training to improve performance and data efficiency.

### Two Versions Available:

1. **Gradient Descent Dendrites (GD)** - Open source version
   - Uses standard backpropagation for dendrite training
   - Available without license
   
2. **Perforated Backpropagation (PB)** - Licensed version
   - Superior dendrite training through alternating phases
   - Requires PAIEMAIL and PAITOKEN environment variables
   - Provides better convergence and performance

### Key Benefits:

- **Data Efficiency**: Achieve similar performance with less training data
- **Adaptive Architecture**: Network grows only when needed (plateau detection)
- **Parameter Efficiency**: Smaller parameter increase compared to traditional scaling
- **No Manual Architecture Search**: System automatically finds optimal structure

---

## Core Concepts

### 1. Dendrites

**Dendrites** are specialized sub-networks added to existing layers when the model reaches a performance plateau. They act as additional pathways for information flow, allowing the network to learn more complex representations without full retraining.

**Key Properties:**
- Added dynamically during training
- Can be frozen/unfrozen independently from neurons
- Minimal parameter overhead per dendrite
- Biologically inspired (mimics biological neural dendrites)

### 2. Training Phases (Perforated Backpropagation Only)

**N Phase (Neuron Training):**
- Train neuron parameters with dendrites frozen
- Builds validation history for plateau detection
- Uses higher learning rate
- Lasts for `n_epochs_to_switch` epochs

**P Phase (Dendrite Training):**
- Train dendrite parameters with neurons frozen
- Allows dendrites to specialize without interference
- Uses lower learning rate (typically 0.5x neuron LR)
- Lasts for `p_epochs_to_switch` epochs

**Phase Alternation:**
```
N Phase (6 epochs) → Plateau Detected → Add Dendrite → P Phase (6 epochs) 
→ Switch back to N Phase → Continue...
```

### 3. Restructuring

**Restructuring** occurs when PAI adds a new dendrite to the model:
- Triggered by plateau detection or trial completion
- Model architecture changes (new parameters added)
- **Optimizer must be reset** after restructuring
- Learning rate may be adjusted for new phase

### 4. Plateau Detection (DOING_HISTORY Mode)

PAI monitors validation scores over a history window to detect when the model stops improving:
- Tracks last N validation scores
- Compares against improvement threshold
- Adds dendrite when plateau confirmed
- More adaptive than fixed-epoch switching

---

## Installation & Setup

### Installation

```bash
# Install PerforatedAI
pip install perforatedai

# For Perforated Backpropagation (requires license)
pip install perforatedbp
```

### License Setup (for PB)

Create `.env` file in project root:

```bash
PAIEMAIL=your_email@example.com
PAITOKEN=your_license_token_here
```

Load in code:

```python
from dotenv import load_dotenv
load_dotenv()  # Must be called BEFORE importing perforatedai

import os
if os.getenv('PAIEMAIL') and os.getenv('PAITOKEN'):
    print("✓ PAI License loaded")
else:
    print("⚠ License not found - using GD mode")
```

### Basic Imports

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Check PB availability
pb_available = GPA.pc.get_perforated_backpropagation()
if pb_available:
    print("🔥 Perforated Backpropagation enabled!")
else:
    print("📊 Using Gradient Descent Dendrites")
```

---

## Training Modes

### 1. DOING_HISTORY (Recommended)

**Adaptive plateau detection** - Adds dendrites when model stops improving.

```python
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(6)  # Wait 6 epochs before checking plateau
GPA.pc.set_p_epochs_to_switch(6)  # Dendrite training phase duration (PB only)
GPA.pc.set_improvement_threshold(0.0)  # Add dendrite if ANY improvement
```

**When to Use:**
- Most experiments (default recommendation)
- When you want adaptive architecture growth
- For fair comparisons (dendrites added only when needed)

**How it Works:**
1. Train for N epochs in N phase
2. Monitor validation scores
3. If plateau detected → add dendrite
4. Switch to P phase (PB only) or continue
5. Repeat until no more improvement

### 2. DOING_TRIALS

**Fixed-epoch dendrite addition** - Adds dendrites at predetermined intervals.

```python
GPA.pc.set_switch_mode(GPA.pc.DOING_TRIALS)
GPA.pc.set_n_epochs_to_switch(10)  # Add dendrite every 10 epochs
GPA.pc.set_max_dendrites(5)  # Stop after 5 dendrites
```

**When to Use:**
- When you want predictable training time
- For ablation studies (controlled dendrite count)
- When plateau detection is unreliable (noisy data)

**How it Works:**
1. Train for N epochs
2. Add dendrite (regardless of performance)
3. Continue for another N epochs
4. Repeat until max_dendrites reached

### 3. Testing Dendrite Capacity

**Quick probe mode** - Tests how many dendrites the model needs.

```python
GPA.pc.set_testing_dendrite_capacity(True)
GPA.pc.set_max_dendrites(10)  # Test up to 10 dendrites
```

**When to Use:**
- Initial exploration of dataset/model
- To determine optimal `max_dendrites` for full runs
- Fast iteration (typically ~7 epochs per dendrite)

**How it Works:**
- Runs shortened training cycles
- Quickly adds dendrites to probe capacity
- Reports diminishing returns point
- Use findings to set max_dendrites for real training

---

## Configuration Settings

### Core PAI Settings

#### Switch Mode

```python
# Adaptive (recommended)
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)

# Fixed intervals
GPA.pc.set_switch_mode(GPA.pc.DOING_TRIALS)

# Auto (PAI decides)
GPA.pc.set_switch_mode(GPA.pc.AUTO)
```

#### Epoch Control

```python
# N phase duration (neuron training)
GPA.pc.set_n_epochs_to_switch(6)  # Common: 5-10 epochs

# P phase duration (dendrite training - PB only)
GPA.pc.set_p_epochs_to_switch(6)  # Common: same as N phase

# Grace period after restructuring
GPA.pc.set_initial_history_after_switches(2)  # Wait 2 epochs before next dendrite

# Cap subsequent rounds
GPA.pc.set_cap_at_n(True)  # Later rounds don't exceed first round length
```

#### Dendrite Control

```python
# Maximum dendrites to add
GPA.pc.set_max_dendrites(99)  # High value = let improvement_threshold decide
                               # Low value (1-2) = conservative growth

# Improvement threshold (relative improvement required)
GPA.pc.set_improvement_threshold(0.0)   # ANY improvement → add dendrite
GPA.pc.set_improvement_threshold(0.01)  # Need 1% improvement → add dendrite
GPA.pc.set_improvement_threshold(0.05)  # Need 5% improvement → add dendrite

# Raw threshold (absolute score difference)
GPA.pc.set_improvement_threshold_raw(0.001)  # Rarely used

# Testing mode
GPA.pc.set_testing_dendrite_capacity(False)  # False = normal training
                                              # True = fast probe mode
```

**Guidance from Rorry Brenner (PAI creator):**
> "set_improvement_threshold(0) = keep adding dendrites if ANY improvement"  
> "set_improvement_threshold(0.01) = needs 1% relative improvement to add dendrite"  
> "By NOT setting max_dendrites (or setting high), PAI will keep adding as long as threshold is met"

### Perforated Backpropagation Settings (PB Only)

```python
# Enable gradient flow to dendrites (recommended for PEFT/LoRA)
GPA.pc.set_dendrite_update_mode(True)  # True = dendrites receive gradients
                                        # False = dendrites frozen during N phase

# Dendrite connectivity pattern
GPA.pc.set_dendrite_graph_mode(False)  # False = standard (recommended)
                                        # True = alternative connectivity (experimental)

# Initial correlation batches for PB
GPA.pc.set_initial_correlation_batches(40)  # Batches for initial PB setup
```

### PEFT/LoRA Specific Settings

```python
from peft import tuners

# Specify which layers to convert to dendrites
GPA.pc.set_modules_to_convert([tuners.lora.layer.Linear])  # LoRA Linear layers

# Confirm no wrapper modules
GPA.pc.set_unwrapped_modules_confirmed(True)

# Allow weight decay in optimizer
GPA.pc.set_weight_decay_accepted(True)

# Don't use safe tensors
GPA.pc.set_using_safe_tensors(False)

# Skip unused layers (frozen base model)
GPA.pc.set_checked_skipped_modules(True)

# Disable test saves (prevents PEFT serialization issues)
GPA.pc.set_test_saves(False)
GPA.pc.set_pai_saves(False)

# Output dimensions for LLMs: [batch, seq, hidden]
GPA.pc.set_output_dimensions([-1, -1, 0])

# Don't save base model
GPA.pc.set_module_names_to_not_save(['.base_model.model.base_model'])
```

### Scoring Configuration

```python
# Set save name for outputs
GPA.pc.set_save_name("MyExperiment_100pct")

# Enable graph generation
GPA.pc.set_making_graphs(True)

# Maximizing vs minimizing
# Use in initialize_pai:
model = UPA.initialize_pai(
    model,
    maximizing_score=True   # True = higher is better (accuracy)
                            # False = lower is better (loss)
)
```

---

## API Reference

### Initialization

```python
from perforatedai import utils_perforatedai as UPA

# Initialize PAI on model
model = UPA.initialize_pai(
    model,
    maximizing_score=True  # True if using accuracy, False if using loss
)
```

### Training Loop Integration

#### Adding Validation Scores

```python
# During training loop (each epoch)
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
    validation_score,  # Your validation metric (accuracy or loss)
    model
)

# Check flags
if restructured:
    print("Dendrite added! Reset optimizer.")
    # MUST reset optimizer after restructuring
    optimizer = create_optimizer(model)
    GPA.pai_tracker.set_optimizer_instance(optimizer)

if training_complete:
    print("PAI: Training complete!")
    break
```

#### Adding Extra Scores (for logging)

```python
# Log additional metrics (won't trigger restructuring)
GPA.pai_tracker.add_extra_score(train_loss, 'Train Loss')
GPA.pai_tracker.add_extra_score(train_acc, 'Train Accuracy')
GPA.pai_tracker.add_extra_score(learning_rate, 'Learning Rate')
```

#### Optimizer Management

```python
# Set optimizer instance for PAI tracking
GPA.pai_tracker.set_optimizer_instance(optimizer)

# After restructuring, create new optimizer
optimizer = AdamW(model.parameters(), lr=new_lr)
GPA.pai_tracker.set_optimizer_instance(optimizer)
```

### Perforated Backpropagation Gradient Management (PB Only)

```python
# In training loop, after loss.backward()
if pb_available:
    try:
        # Apply PB-specific gradients to dendrites
        GPA.pai_tracker.apply_pb_grads()
    except:
        pass  # May fail if not in P mode

# After optimizer.step()
if pb_available:
    try:
        # Clear PB gradient state
        GPA.pai_tracker.apply_pb_zero()
    except:
        pass  # May fail if not in P mode
```

### Checking Current State

```python
# Check current mode (N or P phase)
current_mode = GPA.pai_tracker.member_vars.get('mode', 'n')
if current_mode == 'p':
    print("In P phase (dendrite training)")
else:
    print("In N phase (neuron training)")

# Check PB availability
pb_available = GPA.pc.get_perforated_backpropagation()

# Check if restructuring occurred
if GPA.pai_tracker.member_vars.get('restructured', False):
    print("Model was restructured this epoch")
```

---

## Integration with PEFT/LoRA

### Why LoRA + PAI?

Combining LoRA with PAI provides:
- **Double efficiency**: LoRA reduces base parameters, PAI adds targeted capacity
- **Data efficiency**: Both techniques improve sample efficiency
- **Flexibility**: Dendrites adapt LoRA adapters dynamically

### Setup Steps

#### 1. Create LoRA Model

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("model_name")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    inference_mode=False
)

model = get_peft_model(model, lora_config)
```

#### 2. Configure PAI for PEFT

```python
from perforatedai import globals_perforatedai as GPA
from peft import tuners

# CRITICAL: Tell PAI to only convert LoRA layers
GPA.pc.set_modules_to_convert([tuners.lora.layer.Linear])
GPA.pc.set_unwrapped_modules_confirmed(True)
GPA.pc.set_weight_decay_accepted(True)
GPA.pc.set_using_safe_tensors(False)

# Handle frozen base model
GPA.pc.set_checked_skipped_modules(True)

# Disable problematic saves
GPA.pc.set_test_saves(False)
GPA.pc.set_pai_saves(False)

# LLM-specific settings
GPA.pc.set_output_dimensions([-1, -1, 0])  # [batch, seq, hidden]
GPA.pc.set_module_names_to_not_save(['.base_model.model.base_model'])
```

#### 3. Initialize PAI

```python
from perforatedai import utils_perforatedai as UPA

model = UPA.initialize_pai(
    model,
    maximizing_score=True  # True for accuracy metrics
)
```

#### 4. Train with Optimizer Reset

```python
for epoch in range(max_epochs):
    # Training loop
    train_one_epoch(model, optimizer)
    
    # Validation
    val_score = validate(model)
    
    # PAI scoring
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_score,
        model
    )
    
    # CRITICAL: Reset optimizer after restructuring
    if restructured:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
    
    if training_complete:
        break
```

---

## Advanced Features

### 1. Multi-Objective Optimization

Track multiple metrics simultaneously:

```python
# Primary metric (triggers restructuring)
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
    val_accuracy,
    model
)

# Secondary metrics (logging only)
GPA.pai_tracker.add_extra_score(val_loss, 'Val Loss')
GPA.pai_tracker.add_extra_score(val_f1, 'Val F1')
GPA.pai_tracker.add_extra_score(val_perplexity, 'Val Perplexity')
```

### 2. Custom Learning Rate Schedules

Adjust LR based on phase:

```python
if restructured:
    # Get current phase
    current_mode = GPA.pai_tracker.member_vars.get('mode', 'n')
    
    if current_mode == 'n':
        # N phase: Higher LR for neurons
        lr = base_lr
    else:
        # P phase: Lower LR for dendrites
        lr = base_lr * 0.5
    
    optimizer = AdamW(model.parameters(), lr=lr)
    GPA.pai_tracker.set_optimizer_instance(optimizer)
```

### 3. Dendrite Capacity Testing

Quick probe to find optimal dendrite count:

```python
# Phase 1: Quick test
GPA.pc.set_testing_dendrite_capacity(True)
GPA.pc.set_max_dendrites(10)
# ... train briefly ...

# Phase 2: Full training with optimal count
GPA.pc.set_testing_dendrite_capacity(False)
GPA.pc.set_max_dendrites(optimal_count)  # From phase 1 results
# ... train fully ...
```

### 4. Conditional Dendrite Addition

Add logic to control when dendrites are added:

```python
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
    val_score,
    model
)

if restructured:
    # Check if we should actually add the dendrite
    if current_epoch < min_epoch_for_dendrites:
        print("Too early for dendrites, skipping...")
        # Could potentially revert here (advanced)
    else:
        print("Accepting dendrite addition")
        optimizer = AdamW(model.parameters(), lr=lr)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
```

### 5. Hybrid Training Strategies

Combine multiple approaches:

```python
# Start with trials mode for stable base
GPA.pc.set_switch_mode(GPA.pc.DOING_TRIALS)
GPA.pc.set_n_epochs_to_switch(10)
# Train for 30 epochs...

# Switch to history mode for fine-tuning
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_improvement_threshold(0.01)
# Continue training...
```

### 6. Mixed Precision Training with PAI

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = compute_loss(model, batch)
    
    scaler.scale(loss).backward()
    
    # PB gradient application (if available)
    if pb_available:
        GPA.pai_tracker.apply_pb_grads()
    
    scaler.step(optimizer)
    scaler.update()
    
    if pb_available:
        GPA.pai_tracker.apply_pb_zero()
```

---

## Best Practices

### 1. Optimizer Management

**CRITICAL:** Always reset optimizer after restructuring:

```python
if restructured:
    # Create fresh optimizer with new parameters
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Update PAI tracker
    GPA.pai_tracker.set_optimizer_instance(optimizer)
    
    # Also recreate scheduler if using one
    scheduler = get_scheduler(optimizer, ...)
```

**Why?** Restructuring adds new parameters. Old optimizer doesn't know about them.

### 2. Choosing Switch Mode

```python
# For most experiments: DOING_HISTORY
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_improvement_threshold(0.0)  # Liberal growth

# For controlled studies: DOING_TRIALS
GPA.pc.set_switch_mode(GPA.pc.DOING_TRIALS)
GPA.pc.set_max_dendrites(3)  # Fixed count

# For exploration: Testing mode first
GPA.pc.set_testing_dendrite_capacity(True)
# ... then switch to DOING_HISTORY with findings
```

### 3. Improvement Threshold Tuning

```python
# Very liberal (any improvement)
GPA.pc.set_improvement_threshold(0.0)
# → Pro: Explores fully, may find optimal architecture
# → Con: May add too many dendrites, slower training

# Conservative (1% improvement required)
GPA.pc.set_improvement_threshold(0.01)
# → Pro: Faster training, fewer dendrites
# → Con: May miss subtle improvements

# Balanced (0.5% improvement)
GPA.pc.set_improvement_threshold(0.005)
# → Good middle ground for most tasks
```

### 4. Max Dendrites Strategy

From PAI creator (Rorry Brenner):

```python
# Approach 1: Let improvement_threshold decide (recommended)
GPA.pc.set_max_dendrites(99)  # No practical limit
GPA.pc.set_improvement_threshold(0.01)  # Stop when gains < 1%

# Approach 2: Conservative cap (faster iteration)
GPA.pc.set_max_dendrites(2)  # Only 1-2 dendrites
GPA.pc.set_improvement_threshold(0.0)  # Add if any improvement
```

> "Often 1-2 dendrites are sufficient. By setting max_dendrites low, you can iterate faster while still getting dendrite benefits."

### 5. Learning Rate Recommendations

```python
# Base LR (from official PAI PEFT example)
base_lr = 3e-4  # Good for most LLM fine-tuning

# Neuron phase LR
neuron_lr = base_lr

# Dendrite phase LR (PB only)
dendrite_lr = base_lr * 0.5  # Lower for dendrite specialization

# After restructuring in N phase
post_restructure_lr = base_lr * 0.5  # Lower after adding dendrite
```

### 6. Validation Frequency

```python
# Validate every epoch (recommended for DOING_HISTORY)
val_score = validate_every_epoch(model)

# Don't validate too frequently (wastes time)
# Don't validate too rarely (miss plateau detection)

# For large datasets, can validate on subset
val_score = validate_on_subset(model, subset_size=1000)
```

### 7. Early Stopping with PAI

Combine PAI's `training_complete` with your own early stopping:

```python
patience = 10
patience_counter = 0
best_score = 0.0

for epoch in range(max_epochs):
    # ... training ...
    
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_score,
        model
    )
    
    # PAI says we're done
    if training_complete:
        break
    
    # Your early stopping
    if val_score > best_score + min_delta:
        best_score = val_score
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**Note:** Make sure `patience > n_epochs_to_switch` to avoid conflict!

### 8. Logging Best Practices

```python
import wandb

wandb.log({
    # Basic metrics
    "epoch": epoch,
    "train/loss": train_loss,
    "val/accuracy": val_acc,
    
    # PAI-specific
    "pai/mode": 1 if current_mode == 'p' else 0,
    "pai/num_dendrites": num_dendrites_added,
    "pai/restructured": int(restructured),
    "pai/current_params": sum(p.numel() for p in model.parameters()),
    
    # PB status
    "pb/enabled": int(pb_available),
    
    # Efficiency metrics
    "efficiency/params_per_dendrite": params_increase / num_dendrites,
    "efficiency/acc_per_epoch": val_acc / epoch
})
```

---

## Examples & Use Cases

### Example 1: Basic Image Classification

```python
import torch
from torch.optim import AdamW
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Setup
model = create_resnet_model()
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(5)
GPA.pc.set_improvement_threshold(0.01)
model = UPA.initialize_pai(model, maximizing_score=True)

optimizer = AdamW(model.parameters(), lr=1e-3)
GPA.pai_tracker.set_optimizer_instance(optimizer)

# Training
for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_acc = evaluate(model, val_loader)
    
    model, restructured, complete = GPA.pai_tracker.add_validation_score(
        val_acc, model
    )
    
    if restructured:
        optimizer = AdamW(model.parameters(), lr=1e-3)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
    
    if complete:
        break
```

### Example 2: LLM Fine-tuning with LoRA

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, tuners
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    inference_mode=False
)
model = get_peft_model(model, lora_config)

# Configure PAI for PEFT
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(6)
GPA.pc.set_improvement_threshold(0.0)
GPA.pc.set_modules_to_convert([tuners.lora.layer.Linear])
GPA.pc.set_unwrapped_modules_confirmed(True)
GPA.pc.set_test_saves(False)
GPA.pc.set_output_dimensions([-1, -1, 0])

# Initialize
model = UPA.initialize_pai(model, maximizing_score=True)
optimizer = AdamW(model.parameters(), lr=3e-4)
GPA.pai_tracker.set_optimizer_instance(optimizer)

# Train
for epoch in range(1000):
    train(model, train_loader, optimizer)
    val_acc = evaluate(model, val_loader)
    
    model, restructured, complete = GPA.pai_tracker.add_validation_score(
        val_acc, model
    )
    
    if restructured:
        optimizer = AdamW(model.parameters(), lr=3e-4)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
    
    if complete:
        break
```

### Example 3: Data Efficiency Experiment

```python
def run_data_efficiency_study(data_percentages=[1.0, 0.75, 0.5, 0.25]):
    """Compare baseline vs dendritic across data percentages."""
    
    results = {}
    
    for pct in data_percentages:
        print(f"\n=== Training with {pct*100}% data ===")
        
        # Baseline (no PAI)
        model_baseline = create_model()
        train_loader = get_data_loader(percentage=pct)
        baseline_acc = train_baseline(model_baseline, train_loader)
        
        # Dendritic (with PAI)
        model_dendritic = create_model()
        GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
        GPA.pc.set_improvement_threshold(0.0)
        model_dendritic = UPA.initialize_pai(model_dendritic, maximizing_score=True)
        dendritic_acc = train_dendritic(model_dendritic, train_loader)
        
        results[pct] = {
            'baseline': baseline_acc,
            'dendritic': dendritic_acc,
            'improvement': dendritic_acc - baseline_acc
        }
    
    return results
```

### Example 4: Custom Dendrite Control

```python
class AdaptiveDendriteController:
    """Custom logic for dendrite addition."""
    
    def __init__(self, model):
        self.model = model
        self.dendrite_history = []
        
    def should_add_dendrite(self, val_score, epoch):
        """Custom logic for dendrite addition."""
        
        # Don't add dendrites too early
        if epoch < 10:
            return False
        
        # Don't add if score decreasing
        if len(self.dendrite_history) > 0:
            if val_score < self.dendrite_history[-1]:
                return False
        
        # Check if improvement is significant
        if len(self.dendrite_history) >= 5:
            recent_avg = sum(self.dendrite_history[-5:]) / 5
            if val_score - recent_avg < 0.01:  # Less than 1% improvement
                return False
        
        return True
    
    def train_with_control(self, train_loader, val_loader):
        """Training loop with custom dendrite control."""
        
        for epoch in range(max_epochs):
            train_epoch(self.model, train_loader)
            val_score = evaluate(self.model, val_loader)
            
            # Let PAI do its scoring
            self.model, restructured, complete = GPA.pai_tracker.add_validation_score(
                val_score, self.model
            )
            
            # Our custom check
            if restructured and not self.should_add_dendrite(val_score, epoch):
                print("Custom controller: Rejecting dendrite")
                # Would need PAI API to revert (advanced)
            
            self.dendrite_history.append(val_score)
            
            if complete:
                break
```

---

## Troubleshooting

### Common Issues

#### 1. "Optimizer doesn't have new parameters"

**Cause:** Forgot to reset optimizer after restructuring.

**Fix:**
```python
if restructured:
    optimizer = AdamW(model.parameters(), lr=lr)
    GPA.pai_tracker.set_optimizer_instance(optimizer)
```

#### 2. "PB features not working despite license"

**Cause:** License not loaded before importing PAI.

**Fix:**
```python
# MUST be FIRST
from dotenv import load_dotenv
load_dotenv()

# THEN import PAI
from perforatedai import globals_perforatedai as GPA
```

#### 3. "PEFT model serialization errors"

**Cause:** PAI trying to save full model with PEFT wrappers.

**Fix:**
```python
GPA.pc.set_test_saves(False)
GPA.pc.set_pai_saves(False)
GPA.pc.set_module_names_to_not_save(['.base_model.model.base_model'])
```

#### 4. "Dendrites never added"

**Cause:** `improvement_threshold` too high or `n_epochs_to_switch` too short.

**Fix:**
```python
GPA.pc.set_improvement_threshold(0.0)  # Lower threshold
GPA.pc.set_n_epochs_to_switch(6)  # More epochs to build history
```

#### 5. "Too many dendrites added"

**Cause:** `improvement_threshold` too low or `max_dendrites` too high.

**Fix:**
```python
GPA.pc.set_improvement_threshold(0.01)  # Require 1% improvement
GPA.pc.set_max_dendrites(3)  # Cap at 3 dendrites
```

---

## Additional Resources

### Official Examples

- **PEFT LoRA Example**: https://github.com/perforatedai/perforatedai-examples/tree/main/libraryExamples/huggingface/PEFT
- **Vision Examples**: https://github.com/perforatedai/perforatedai-examples/tree/main/libraryExamples/vision
- **NLP Examples**: https://github.com/perforatedai/perforatedai-examples/tree/main/libraryExamples/nlp

### Discord Community

Join the PerforatedAI Discord for:
- Direct support from Rorry Brenner (creator)
- Community examples and tips
- Latest feature announcements
- Troubleshooting help

### Key Papers & Concepts

- **Dendritic Learning**: Biologically-inspired adaptive architecture
- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA, Adapters, etc.
- **Neural Architecture Search (NAS)**: Automated architecture optimization

---

## Version Notes

This guide covers PerforatedAI as of January 2025. API may evolve.

**Check for updates:**
```python
import perforatedai
print(perforatedai.__version__)
```

**Key Version Differences:**
- **v1.x**: GD Dendrites only
- **v2.x**: Perforated Backpropagation introduced
- **Latest**: Enhanced PEFT integration, improved PB algorithms

---

## Summary Cheatsheet

### Quick Start (5 lines)

```python
from perforatedai import globals_perforatedai as GPA, utils_perforatedai as UPA
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_improvement_threshold(0.0)
model = UPA.initialize_pai(model, maximizing_score=True)
GPA.pai_tracker.set_optimizer_instance(optimizer)
```

### Training Loop Template

```python
for epoch in range(max_epochs):
    train_epoch(model, train_loader, optimizer)
    val_score = evaluate(model, val_loader)
    
    model, restructured, complete = GPA.pai_tracker.add_validation_score(val_score, model)
    
    if restructured:
        optimizer = create_optimizer(model)
        GPA.pai_tracker.set_optimizer_instance(optimizer)
    
    if complete:
        break
```

### Common Configurations

```python
# Liberal (explore fully)
GPA.pc.set_improvement_threshold(0.0)
GPA.pc.set_max_dendrites(99)

# Conservative (fast iteration)
GPA.pc.set_improvement_threshold(0.01)
GPA.pc.set_max_dendrites(2)

# Balanced (recommended starting point)
GPA.pc.set_improvement_threshold(0.005)
GPA.pc.set_max_dendrites(5)
```

---

**Last Updated:** January 15, 2026  
**Author:** Research documentation for dendritic optimization projects  
**License:** Educational/Research Use
