# Model Training Guide

This guide explains how to train your LLM Transformer implementation with custom data.

## Overview

The training system includes:
- **DataLoader**: Processes text files and creates training batches
- **LossFunction**: Calculates cross-entropy loss for language modeling
- **Optimizer**: Adam optimizer with gradient clipping and learning rate scheduling
- **ModelTrainer**: Main training orchestrator
- **TrainingController**: REST API endpoints for training management

## Quick Start

### 1. Prepare Training Data

Create text files in the `training_data/` directory:

```bash
# Create training data directory
mkdir training_data

# Add your text files (one paragraph per line, empty lines separate documents)
echo "Your training text here..." > training_data/my_data.txt
```

**Data Format:**
- Plain text files (.txt)
- One paragraph per line
- Empty lines separate documents
- UTF-8 encoding

### 2. Configure Training Parameters

Update `src/main/resources/application.yml`:

```yaml
training:
  learning-rate: 0.0001      # Learning rate (start with 1e-4)
  batch-size: 32             # Batch size (adjust based on memory)
  epochs: 10                 # Number of training epochs
  max-sequence-length: 512   # Maximum tokens per sequence
  data-path: "training_data" # Path to training data
  checkpoint-path: "checkpoints" # Where to save model checkpoints
```

### 3. Start Training

#### Option A: Using REST API

```bash
# Start the application
mvn spring-boot:run

# Start training
curl -X POST http://localhost:8083/api/training/start

# Check training status
curl http://localhost:8083/api/training/status

# Stop training (if needed)
curl -X POST http://localhost:8083/api/training/stop
```

#### Option B: Using Swagger UI

1. Open http://localhost:8083/swagger-ui.html
2. Navigate to "Training" section
3. Use `/api/training/start` endpoint
4. Monitor progress with `/api/training/status`

## Training Process

### Data Loading
1. Scans `training_data/` for .txt files
2. Splits text into paragraphs
3. Tokenizes using the model's vocabulary
4. Creates batches for training

### Training Loop
1. **Forward Pass**: Model processes input sequences
2. **Loss Calculation**: Cross-entropy loss between predictions and targets
3. **Backward Pass**: Compute gradients (simplified implementation)
4. **Parameter Update**: Adam optimizer updates model weights
5. **Checkpointing**: Saves model state periodically

### Monitoring
- Loss values logged every 100 steps
- Perplexity calculated every 500 steps
- Checkpoints saved every 1000 steps
- Training metrics available via API

## API Endpoints

### Start Training
```bash
POST /api/training/start
```
Response:
```json
{
  "status": "started",
  "message": "Training started successfully",
  "config": { ... }
}
```

### Check Status
```bash
GET /api/training/status
```
Response:
```json
{
  "status": "running|completed|failed|not_started",
  "message": "...",
  "total_steps": 1000,
  "final_losses": [2.5, 2.3, 2.1],
  "perplexities": [12.5, 10.2, 8.7]
}
```

### Update Configuration
```bash
POST /api/training/config?learningRate=0.0002&batchSize=16&epochs=5
```

### Get Configuration
```bash
GET /api/training/config
```

## Training Configuration

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `learning-rate` | Learning rate for optimization | 0.0001 | 1e-4 to 1e-5 |
| `batch-size` | Number of sequences per batch | 32 | 16-64 |
| `epochs` | Number of training epochs | 10 | 5-20 |
| `max-sequence-length` | Maximum tokens per sequence | 512 | 256-1024 |
| `weight-decay` | L2 regularization | 0.01 | 0.01-0.1 |
| `gradient-clipping` | Max gradient norm | 1.0 | 0.5-2.0 |
| `warmup-steps` | Learning rate warmup | 4000 | 1000-10000 |

## Data Requirements

### Minimum Dataset Size
- **Small experiments**: 1MB+ of text
- **Decent performance**: 10MB+ of text  
- **Good results**: 100MB+ of text

### Data Quality
- Clean, well-formatted text
- Consistent language and style
- Remove special characters if needed
- Ensure proper encoding (UTF-8)

### Example Data Structure
```
training_data/
├── book1.txt
├── articles.txt
├── conversations.txt
└── technical_docs.txt
```

## Monitoring Training

### Loss Metrics
- **Cross-entropy loss**: Should decrease over time
- **Perplexity**: Lower is better (good models: 10-50)
- **Training steps**: Total optimization steps completed

### Checkpoints
Saved in `checkpoints/` directory:
```
checkpoints/
├── checkpoint_epoch_1_step_1000_20241209_143022.json
├── checkpoint_epoch_2_step_2000_20241209_144515.json
└── ...
```

### Log Output
```
2024-12-09 14:30:22 - Starting model training with config: lr=0.0001, batch_size=32, epochs=10
2024-12-09 14:30:25 - Loaded 150 texts from training_data/sample_text.txt
2024-12-09 14:30:25 - Created 5 training batches from 150 texts
2024-12-09 14:30:25 - Starting epoch 1/10
2024-12-09 14:30:30 - Step 100: Loss = 3.2456
2024-12-09 14:30:35 - Step 200: Loss = 2.8934
2024-12-09 14:30:40 - Step 500: Perplexity = 15.67
```

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce `batch-size` (try 16 or 8)
- Reduce `max-sequence-length` (try 256)
- Use gradient accumulation

**Training Not Starting**
- Check data path exists: `training_data/`
- Ensure .txt files are present
- Verify file permissions

**High Loss/Poor Convergence**
- Lower learning rate (try 5e-5)
- Increase warmup steps
- Check data quality
- Reduce batch size

**Training Too Slow**
- Increase batch size (if memory allows)
- Use fewer epochs initially
- Reduce sequence length

### Performance Tips

1. **Start Small**: Begin with a small dataset and few epochs
2. **Monitor Closely**: Watch loss and perplexity trends
3. **Adjust Gradually**: Make small parameter changes
4. **Save Frequently**: Use regular checkpointing
5. **Validate Results**: Test generation quality during training

## Next Steps

After training:
1. **Evaluate**: Test text generation quality
2. **Fine-tune**: Adjust hyperparameters based on results  
3. **Scale Up**: Use larger datasets for better performance
4. **Deploy**: Use the trained model for inference

## Example Training Session

```bash
# 1. Prepare data
mkdir training_data
echo "Your training corpus here..." > training_data/corpus.txt

# 2. Start application
mvn spring-boot:run

# 3. Configure training (optional)
curl -X POST "http://localhost:8083/api/training/config?learningRate=0.0001&batchSize=16&epochs=5"

# 4. Start training
curl -X POST http://localhost:8083/api/training/start

# 5. Monitor progress
curl http://localhost:8083/api/training/status

# 6. Test the trained model
curl -X POST http://localhost:8083/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI", "maxLength": 100}'
```

The training system provides a complete framework for training your transformer model on custom data with proper monitoring and checkpointing capabilities.
