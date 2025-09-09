# LLM Transformer - Java Spring Boot Implementation

A complete Large Language Model (LLM) Transformer implementation built with Java and Spring Boot. This project provides a fully functional transformer architecture with REST API endpoints for text generation and model inference.

## Features

- **Complete Transformer Architecture**: Multi-head attention, feed-forward networks, positional encoding
- **REST API**: Easy-to-use endpoints for text generation and model operations
- **Configurable Model**: Customizable model parameters (layers, heads, dimensions)
- **Async Processing**: Support for asynchronous text generation
- **Model Persistence**: Save and load trained models
- **Comprehensive Tokenization**: Basic tokenization with extensible design
- **Performance Metrics**: Perplexity calculation and generation statistics

## Architecture Components

### Core Components
- **TransformerModel**: Main model class orchestrating the entire architecture
- **MultiHeadAttention**: Scaled dot-product attention with multiple heads
- **FeedForwardNetwork**: Position-wise feed-forward networks
- **TransformerBlock**: Complete transformer block with residual connections
- **TokenEmbedding**: Token-to-vector embedding layer
- **PositionalEncoding**: Sinusoidal positional encodings

### Mathematical Operations
- **Matrix**: Custom matrix operations optimized for neural network computations
- Support for GELU, ReLU, Softmax, Layer Normalization

## Quick Start

### Prerequisites
- Java 17 or higher
- Maven 3.6+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MAAK-LLM-JAVA
```

2. Build the project:
```bash
mvn clean install
```

3. Run the application:
```bash
mvn spring-boot:run
```

The application will start on `http://localhost:8080`

## API Endpoints

### Text Generation
```bash
# Synchronous generation
POST /api/llm/generate
Content-Type: application/json

{
  "prompt": "The future of artificial intelligence",
  "maxLength": 100,
  "temperature": 1.0,
  "topP": 0.9
}
```

### Async Generation
```bash
POST /api/llm/generate/async
```

### Model Information
```bash
GET /api/llm/info
```

### Tokenization
```bash
POST /api/llm/tokenize
{
  "text": "Hello, world!"
}
```

### Perplexity Calculation
```bash
POST /api/llm/perplexity
{
  "text": "Sample text for perplexity calculation"
}
```

### Quick Completion
```bash
GET /api/llm/complete?prompt=The weather today&maxLength=50&temperature=1.0
```

## Configuration

Model parameters can be configured in `application.yml`:

```yaml
transformer:
  vocabulary-size: 50257
  max-sequence-length: 1024
  embedding-dimension: 768
  num-layers: 12
  num-heads: 12
  feed-forward-dimension: 3072
  dropout-rate: 0.1
  layer-norm-epsilon: 1e-5
  activation-function: gelu
```

## Model Architecture

The transformer follows the standard architecture:

1. **Input Processing**:
   - Token embedding
   - Positional encoding
   - Input dropout

2. **Transformer Blocks** (repeated N times):
   - Multi-head self-attention
   - Residual connection + Layer normalization
   - Feed-forward network
   - Residual connection + Layer normalization

3. **Output Processing**:
   - Final layer normalization
   - Linear projection to vocabulary
   - Softmax for probability distribution

## Performance Considerations

- **Memory Usage**: Model size scales with vocabulary size and embedding dimensions
- **Computation**: Attention mechanism has O(n²) complexity with sequence length
- **Parallelization**: Multi-head attention can be parallelized
- **Caching**: Consider implementing KV-cache for inference optimization

## Extending the Model

### Custom Tokenizers
Implement your own tokenizer by extending the `TokenEmbedding` class:

```java
@Component
public class CustomTokenizer extends TokenEmbedding {
    // Custom implementation
}
```

### Different Attention Mechanisms
Create custom attention by implementing new attention classes:

```java
@Component
public class CustomAttention {
    public Matrix forward(Matrix input, Matrix mask) {
        // Custom attention implementation
    }
}
```

## Model Persistence

Save and load models:

```java
// Save model
modelPersistence.saveModel("model.json", config, weights);

// Load model
ModelData modelData = modelPersistence.loadModel("model.json");
```

## Testing

Run tests with:
```bash
mvn test
```

## Monitoring

The application includes Spring Boot Actuator endpoints:
- `/actuator/health` - Health check
- `/actuator/info` - Application information
- `/actuator/metrics` - Performance metrics

## Limitations

- **Training**: This implementation focuses on inference; training capabilities would require additional components
- **Tokenization**: Uses basic word-level tokenization; production systems should use BPE or SentencePiece
- **Model Weights**: Randomly initialized; real applications need pre-trained weights
- **Optimization**: No advanced optimizations like flash attention or quantization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Based on "Attention Is All You Need" (Vaswani et al., 2017)
- Inspired by GPT and BERT architectures
- Built with Spring Boot and Java ecosystem

## Support

For questions and support, please open an issue in the repository.
