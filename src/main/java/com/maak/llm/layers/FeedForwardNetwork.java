package com.maak.llm.layers;

import com.maak.llm.math.Matrix;

/**
 * Position-wise Feed Forward Network for Transformer architecture
 */
public class FeedForwardNetwork {
    
    private final Matrix firstLayerWeights;
    private final Matrix firstLayerBias;
    private final Matrix secondLayerWeights;
    private final Matrix secondLayerBias;
    private final String activationFunction;
    
    public FeedForwardNetwork(int embeddingDimension, int hiddenDimension, String activationFunction) {
        this.activationFunction = activationFunction;
        
        // Initialize weights with Xavier initialization
        double scale1 = Math.sqrt(2.0 / embeddingDimension);
        double scale2 = Math.sqrt(2.0 / hiddenDimension);
        
        this.firstLayerWeights = Matrix.random(embeddingDimension, hiddenDimension, scale1);
        this.firstLayerBias = Matrix.zeros(1, hiddenDimension);
        this.secondLayerWeights = Matrix.random(hiddenDimension, embeddingDimension, scale2);
        this.secondLayerBias = Matrix.zeros(1, embeddingDimension);
    }
    
    /**
     * Forward pass through the feed-forward network
     * @param input Input tensor [sequence_length, embedding_dimension]
     * @return Output tensor [sequence_length, embedding_dimension]
     */
    public Matrix forward(Matrix input) {
        // First linear transformation
        Matrix hidden = input.multiply(firstLayerWeights);
        hidden = addBias(hidden, firstLayerBias);
        
        // Apply activation function
        hidden = applyActivation(hidden);
        
        // Second linear transformation
        Matrix output = hidden.multiply(secondLayerWeights);
        output = addBias(output, secondLayerBias);
        
        return output;
    }
    
    /**
     * Add bias to matrix (broadcast bias across sequence dimension)
     */
    private Matrix addBias(Matrix input, Matrix bias) {
        Matrix result = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                result.set(i, j, input.get(i, j) + bias.get(0, j));
            }
        }
        return result;
    }
    
    /**
     * Apply activation function
     */
    private Matrix applyActivation(Matrix input) {
        switch (activationFunction.toLowerCase()) {
            case "gelu":
                return input.gelu();
            case "relu":
                return input.relu();
            case "swish":
                return applySwish(input);
            default:
                throw new IllegalArgumentException("Unsupported activation function: " + activationFunction);
        }
    }
    
    /**
     * Apply Swish activation function: x * sigmoid(x)
     */
    private Matrix applySwish(Matrix input) {
        Matrix result = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double x = input.get(i, j);
                double sigmoid = 1.0 / (1.0 + Math.exp(-x));
                result.set(i, j, x * sigmoid);
            }
        }
        return result;
    }
}
