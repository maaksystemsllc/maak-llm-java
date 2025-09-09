package com.maak.llm.layers;

import com.maak.llm.math.Matrix;

/**
 * Complete Transformer Block with Multi-Head Attention and Feed Forward Network
 */
public class TransformerBlock {
    
    private final MultiHeadAttention attention;
    private final FeedForwardNetwork feedForward;
    private final double layerNormEpsilon;
    private final double dropoutRate;
    
    public TransformerBlock(int embeddingDimension, int numHeads, int feedForwardDimension, 
                           String activationFunction, double layerNormEpsilon, double dropoutRate) {
        this.attention = new MultiHeadAttention(embeddingDimension, numHeads);
        this.feedForward = new FeedForwardNetwork(embeddingDimension, feedForwardDimension, activationFunction);
        this.layerNormEpsilon = layerNormEpsilon;
        this.dropoutRate = dropoutRate;
    }
    
    /**
     * Forward pass through the transformer block
     * @param input Input tensor [sequence_length, embedding_dimension]
     * @param mask Optional attention mask for causal attention
     * @return Output tensor [sequence_length, embedding_dimension]
     */
    public Matrix forward(Matrix input, Matrix mask) {
        // Multi-Head Attention with residual connection and layer normalization
        Matrix attentionOutput = attention.forward(input, mask);
        attentionOutput = applyDropout(attentionOutput);
        Matrix afterAttention = input.add(attentionOutput); // Residual connection
        afterAttention = afterAttention.layerNorm(layerNormEpsilon); // Layer normalization
        
        // Feed Forward Network with residual connection and layer normalization
        Matrix feedForwardOutput = feedForward.forward(afterAttention);
        feedForwardOutput = applyDropout(feedForwardOutput);
        Matrix output = afterAttention.add(feedForwardOutput); // Residual connection
        output = output.layerNorm(layerNormEpsilon); // Layer normalization
        
        return output;
    }
    
    /**
     * Apply dropout (simplified implementation - in practice would be more sophisticated)
     */
    private Matrix applyDropout(Matrix input) {
        if (dropoutRate == 0.0) {
            return input;
        }
        
        // Simplified dropout - in production, this would be more sophisticated
        // and would only be applied during training
        Matrix result = new Matrix(input.getRows(), input.getCols());
        java.util.Random random = new java.util.Random();
        
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                if (random.nextDouble() > dropoutRate) {
                    result.set(i, j, input.get(i, j) / (1.0 - dropoutRate));
                } else {
                    result.set(i, j, 0.0);
                }
            }
        }
        return result;
    }
}
