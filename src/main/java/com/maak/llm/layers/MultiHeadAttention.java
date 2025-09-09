package com.maak.llm.layers;

import com.maak.llm.math.Matrix;

/**
 * Multi-Head Attention mechanism for Transformer architecture
 */
public class MultiHeadAttention {
    
    private final Matrix queryWeights;
    private final Matrix keyWeights;
    private final Matrix valueWeights;
    private final Matrix outputWeights;
    private final int numHeads;
    private final int headDimension;
    private final int embeddingDimension;
    
    public MultiHeadAttention(int embeddingDimension, int numHeads) {
        this.embeddingDimension = embeddingDimension;
        this.numHeads = numHeads;
        this.headDimension = embeddingDimension / numHeads;
        
        if (embeddingDimension % numHeads != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by number of heads");
        }
        
        // Initialize weight matrices with Xavier initialization
        double scale = Math.sqrt(2.0 / embeddingDimension);
        this.queryWeights = Matrix.random(embeddingDimension, embeddingDimension, scale);
        this.keyWeights = Matrix.random(embeddingDimension, embeddingDimension, scale);
        this.valueWeights = Matrix.random(embeddingDimension, embeddingDimension, scale);
        this.outputWeights = Matrix.random(embeddingDimension, embeddingDimension, scale);
    }
    
    /**
     * Forward pass of multi-head attention
     * @param input Input tensor [sequence_length, embedding_dimension]
     * @param mask Optional attention mask
     * @return Attention output [sequence_length, embedding_dimension]
     */
    public Matrix forward(Matrix input, Matrix mask) {
        int sequenceLength = input.getRows();
        
        // Linear transformations for Q, K, V
        Matrix queries = input.multiply(queryWeights);
        Matrix keys = input.multiply(keyWeights);
        Matrix values = input.multiply(valueWeights);
        
        // Reshape for multi-head attention
        Matrix[] queryHeads = splitHeads(queries);
        Matrix[] keyHeads = splitHeads(keys);
        Matrix[] valueHeads = splitHeads(values);
        
        // Apply attention for each head
        Matrix[] attentionHeads = new Matrix[numHeads];
        for (int i = 0; i < numHeads; i++) {
            attentionHeads[i] = scaledDotProductAttention(queryHeads[i], keyHeads[i], valueHeads[i], mask);
        }
        
        // Concatenate heads
        Matrix concatenated = concatenateHeads(attentionHeads);
        
        // Final linear transformation
        return concatenated.multiply(outputWeights);
    }
    
    /**
     * Scaled dot-product attention
     */
    private Matrix scaledDotProductAttention(Matrix query, Matrix key, Matrix value, Matrix mask) {
        // Compute attention scores
        Matrix scores = query.multiply(key.transpose());
        
        // Scale by sqrt(head_dimension)
        double scale = 1.0 / Math.sqrt(headDimension);
        scores = scores.scale(scale);
        
        // Apply mask if provided
        if (mask != null) {
            scores = applyMask(scores, mask);
        }
        
        // Apply softmax
        Matrix attentionWeights = scores.softmax();
        
        // Apply attention to values
        return attentionWeights.multiply(value);
    }
    
    /**
     * Split input into multiple heads
     */
    private Matrix[] splitHeads(Matrix input) {
        int sequenceLength = input.getRows();
        Matrix[] heads = new Matrix[numHeads];
        
        for (int h = 0; h < numHeads; h++) {
            heads[h] = new Matrix(sequenceLength, headDimension);
            for (int i = 0; i < sequenceLength; i++) {
                for (int j = 0; j < headDimension; j++) {
                    int originalCol = h * headDimension + j;
                    heads[h].set(i, j, input.get(i, originalCol));
                }
            }
        }
        return heads;
    }
    
    /**
     * Concatenate multiple heads back together
     */
    private Matrix concatenateHeads(Matrix[] heads) {
        int sequenceLength = heads[0].getRows();
        Matrix result = new Matrix(sequenceLength, embeddingDimension);
        
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < sequenceLength; i++) {
                for (int j = 0; j < headDimension; j++) {
                    int resultCol = h * headDimension + j;
                    result.set(i, resultCol, heads[h].get(i, j));
                }
            }
        }
        return result;
    }
    
    /**
     * Apply attention mask (for causal attention in language modeling)
     */
    private Matrix applyMask(Matrix scores, Matrix mask) {
        Matrix maskedScores = new Matrix(scores.getRows(), scores.getCols());
        for (int i = 0; i < scores.getRows(); i++) {
            for (int j = 0; j < scores.getCols(); j++) {
                if (mask.get(i, j) == 0) {
                    maskedScores.set(i, j, Double.NEGATIVE_INFINITY);
                } else {
                    maskedScores.set(i, j, scores.get(i, j));
                }
            }
        }
        return maskedScores;
    }
    
    /**
     * Create causal mask for autoregressive generation
     */
    public static Matrix createCausalMask(int sequenceLength) {
        Matrix mask = new Matrix(sequenceLength, sequenceLength);
        for (int i = 0; i < sequenceLength; i++) {
            for (int j = 0; j <= i; j++) {
                mask.set(i, j, 1.0);
            }
        }
        return mask;
    }
}
