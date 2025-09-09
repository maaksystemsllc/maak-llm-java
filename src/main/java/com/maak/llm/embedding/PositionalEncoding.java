package com.maak.llm.embedding;

import com.maak.llm.math.Matrix;

/**
 * Positional Encoding for Transformer architecture
 */
public class PositionalEncoding {
    
    private final Matrix positionEncodings;
    private final int maxSequenceLength;
    private final int embeddingDimension;
    
    public PositionalEncoding(int maxSequenceLength, int embeddingDimension) {
        this.maxSequenceLength = maxSequenceLength;
        this.embeddingDimension = embeddingDimension;
        this.positionEncodings = generatePositionalEncodings();
    }
    
    /**
     * Generate sinusoidal positional encodings
     */
    private Matrix generatePositionalEncodings() {
        Matrix encodings = new Matrix(maxSequenceLength, embeddingDimension);
        
        for (int pos = 0; pos < maxSequenceLength; pos++) {
            for (int i = 0; i < embeddingDimension; i++) {
                double angle = pos / Math.pow(10000.0, (2.0 * (i / 2)) / embeddingDimension);
                
                if (i % 2 == 0) {
                    encodings.set(pos, i, Math.sin(angle));
                } else {
                    encodings.set(pos, i, Math.cos(angle));
                }
            }
        }
        
        return encodings;
    }
    
    /**
     * Add positional encodings to input embeddings
     * @param embeddings Input embeddings [sequence_length, embedding_dimension]
     * @return Embeddings with positional information added
     */
    public Matrix addPositionalEncoding(Matrix embeddings) {
        int sequenceLength = embeddings.getRows();
        
        if (sequenceLength > maxSequenceLength) {
            throw new IllegalArgumentException("Sequence length exceeds maximum allowed length");
        }
        
        Matrix result = new Matrix(sequenceLength, embeddingDimension);
        for (int i = 0; i < sequenceLength; i++) {
            for (int j = 0; j < embeddingDimension; j++) {
                result.set(i, j, embeddings.get(i, j) + positionEncodings.get(i, j));
            }
        }
        
        return result;
    }
    
    /**
     * Get positional encoding for a specific position
     */
    public Matrix getPositionEncoding(int position) {
        if (position >= maxSequenceLength) {
            throw new IllegalArgumentException("Position exceeds maximum sequence length");
        }
        
        Matrix encoding = new Matrix(1, embeddingDimension);
        for (int i = 0; i < embeddingDimension; i++) {
            encoding.set(0, i, positionEncodings.get(position, i));
        }
        
        return encoding;
    }
}
