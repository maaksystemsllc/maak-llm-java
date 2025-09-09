package com.maak.llm.model;

import com.maak.llm.config.TransformerConfig;
import com.maak.llm.embedding.PositionalEncoding;
import com.maak.llm.embedding.TokenEmbedding;
import com.maak.llm.layers.MultiHeadAttention;
import com.maak.llm.layers.TransformerBlock;
import com.maak.llm.math.Matrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * Complete Transformer Language Model
 */
@Component
public class TransformerModel {
    
    private final TransformerConfig config;
    private final TokenEmbedding tokenEmbedding;
    private final PositionalEncoding positionalEncoding;
    private final List<TransformerBlock> transformerBlocks;
    private final Matrix outputProjection;
    
    @Autowired
    public TransformerModel(TransformerConfig config, TokenEmbedding tokenEmbedding, PositionalEncoding positionalEncoding) {
        this.config = config;
        this.tokenEmbedding = tokenEmbedding;
        this.positionalEncoding = positionalEncoding;
        
        // Initialize transformer blocks
        this.transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNumLayers(); i++) {
            transformerBlocks.add(new TransformerBlock(
                config.getEmbeddingDimension(),
                config.getNumHeads(),
                config.getFeedForwardDimension(),
                config.getActivationFunction(),
                config.getLayerNormEpsilon(),
                config.getDropoutRate()
            ));
        }
        
        // Output projection to vocabulary
        double scale = Math.sqrt(2.0 / config.getEmbeddingDimension());
        this.outputProjection = Matrix.random(config.getEmbeddingDimension(), config.getVocabularySize(), scale);
    }
    
    /**
     * Forward pass through the entire model
     * @param inputTokens Array of token IDs
     * @return Logits over vocabulary [sequence_length, vocabulary_size]
     */
    public Matrix forward(int[] inputTokens) {
        // Token embedding
        Matrix embeddings = tokenEmbedding.embed(inputTokens);
        
        // Add positional encoding
        Matrix input = positionalEncoding.addPositionalEncoding(embeddings);
        
        // Create causal mask for autoregressive generation
        Matrix mask = MultiHeadAttention.createCausalMask(inputTokens.length);
        
        // Pass through transformer blocks
        Matrix hidden = input;
        for (TransformerBlock block : transformerBlocks) {
            hidden = block.forward(hidden, mask);
        }
        
        // Final layer normalization
        hidden = hidden.layerNorm(config.getLayerNormEpsilon());
        
        // Project to vocabulary
        return hidden.multiply(outputProjection);
    }
    
    /**
     * Generate text using the model
     * @param prompt Initial prompt text
     * @param maxLength Maximum length of generated text
     * @param temperature Sampling temperature (higher = more random)
     * @return Generated text
     */
    public String generate(String prompt, int maxLength, double temperature) {
        // Tokenize prompt
        int[] promptTokens = tokenEmbedding.tokenize(prompt);
        List<Integer> generatedTokens = new ArrayList<>();
        
        // Add prompt tokens
        for (int token : promptTokens) {
            generatedTokens.add(token);
        }
        
        // Generate tokens one by one
        for (int i = 0; i < maxLength; i++) {
            // Convert list to array
            int[] currentTokens = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
            
            // Forward pass
            Matrix logits = forward(currentTokens);
            
            // Get logits for the last position
            Matrix lastLogits = new Matrix(1, config.getVocabularySize());
            int lastPos = logits.getRows() - 1;
            for (int j = 0; j < config.getVocabularySize(); j++) {
                lastLogits.set(0, j, logits.get(lastPos, j) / temperature);
            }
            
            // Apply softmax to get probabilities
            Matrix probabilities = lastLogits.softmax();
            
            // Sample next token
            int nextToken = sampleFromProbabilities(probabilities);
            
            // Check for end of sequence
            if (nextToken == tokenEmbedding.getTokenId("<eos>")) {
                break;
            }
            
            generatedTokens.add(nextToken);
        }
        
        // Convert back to text
        int[] finalTokens = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        return tokenEmbedding.detokenize(finalTokens);
    }
    
    /**
     * Sample a token from probability distribution
     */
    private int sampleFromProbabilities(Matrix probabilities) {
        double random = Math.random();
        double cumulative = 0.0;
        
        for (int i = 0; i < probabilities.getCols(); i++) {
            cumulative += probabilities.get(0, i);
            if (random <= cumulative) {
                return i;
            }
        }
        
        // Fallback to last token
        return probabilities.getCols() - 1;
    }
    
    /**
     * Calculate perplexity on a given text
     */
    public double calculatePerplexity(String text) {
        int[] tokens = tokenEmbedding.tokenize(text);
        if (tokens.length <= 1) {
            return Double.POSITIVE_INFINITY;
        }
        
        double totalLogLikelihood = 0.0;
        int numTokens = 0;
        
        // Calculate log likelihood for each token given previous context
        for (int i = 1; i < tokens.length; i++) {
            int[] context = new int[i];
            System.arraycopy(tokens, 0, context, 0, i);
            
            Matrix logits = forward(context);
            Matrix probabilities = logits.softmax();
            
            // Get probability of the target token
            int targetToken = tokens[i];
            double probability = probabilities.get(probabilities.getRows() - 1, targetToken);
            
            if (probability > 0) {
                totalLogLikelihood += Math.log(probability);
                numTokens++;
            }
        }
        
        if (numTokens == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        double averageLogLikelihood = totalLogLikelihood / numTokens;
        return Math.exp(-averageLogLikelihood);
    }
    
    public TransformerConfig getConfig() {
        return config;
    }
    
    public TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }
}
