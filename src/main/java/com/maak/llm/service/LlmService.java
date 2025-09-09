package com.maak.llm.service;

import com.maak.llm.dto.GenerationRequest;
import com.maak.llm.dto.GenerationResponse;
import com.maak.llm.model.TransformerModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

/**
 * Service for LLM operations
 */
@Service
public class LlmService {
    
    private static final Logger logger = LoggerFactory.getLogger(LlmService.class);
    
    private final TransformerModel model;
    
    @Autowired
    public LlmService(TransformerModel model) {
        this.model = model;
    }
    
    /**
     * Generate text synchronously
     */
    public GenerationResponse generateText(GenerationRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            logger.info("Starting text generation for prompt: {}", 
                       request.getPrompt().substring(0, Math.min(50, request.getPrompt().length())));
            
            // Generate text using the model
            String generatedText = model.generate(
                request.getPrompt(), 
                request.getMaxLength(), 
                request.getTemperature()
            );
            
            // Calculate processing time
            long processingTime = System.currentTimeMillis() - startTime;
            
            // Create response
            GenerationResponse response = new GenerationResponse(generatedText, request.getPrompt());
            response.setProcessingTimeMs(processingTime);
            
            // Calculate additional metrics
            int[] tokens = model.getTokenEmbedding().tokenize(generatedText);
            response.setTokensGenerated(tokens.length);
            
            try {
                double perplexity = model.calculatePerplexity(generatedText);
                response.setPerplexity(perplexity);
            } catch (Exception e) {
                logger.warn("Could not calculate perplexity: {}", e.getMessage());
                response.setPerplexity(-1.0);
            }
            
            // Set metadata
            GenerationResponse.GenerationMetadata metadata = new GenerationResponse.GenerationMetadata(
                request.getTemperature(),
                request.getTopP(),
                request.getMaxLength(),
                "transformer-v1.0"
            );
            response.setMetadata(metadata);
            
            logger.info("Text generation completed in {}ms, generated {} tokens", 
                       processingTime, tokens.length);
            
            return response;
            
        } catch (Exception e) {
            logger.error("Error during text generation", e);
            throw new RuntimeException("Text generation failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Generate text asynchronously
     */
    @Async("taskExecutor")
    public CompletableFuture<GenerationResponse> generateTextAsync(GenerationRequest request) {
        try {
            GenerationResponse response = generateText(request);
            return CompletableFuture.completedFuture(response);
        } catch (Exception e) {
            CompletableFuture<GenerationResponse> future = new CompletableFuture<>();
            future.completeExceptionally(e);
            return future;
        }
    }
    
    /**
     * Calculate perplexity for given text
     */
    public double calculatePerplexity(String text) {
        try {
            return model.calculatePerplexity(text);
        } catch (Exception e) {
            logger.error("Error calculating perplexity", e);
            throw new RuntimeException("Perplexity calculation failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Tokenize text
     */
    public int[] tokenize(String text) {
        return model.getTokenEmbedding().tokenize(text);
    }
    
    /**
     * Detokenize token IDs back to text
     */
    public String detokenize(int[] tokenIds) {
        return model.getTokenEmbedding().detokenize(tokenIds);
    }
    
    /**
     * Get model information
     */
    public ModelInfo getModelInfo() {
        return new ModelInfo(
            model.getConfig().getVocabularySize(),
            model.getConfig().getEmbeddingDimension(),
            model.getConfig().getNumLayers(),
            model.getConfig().getNumHeads(),
            model.getConfig().getMaxSequenceLength(),
            "transformer-v1.0"
        );
    }
    
    /**
     * Model information DTO
     */
    public static class ModelInfo {
        private final int vocabularySize;
        private final int embeddingDimension;
        private final int numLayers;
        private final int numHeads;
        private final int maxSequenceLength;
        private final String version;
        
        public ModelInfo(int vocabularySize, int embeddingDimension, int numLayers, 
                        int numHeads, int maxSequenceLength, String version) {
            this.vocabularySize = vocabularySize;
            this.embeddingDimension = embeddingDimension;
            this.numLayers = numLayers;
            this.numHeads = numHeads;
            this.maxSequenceLength = maxSequenceLength;
            this.version = version;
        }
        
        // Getters
        public int getVocabularySize() { return vocabularySize; }
        public int getEmbeddingDimension() { return embeddingDimension; }
        public int getNumLayers() { return numLayers; }
        public int getNumHeads() { return numHeads; }
        public int getMaxSequenceLength() { return maxSequenceLength; }
        public String getVersion() { return version; }
    }
}
