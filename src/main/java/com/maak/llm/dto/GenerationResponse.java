package com.maak.llm.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.time.LocalDateTime;

/**
 * Response DTO for text generation
 */
@Schema(description = "Response object containing generated text and metadata")
public class GenerationResponse {
    
    @Schema(description = "The generated text output", example = "The future of artificial intelligence looks promising with advances in machine learning...")
    private String generatedText;
    
    @Schema(description = "The original input prompt", example = "The future of artificial intelligence")
    private String originalPrompt;
    
    @Schema(description = "Number of tokens generated", example = "42")
    private int tokensGenerated;
    
    @Schema(description = "Perplexity score of the generated text (lower is better)", example = "15.7")
    private double perplexity;
    
    @Schema(description = "Processing time in milliseconds", example = "1250")
    private long processingTimeMs;
    
    @Schema(description = "Timestamp when the generation was completed")
    private LocalDateTime timestamp;
    
    @Schema(description = "Additional metadata about the generation process")
    private GenerationMetadata metadata;
    
    // Constructors
    public GenerationResponse() {
        this.timestamp = LocalDateTime.now();
    }
    
    public GenerationResponse(String generatedText, String originalPrompt) {
        this();
        this.generatedText = generatedText;
        this.originalPrompt = originalPrompt;
    }
    
    // Getters and Setters
    public String getGeneratedText() {
        return generatedText;
    }
    
    public void setGeneratedText(String generatedText) {
        this.generatedText = generatedText;
    }
    
    public String getOriginalPrompt() {
        return originalPrompt;
    }
    
    public void setOriginalPrompt(String originalPrompt) {
        this.originalPrompt = originalPrompt;
    }
    
    public int getTokensGenerated() {
        return tokensGenerated;
    }
    
    public void setTokensGenerated(int tokensGenerated) {
        this.tokensGenerated = tokensGenerated;
    }
    
    public double getPerplexity() {
        return perplexity;
    }
    
    public void setPerplexity(double perplexity) {
        this.perplexity = perplexity;
    }
    
    public long getProcessingTimeMs() {
        return processingTimeMs;
    }
    
    public void setProcessingTimeMs(long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public GenerationMetadata getMetadata() {
        return metadata;
    }
    
    public void setMetadata(GenerationMetadata metadata) {
        this.metadata = metadata;
    }
    
    /**
     * Nested class for additional generation metadata
     */
    public static class GenerationMetadata {
        private double temperature;
        private double topP;
        private int maxLength;
        private String modelVersion;
        
        public GenerationMetadata() {}
        
        public GenerationMetadata(double temperature, double topP, int maxLength, String modelVersion) {
            this.temperature = temperature;
            this.topP = topP;
            this.maxLength = maxLength;
            this.modelVersion = modelVersion;
        }
        
        // Getters and Setters
        public double getTemperature() {
            return temperature;
        }
        
        public void setTemperature(double temperature) {
            this.temperature = temperature;
        }
        
        public double getTopP() {
            return topP;
        }
        
        public void setTopP(double topP) {
            this.topP = topP;
        }
        
        public int getMaxLength() {
            return maxLength;
        }
        
        public void setMaxLength(int maxLength) {
            this.maxLength = maxLength;
        }
        
        public String getModelVersion() {
            return modelVersion;
        }
        
        public void setModelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
        }
    }
}
