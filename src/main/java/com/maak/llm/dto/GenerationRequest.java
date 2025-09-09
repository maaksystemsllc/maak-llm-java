package com.maak.llm.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

/**
 * Request DTO for text generation
 */
@Schema(description = "Request object for text generation with LLM Transformer")
public class GenerationRequest {
    
    @Schema(description = "Input text prompt for generation", example = "The future of artificial intelligence", required = true)
    @NotBlank(message = "Prompt cannot be blank")
    private String prompt;
    
    @Schema(description = "Maximum number of tokens to generate", example = "100", minimum = "1", maximum = "1000")
    @Min(value = 1, message = "Max length must be at least 1")
    @Max(value = 1000, message = "Max length cannot exceed 1000")
    private int maxLength = 100;
    
    @Schema(description = "Sampling temperature for randomness control (higher = more random)", example = "1.0", minimum = "0.1", maximum = "2.0")
    @Min(value = 0, message = "Temperature must be positive")
    @Max(value = 2, message = "Temperature cannot exceed 2.0")
    private double temperature = 1.0;
    
    @Schema(description = "Top-p sampling parameter for nucleus sampling", example = "0.9", minimum = "0.0", maximum = "1.0")
    @Min(value = 0, message = "Top-p must be between 0 and 1")
    @Max(value = 1, message = "Top-p must be between 0 and 1")
    private double topP = 0.9;
    
    @Schema(description = "Whether to include the original prompt in the response", example = "true")
    private boolean includePrompt = true;
    
    // Constructors
    public GenerationRequest() {}
    
    public GenerationRequest(String prompt, int maxLength, double temperature) {
        this.prompt = prompt;
        this.maxLength = maxLength;
        this.temperature = temperature;
    }
    
    // Getters and Setters
    public String getPrompt() {
        return prompt;
    }
    
    public void setPrompt(String prompt) {
        this.prompt = prompt;
    }
    
    public int getMaxLength() {
        return maxLength;
    }
    
    public void setMaxLength(int maxLength) {
        this.maxLength = maxLength;
    }
    
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
    
    public boolean isIncludePrompt() {
        return includePrompt;
    }
    
    public void setIncludePrompt(boolean includePrompt) {
        this.includePrompt = includePrompt;
    }
}
