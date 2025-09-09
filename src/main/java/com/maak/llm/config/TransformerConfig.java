package com.maak.llm.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "transformer")
public class TransformerConfig {
    
    private int vocabularySize = 50257; // GPT-2 vocabulary size
    private int maxSequenceLength = 1024;
    private int embeddingDimension = 768;
    private int numLayers = 12;
    private int numHeads = 12;
    private int feedForwardDimension = 3072;
    private double dropoutRate = 0.1;
    private double layerNormEpsilon = 1e-5;
    private String activationFunction = "gelu";
    
    // Getters and Setters
    public int getVocabularySize() {
        return vocabularySize;
    }
    
    public void setVocabularySize(int vocabularySize) {
        this.vocabularySize = vocabularySize;
    }
    
    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }
    
    public void setMaxSequenceLength(int maxSequenceLength) {
        this.maxSequenceLength = maxSequenceLength;
    }
    
    public int getEmbeddingDimension() {
        return embeddingDimension;
    }
    
    public void setEmbeddingDimension(int embeddingDimension) {
        this.embeddingDimension = embeddingDimension;
    }
    
    public int getNumLayers() {
        return numLayers;
    }
    
    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }
    
    public int getNumHeads() {
        return numHeads;
    }
    
    public void setNumHeads(int numHeads) {
        this.numHeads = numHeads;
    }
    
    public int getFeedForwardDimension() {
        return feedForwardDimension;
    }
    
    public void setFeedForwardDimension(int feedForwardDimension) {
        this.feedForwardDimension = feedForwardDimension;
    }
    
    public double getDropoutRate() {
        return dropoutRate;
    }
    
    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }
    
    public double getLayerNormEpsilon() {
        return layerNormEpsilon;
    }
    
    public void setLayerNormEpsilon(double layerNormEpsilon) {
        this.layerNormEpsilon = layerNormEpsilon;
    }
    
    public String getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public int getHeadDimension() {
        return embeddingDimension / numHeads;
    }
}
