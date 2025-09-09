package com.maak.llm.training;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "training")
public class TrainingConfig {
    
    private double learningRate = 0.0001;
    private int batchSize = 32;
    private int epochs = 10;
    private int maxSequenceLength = 512;
    private double weightDecay = 0.01;
    private double gradientClipping = 1.0;
    private int warmupSteps = 4000;
    private int saveEverySteps = 1000;
    private int evaluateEverySteps = 500;
    private String dataPath = "data/training";
    private String checkpointPath = "checkpoints";
    private boolean useGradientAccumulation = true;
    private int accumulationSteps = 4;
    
    // Getters and Setters
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    
    public int getEpochs() {
        return epochs;
    }
    
    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }
    
    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }
    
    public void setMaxSequenceLength(int maxSequenceLength) {
        this.maxSequenceLength = maxSequenceLength;
    }
    
    public double getWeightDecay() {
        return weightDecay;
    }
    
    public void setWeightDecay(double weightDecay) {
        this.weightDecay = weightDecay;
    }
    
    public double getGradientClipping() {
        return gradientClipping;
    }
    
    public void setGradientClipping(double gradientClipping) {
        this.gradientClipping = gradientClipping;
    }
    
    public int getWarmupSteps() {
        return warmupSteps;
    }
    
    public void setWarmupSteps(int warmupSteps) {
        this.warmupSteps = warmupSteps;
    }
    
    public int getSaveEverySteps() {
        return saveEverySteps;
    }
    
    public void setSaveEverySteps(int saveEverySteps) {
        this.saveEverySteps = saveEverySteps;
    }
    
    public int getEvaluateEverySteps() {
        return evaluateEverySteps;
    }
    
    public void setEvaluateEverySteps(int evaluateEverySteps) {
        this.evaluateEverySteps = evaluateEverySteps;
    }
    
    public String getDataPath() {
        return dataPath;
    }
    
    public void setDataPath(String dataPath) {
        this.dataPath = dataPath;
    }
    
    public String getCheckpointPath() {
        return checkpointPath;
    }
    
    public void setCheckpointPath(String checkpointPath) {
        this.checkpointPath = checkpointPath;
    }
    
    public boolean isUseGradientAccumulation() {
        return useGradientAccumulation;
    }
    
    public void setUseGradientAccumulation(boolean useGradientAccumulation) {
        this.useGradientAccumulation = useGradientAccumulation;
    }
    
    public int getAccumulationSteps() {
        return accumulationSteps;
    }
    
    public void setAccumulationSteps(int accumulationSteps) {
        this.accumulationSteps = accumulationSteps;
    }
}
