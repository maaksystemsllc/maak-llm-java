package com.maak.llm.training;

import com.maak.llm.model.TransformerModel;
import com.maak.llm.training.DataLoader.TrainingBatch;
import com.maak.llm.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Main training service for the transformer model
 */
@Service
public class ModelTrainer {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelTrainer.class);
    
    private final TransformerModel model;
    private final DataLoader dataLoader;
    private final LossFunction lossFunction;
    private final Optimizer optimizer;
    private final TrainingConfig config;
    
    public ModelTrainer(TransformerModel model, DataLoader dataLoader, 
                       LossFunction lossFunction, Optimizer optimizer, TrainingConfig config) {
        this.model = model;
        this.dataLoader = dataLoader;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.config = config;
    }
    
    /**
     * Train the model on the provided dataset
     */
    public TrainingResult trainModel() {
        logger.info("Starting model training with config: lr={}, batch_size={}, epochs={}", 
                   config.getLearningRate(), config.getBatchSize(), config.getEpochs());
        
        // Load training data
        List<TrainingBatch> trainingBatches = dataLoader.loadTrainingData(
            config.getDataPath(), config.getBatchSize(), config.getMaxSequenceLength());
        
        if (trainingBatches.isEmpty()) {
            throw new RuntimeException("No training data found in: " + config.getDataPath());
        }
        
        // Create checkpoint directory
        createCheckpointDirectory();
        
        TrainingMetrics metrics = new TrainingMetrics();
        int globalStep = 0;
        
        // Training loop
        for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
            logger.info("Starting epoch {}/{}", epoch + 1, config.getEpochs());
            
            double epochLoss = 0.0;
            int batchCount = 0;
            
            for (TrainingBatch batch : trainingBatches) {
                globalStep++;
                
                // Forward pass and loss calculation
                double batchLoss = trainBatch(batch);
                epochLoss += batchLoss;
                batchCount++;
                
                // Log progress
                if (globalStep % 100 == 0) {
                    logger.info("Step {}: Loss = {:.4f}", globalStep, batchLoss);
                }
                
                // Save checkpoint
                if (globalStep % config.getSaveEverySteps() == 0) {
                    saveCheckpoint(epoch, globalStep, batchLoss);
                }
                
                // Evaluate model
                if (globalStep % config.getEvaluateEverySteps() == 0) {
                    double perplexity = evaluateModel(batch);
                    metrics.addPerplexity(perplexity);
                    logger.info("Step {}: Perplexity = {:.2f}", globalStep, perplexity);
                }
            }
            
            double avgEpochLoss = epochLoss / batchCount;
            metrics.addEpochLoss(avgEpochLoss);
            logger.info("Epoch {} completed. Average loss: {:.4f}", epoch + 1, avgEpochLoss);
        }
        
        logger.info("Training completed successfully!");
        return new TrainingResult(metrics, globalStep);
    }
    
    /**
     * Train on a single batch
     */
    private double trainBatch(TrainingBatch batch) {
        List<int[]> inputs = batch.getInputSequences();
        List<int[]> targets = batch.getTargetSequences();
        
        double totalLoss = 0.0;
        Map<String, Matrix> accumulatedGradients = new HashMap<>();
        
        for (int i = 0; i < inputs.size(); i++) {
            // Forward pass
            Matrix output = model.forward(inputs.get(i));
            
            // Calculate loss
            int[][] targetBatch = {targets.get(i)};
            LossFunction.LossResult lossResult = lossFunction.crossEntropyLoss(output, targetBatch);
            totalLoss += lossResult.getLoss();
            
            // Backward pass (simplified - in practice you'd need full backpropagation)
            Map<String, Matrix> gradients = computeGradients(inputs.get(i), lossResult.getGradients());
            
            // Accumulate gradients
            for (Map.Entry<String, Matrix> entry : gradients.entrySet()) {
                String paramName = entry.getKey();
                Matrix grad = entry.getValue();
                
                if (accumulatedGradients.containsKey(paramName)) {
                    Matrix accumulated = accumulatedGradients.get(paramName);
                    for (int r = 0; r < grad.getRows(); r++) {
                        for (int c = 0; c < grad.getCols(); c++) {
                            accumulated.set(r, c, accumulated.get(r, c) + grad.get(r, c));
                        }
                    }
                } else {
                    accumulatedGradients.put(paramName, grad);
                }
            }
        }
        
        // Average gradients
        for (Matrix grad : accumulatedGradients.values()) {
            for (int r = 0; r < grad.getRows(); r++) {
                for (int c = 0; c < grad.getCols(); c++) {
                    grad.set(r, c, grad.get(r, c) / inputs.size());
                }
            }
        }
        
        // Apply gradient clipping
        optimizer.clipGradients(accumulatedGradients, config.getGradientClipping());
        
        // Update parameters
        Map<String, Matrix> parameters = getModelParameters();
        double currentLR = calculateLearningRate(optimizer.getStep());
        optimizer.updateParameters(parameters, accumulatedGradients, currentLR);
        
        return totalLoss / inputs.size();
    }
    
    /**
     * Compute gradients (simplified implementation)
     */
    private Map<String, Matrix> computeGradients(int[] input, Matrix outputGradients) {
        Map<String, Matrix> gradients = new HashMap<>();
        
        // This is a simplified gradient computation
        // In a full implementation, you would need to implement backpropagation
        // through all layers of the transformer
        
        // For demonstration, create dummy gradients
        gradients.put("embedding_weights", new Matrix(model.getVocabSize(), model.getEmbeddingDim()));
        gradients.put("output_weights", new Matrix(model.getEmbeddingDim(), model.getVocabSize()));
        
        return gradients;
    }
    
    /**
     * Get model parameters for optimization
     */
    private Map<String, Matrix> getModelParameters() {
        Map<String, Matrix> parameters = new HashMap<>();
        
        // This would need to be implemented to return actual model parameters
        // For now, return empty map
        
        return parameters;
    }
    
    /**
     * Calculate learning rate with warmup
     */
    private double calculateLearningRate(int step) {
        if (step < config.getWarmupSteps()) {
            return config.getLearningRate() * step / config.getWarmupSteps();
        }
        return config.getLearningRate();
    }
    
    /**
     * Evaluate model perplexity
     */
    private double evaluateModel(TrainingBatch batch) {
        double totalLogProb = 0.0;
        int totalTokens = 0;
        
        List<int[]> inputs = batch.getInputSequences();
        List<int[]> targets = batch.getTargetSequences();
        
        for (int i = 0; i < inputs.size(); i++) {
            Matrix output = model.forward(inputs.get(i));
            int[][] targetBatch = {targets.get(i)};
            LossFunction.LossResult result = lossFunction.crossEntropyLoss(output, targetBatch);
            
            totalLogProb += result.getLoss() * targets.get(i).length;
            totalTokens += targets.get(i).length;
        }
        
        double avgLogProb = totalLogProb / totalTokens;
        return Math.exp(avgLogProb);
    }
    
    /**
     * Save model checkpoint
     */
    private void saveCheckpoint(int epoch, int step, double loss) {
        try {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String checkpointName = String.format("checkpoint_epoch_%d_step_%d_%s.json", epoch, step, timestamp);
            Path checkpointPath = Paths.get(config.getCheckpointPath(), checkpointName);
            
            // Save model state (simplified)
            String modelState = String.format(
                "{\"epoch\": %d, \"step\": %d, \"loss\": %.6f, \"timestamp\": \"%s\"}", 
                epoch, step, loss, timestamp);
            
            Files.write(checkpointPath, modelState.getBytes());
            logger.info("Saved checkpoint: {}", checkpointName);
            
        } catch (IOException e) {
            logger.error("Failed to save checkpoint", e);
        }
    }
    
    /**
     * Create checkpoint directory
     */
    private void createCheckpointDirectory() {
        try {
            Path checkpointDir = Paths.get(config.getCheckpointPath());
            if (!Files.exists(checkpointDir)) {
                Files.createDirectories(checkpointDir);
                logger.info("Created checkpoint directory: {}", checkpointDir);
            }
        } catch (IOException e) {
            logger.error("Failed to create checkpoint directory", e);
            throw new RuntimeException("Cannot create checkpoint directory", e);
        }
    }
    
    /**
     * Training metrics collector
     */
    public static class TrainingMetrics {
        private final java.util.List<Double> epochLosses = new java.util.ArrayList<>();
        private final java.util.List<Double> perplexities = new java.util.ArrayList<>();
        
        public void addEpochLoss(double loss) {
            epochLosses.add(loss);
        }
        
        public void addPerplexity(double perplexity) {
            perplexities.add(perplexity);
        }
        
        public java.util.List<Double> getEpochLosses() {
            return epochLosses;
        }
        
        public java.util.List<Double> getPerplexities() {
            return perplexities;
        }
    }
    
    /**
     * Training result
     */
    public static class TrainingResult {
        private final TrainingMetrics metrics;
        private final int totalSteps;
        
        public TrainingResult(TrainingMetrics metrics, int totalSteps) {
            this.metrics = metrics;
            this.totalSteps = totalSteps;
        }
        
        public TrainingMetrics getMetrics() {
            return metrics;
        }
        
        public int getTotalSteps() {
            return totalSteps;
        }
    }
}
