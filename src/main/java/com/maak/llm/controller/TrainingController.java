package com.maak.llm.controller;

import com.maak.llm.training.ModelTrainer;
import com.maak.llm.training.TrainingConfig;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * REST API endpoints for model training
 */
@RestController
@RequestMapping("/api/training")
@Tag(name = "Training", description = "Model training operations")
@CrossOrigin(origins = "*")
public class TrainingController {
    
    private static final Logger logger = LoggerFactory.getLogger(TrainingController.class);
    
    private final ModelTrainer trainer;
    private final TrainingConfig config;
    private CompletableFuture<ModelTrainer.TrainingResult> currentTraining;
    
    public TrainingController(ModelTrainer trainer, TrainingConfig config) {
        this.trainer = trainer;
        this.config = config;
    }
    
    @PostMapping("/start")
    @Operation(summary = "Start model training", description = "Begin training the transformer model with the configured dataset")
    public ResponseEntity<Map<String, Object>> startTraining() {
        Map<String, Object> response = new HashMap<>();
        
        try {
            if (currentTraining != null && !currentTraining.isDone()) {
                response.put("status", "error");
                response.put("message", "Training is already in progress");
                return ResponseEntity.badRequest().body(response);
            }
            
            // Start training asynchronously
            currentTraining = CompletableFuture.supplyAsync(() -> {
                try {
                    return trainer.trainModel();
                } catch (Exception e) {
                    logger.error("Training failed", e);
                    throw new RuntimeException("Training failed: " + e.getMessage(), e);
                }
            });
            
            response.put("status", "started");
            response.put("message", "Training started successfully");
            response.put("config", getTrainingConfigInfo());
            
            logger.info("Training started");
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            logger.error("Failed to start training", e);
            response.put("status", "error");
            response.put("message", "Failed to start training: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    @GetMapping("/status")
    @Operation(summary = "Get training status", description = "Check the current status of model training")
    public ResponseEntity<Map<String, Object>> getTrainingStatus() {
        Map<String, Object> response = new HashMap<>();
        
        if (currentTraining == null) {
            response.put("status", "not_started");
            response.put("message", "No training session has been started");
        } else if (currentTraining.isDone()) {
            try {
                ModelTrainer.TrainingResult result = currentTraining.get();
                response.put("status", "completed");
                response.put("message", "Training completed successfully");
                response.put("total_steps", result.getTotalSteps());
                response.put("final_losses", result.getMetrics().getEpochLosses());
                response.put("perplexities", result.getMetrics().getPerplexities());
            } catch (Exception e) {
                response.put("status", "failed");
                response.put("message", "Training failed: " + e.getMessage());
            }
        } else {
            response.put("status", "running");
            response.put("message", "Training is currently in progress");
        }
        
        return ResponseEntity.ok(response);
    }
    
    @PostMapping("/stop")
    @Operation(summary = "Stop training", description = "Stop the current training session")
    public ResponseEntity<Map<String, Object>> stopTraining() {
        Map<String, Object> response = new HashMap<>();
        
        if (currentTraining != null && !currentTraining.isDone()) {
            currentTraining.cancel(true);
            response.put("status", "stopped");
            response.put("message", "Training stopped successfully");
            logger.info("Training stopped by user request");
        } else {
            response.put("status", "not_running");
            response.put("message", "No training session is currently running");
        }
        
        return ResponseEntity.ok(response);
    }
    
    @GetMapping("/config")
    @Operation(summary = "Get training configuration", description = "Retrieve current training configuration")
    public ResponseEntity<Map<String, Object>> getTrainingConfig() {
        return ResponseEntity.ok(getTrainingConfigInfo());
    }
    
    @PostMapping("/config")
    @Operation(summary = "Update training configuration", description = "Update training parameters")
    public ResponseEntity<Map<String, Object>> updateTrainingConfig(
            @Parameter(description = "Learning rate") @RequestParam(required = false) Double learningRate,
            @Parameter(description = "Batch size") @RequestParam(required = false) Integer batchSize,
            @Parameter(description = "Number of epochs") @RequestParam(required = false) Integer epochs,
            @Parameter(description = "Maximum sequence length") @RequestParam(required = false) Integer maxSequenceLength,
            @Parameter(description = "Data path") @RequestParam(required = false) String dataPath) {
        
        Map<String, Object> response = new HashMap<>();
        
        try {
            if (currentTraining != null && !currentTraining.isDone()) {
                response.put("status", "error");
                response.put("message", "Cannot update configuration while training is in progress");
                return ResponseEntity.badRequest().body(response);
            }
            
            // Update configuration
            if (learningRate != null) config.setLearningRate(learningRate);
            if (batchSize != null) config.setBatchSize(batchSize);
            if (epochs != null) config.setEpochs(epochs);
            if (maxSequenceLength != null) config.setMaxSequenceLength(maxSequenceLength);
            if (dataPath != null) config.setDataPath(dataPath);
            
            response.put("status", "updated");
            response.put("message", "Configuration updated successfully");
            response.put("config", getTrainingConfigInfo());
            
            logger.info("Training configuration updated");
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            logger.error("Failed to update configuration", e);
            response.put("status", "error");
            response.put("message", "Failed to update configuration: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
    
    private Map<String, Object> getTrainingConfigInfo() {
        Map<String, Object> configInfo = new HashMap<>();
        configInfo.put("learning_rate", config.getLearningRate());
        configInfo.put("batch_size", config.getBatchSize());
        configInfo.put("epochs", config.getEpochs());
        configInfo.put("max_sequence_length", config.getMaxSequenceLength());
        configInfo.put("weight_decay", config.getWeightDecay());
        configInfo.put("gradient_clipping", config.getGradientClipping());
        configInfo.put("warmup_steps", config.getWarmupSteps());
        configInfo.put("data_path", config.getDataPath());
        configInfo.put("checkpoint_path", config.getCheckpointPath());
        return configInfo;
    }
}
