package com.maak.llm.persistence;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.maak.llm.config.TransformerConfig;
import com.maak.llm.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * Model persistence for saving and loading transformer models
 */
@Component
public class ModelPersistence {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelPersistence.class);
    private final ObjectMapper objectMapper;
    
    public ModelPersistence() {
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Save model weights and configuration to disk
     */
    public void saveModel(String modelPath, TransformerConfig config, Map<String, Matrix> weights) {
        try {
            Path path = Paths.get(modelPath);
            Files.createDirectories(path.getParent());
            
            // Create model data structure
            Map<String, Object> modelData = new HashMap<>();
            modelData.put("config", config);
            modelData.put("weights", serializeWeights(weights));
            modelData.put("version", "1.0.0");
            modelData.put("timestamp", System.currentTimeMillis());
            
            // Save to JSON file
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(new File(modelPath), modelData);
            
            logger.info("Model saved successfully to: {}", modelPath);
            
        } catch (Exception e) {
            logger.error("Error saving model to: {}", modelPath, e);
            throw new RuntimeException("Failed to save model", e);
        }
    }
    
    /**
     * Load model weights and configuration from disk
     */
    @SuppressWarnings("unchecked")
    public ModelData loadModel(String modelPath) {
        try {
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                throw new FileNotFoundException("Model file not found: " + modelPath);
            }
            
            // Load from JSON file
            Map<String, Object> modelData = objectMapper.readValue(modelFile, Map.class);
            
            // Extract configuration
            Map<String, Object> configMap = (Map<String, Object>) modelData.get("config");
            TransformerConfig config = objectMapper.convertValue(configMap, TransformerConfig.class);
            
            // Extract weights
            Map<String, Object> weightsData = (Map<String, Object>) modelData.get("weights");
            Map<String, Matrix> weights = deserializeWeights(weightsData);
            
            String version = (String) modelData.get("version");
            Long timestamp = (Long) modelData.get("timestamp");
            
            logger.info("Model loaded successfully from: {}", modelPath);
            
            return new ModelData(config, weights, version, timestamp);
            
        } catch (Exception e) {
            logger.error("Error loading model from: {}", modelPath, e);
            throw new RuntimeException("Failed to load model", e);
        }
    }
    
    /**
     * Serialize matrix weights to a JSON-compatible format
     */
    private Map<String, Object> serializeWeights(Map<String, Matrix> weights) {
        Map<String, Object> serializedWeights = new HashMap<>();
        
        for (Map.Entry<String, Matrix> entry : weights.entrySet()) {
            Matrix matrix = entry.getValue();
            Map<String, Object> matrixData = new HashMap<>();
            matrixData.put("rows", matrix.getRows());
            matrixData.put("cols", matrix.getCols());
            matrixData.put("data", matrix.getData());
            
            serializedWeights.put(entry.getKey(), matrixData);
        }
        
        return serializedWeights;
    }
    
    /**
     * Deserialize matrix weights from JSON format
     */
    @SuppressWarnings("unchecked")
    private Map<String, Matrix> deserializeWeights(Map<String, Object> weightsData) {
        Map<String, Matrix> weights = new HashMap<>();
        
        for (Map.Entry<String, Object> entry : weightsData.entrySet()) {
            Map<String, Object> matrixData = (Map<String, Object>) entry.getValue();
            
            int rows = (Integer) matrixData.get("rows");
            int cols = (Integer) matrixData.get("cols");
            double[][] data = objectMapper.convertValue(matrixData.get("data"), double[][].class);
            
            Matrix matrix = new Matrix(data);
            weights.put(entry.getKey(), matrix);
        }
        
        return weights;
    }
    
    /**
     * Save model in binary format for better performance
     */
    public void saveModelBinary(String modelPath, TransformerConfig config, Map<String, Matrix> weights) {
        try {
            Path path = Paths.get(modelPath);
            Files.createDirectories(path.getParent());
            
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath))) {
                // Write version
                oos.writeUTF("1.0.0");
                oos.writeLong(System.currentTimeMillis());
                
                // Write config as JSON
                String configJson = objectMapper.writeValueAsString(config);
                oos.writeUTF(configJson);
                
                // Write weights
                oos.writeInt(weights.size());
                for (Map.Entry<String, Matrix> entry : weights.entrySet()) {
                    oos.writeUTF(entry.getKey());
                    Matrix matrix = entry.getValue();
                    oos.writeInt(matrix.getRows());
                    oos.writeInt(matrix.getCols());
                    
                    // Write matrix data
                    for (int i = 0; i < matrix.getRows(); i++) {
                        for (int j = 0; j < matrix.getCols(); j++) {
                            oos.writeDouble(matrix.get(i, j));
                        }
                    }
                }
            }
            
            logger.info("Model saved in binary format to: {}", modelPath);
            
        } catch (Exception e) {
            logger.error("Error saving model in binary format to: {}", modelPath, e);
            throw new RuntimeException("Failed to save model in binary format", e);
        }
    }
    
    /**
     * Load model from binary format
     */
    public ModelData loadModelBinary(String modelPath) {
        try {
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                throw new FileNotFoundException("Model file not found: " + modelPath);
            }
            
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFile))) {
                // Read version and timestamp
                String version = ois.readUTF();
                long timestamp = ois.readLong();
                
                // Read config
                String configJson = ois.readUTF();
                TransformerConfig config = objectMapper.readValue(configJson, TransformerConfig.class);
                
                // Read weights
                int numWeights = ois.readInt();
                Map<String, Matrix> weights = new HashMap<>();
                
                for (int w = 0; w < numWeights; w++) {
                    String weightName = ois.readUTF();
                    int rows = ois.readInt();
                    int cols = ois.readInt();
                    
                    Matrix matrix = new Matrix(rows, cols);
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) {
                            matrix.set(i, j, ois.readDouble());
                        }
                    }
                    
                    weights.put(weightName, matrix);
                }
                
                logger.info("Model loaded from binary format: {}", modelPath);
                
                return new ModelData(config, weights, version, timestamp);
            }
            
        } catch (Exception e) {
            logger.error("Error loading model from binary format: {}", modelPath, e);
            throw new RuntimeException("Failed to load model from binary format", e);
        }
    }
    
    /**
     * Check if model file exists
     */
    public boolean modelExists(String modelPath) {
        return Files.exists(Paths.get(modelPath));
    }
    
    /**
     * Data class for loaded model information
     */
    public static class ModelData {
        private final TransformerConfig config;
        private final Map<String, Matrix> weights;
        private final String version;
        private final Long timestamp;
        
        public ModelData(TransformerConfig config, Map<String, Matrix> weights, String version, Long timestamp) {
            this.config = config;
            this.weights = weights;
            this.version = version;
            this.timestamp = timestamp;
        }
        
        public TransformerConfig getConfig() { return config; }
        public Map<String, Matrix> getWeights() { return weights; }
        public String getVersion() { return version; }
        public Long getTimestamp() { return timestamp; }
    }
}
