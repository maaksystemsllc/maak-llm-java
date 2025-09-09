package com.maak.llm.training;

import com.maak.llm.embedding.TokenEmbedding;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

/**
 * Data loader for training text data
 */
@Component
public class DataLoader {
    
    private static final Logger logger = LoggerFactory.getLogger(DataLoader.class);
    
    private final TokenEmbedding tokenEmbedding;
    
    public DataLoader(TokenEmbedding tokenEmbedding) {
        this.tokenEmbedding = tokenEmbedding;
    }
    
    /**
     * Load training data from text files
     */
    public List<TrainingBatch> loadTrainingData(String dataPath, int batchSize, int maxSequenceLength) {
        List<TrainingBatch> batches = new ArrayList<>();
        
        try {
            List<String> textFiles = findTextFiles(dataPath);
            List<String> allTexts = new ArrayList<>();
            
            // Read all text files
            for (String filePath : textFiles) {
                List<String> texts = readTextFile(filePath);
                allTexts.addAll(texts);
                logger.info("Loaded {} texts from {}", texts.size(), filePath);
            }
            
            // Shuffle data
            Collections.shuffle(allTexts);
            
            // Create batches
            for (int i = 0; i < allTexts.size(); i += batchSize) {
                int endIdx = Math.min(i + batchSize, allTexts.size());
                List<String> batchTexts = allTexts.subList(i, endIdx);
                
                TrainingBatch batch = createBatch(batchTexts, maxSequenceLength);
                if (batch != null) {
                    batches.add(batch);
                }
            }
            
            logger.info("Created {} training batches from {} texts", batches.size(), allTexts.size());
            
        } catch (IOException e) {
            logger.error("Error loading training data", e);
            throw new RuntimeException("Failed to load training data", e);
        }
        
        return batches;
    }
    
    /**
     * Find all text files in the data directory
     */
    private List<String> findTextFiles(String dataPath) throws IOException {
        List<String> textFiles = new ArrayList<>();
        Path path = Paths.get(dataPath);
        
        if (!Files.exists(path)) {
            throw new IOException("Data path does not exist: " + dataPath);
        }
        
        try (Stream<Path> paths = Files.walk(path)) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".txt"))
                 .forEach(p -> textFiles.add(p.toString()));
        }
        
        return textFiles;
    }
    
    /**
     * Read text from a file, splitting into sentences or paragraphs
     */
    private List<String> readTextFile(String filePath) throws IOException {
        List<String> texts = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            StringBuilder currentText = new StringBuilder();
            
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                
                if (line.isEmpty()) {
                    // Empty line indicates paragraph break
                    if (currentText.length() > 0) {
                        texts.add(currentText.toString().trim());
                        currentText = new StringBuilder();
                    }
                } else {
                    currentText.append(line).append(" ");
                }
            }
            
            // Add the last text if any
            if (currentText.length() > 0) {
                texts.add(currentText.toString().trim());
            }
        }
        
        return texts;
    }
    
    /**
     * Create a training batch from text samples
     */
    private TrainingBatch createBatch(List<String> texts, int maxSequenceLength) {
        List<int[]> inputSequences = new ArrayList<>();
        List<int[]> targetSequences = new ArrayList<>();
        
        for (String text : texts) {
            int[] tokens = tokenEmbedding.tokenize(text);
            
            if (tokens.length < 2) {
                continue; // Skip very short sequences
            }
            
            // Truncate if too long
            int seqLength = Math.min(tokens.length, maxSequenceLength);
            
            // Input is tokens[0:seqLength-1], target is tokens[1:seqLength]
            int[] input = new int[seqLength - 1];
            int[] target = new int[seqLength - 1];
            
            System.arraycopy(tokens, 0, input, 0, seqLength - 1);
            System.arraycopy(tokens, 1, target, 0, seqLength - 1);
            
            inputSequences.add(input);
            targetSequences.add(target);
        }
        
        if (inputSequences.isEmpty()) {
            return null;
        }
        
        return new TrainingBatch(inputSequences, targetSequences);
    }
    
    /**
     * Training batch data structure
     */
    public static class TrainingBatch {
        private final List<int[]> inputSequences;
        private final List<int[]> targetSequences;
        
        public TrainingBatch(List<int[]> inputSequences, List<int[]> targetSequences) {
            this.inputSequences = inputSequences;
            this.targetSequences = targetSequences;
        }
        
        public List<int[]> getInputSequences() {
            return inputSequences;
        }
        
        public List<int[]> getTargetSequences() {
            return targetSequences;
        }
        
        public int getBatchSize() {
            return inputSequences.size();
        }
    }
}
