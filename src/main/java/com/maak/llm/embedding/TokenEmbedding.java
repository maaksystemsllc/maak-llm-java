package com.maak.llm.embedding;

import com.maak.llm.math.Matrix;

import java.util.HashMap;
import java.util.Map;

/**
 * Token Embedding layer for converting tokens to dense vectors
 */
public class TokenEmbedding {
    
    private final Matrix embeddingMatrix;
    private final int vocabularySize;
    private final int embeddingDimension;
    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;
    
    public TokenEmbedding(int vocabularySize, int embeddingDimension) {
        this.vocabularySize = vocabularySize;
        this.embeddingDimension = embeddingDimension;
        
        // Initialize embedding matrix with small random values
        double scale = Math.sqrt(1.0 / embeddingDimension);
        this.embeddingMatrix = Matrix.random(vocabularySize, embeddingDimension, scale);
        
        // Initialize token mappings (simplified - in practice would load from tokenizer)
        this.tokenToId = new HashMap<>();
        this.idToToken = new HashMap<>();
        initializeBasicVocabulary();
    }
    
    /**
     * Initialize a basic vocabulary (in practice, this would be loaded from a tokenizer)
     */
    private void initializeBasicVocabulary() {
        // Special tokens
        tokenToId.put("<pad>", 0);
        tokenToId.put("<unk>", 1);
        tokenToId.put("<bos>", 2);
        tokenToId.put("<eos>", 3);
        
        idToToken.put(0, "<pad>");
        idToToken.put(1, "<unk>");
        idToToken.put(2, "<bos>");
        idToToken.put(3, "<eos>");
        
        // Add some common words (simplified vocabulary)
        String[] commonWords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "must", "shall",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "this", "that", "these", "those", "what", "which", "who", "when", "where", "why", "how"
        };
        
        int currentId = 4;
        for (String word : commonWords) {
            if (currentId < vocabularySize) {
                tokenToId.put(word, currentId);
                idToToken.put(currentId, word);
                currentId++;
            }
        }
        
        // Fill remaining slots with placeholder tokens
        for (int i = currentId; i < vocabularySize; i++) {
            String token = "token_" + i;
            tokenToId.put(token, i);
            idToToken.put(i, token);
        }
    }
    
    /**
     * Convert token IDs to embeddings
     * @param tokenIds Array of token IDs
     * @return Embedding matrix [sequence_length, embedding_dimension]
     */
    public Matrix embed(int[] tokenIds) {
        Matrix embeddings = new Matrix(tokenIds.length, embeddingDimension);
        
        for (int i = 0; i < tokenIds.length; i++) {
            int tokenId = tokenIds[i];
            if (tokenId >= vocabularySize || tokenId < 0) {
                tokenId = 1; // Use <unk> token for out-of-vocabulary
            }
            
            // Copy embedding vector for this token
            for (int j = 0; j < embeddingDimension; j++) {
                embeddings.set(i, j, embeddingMatrix.get(tokenId, j));
            }
        }
        
        return embeddings;
    }
    
    /**
     * Simple tokenization (in practice, would use a proper tokenizer like BPE)
     */
    public int[] tokenize(String text) {
        String[] words = text.toLowerCase().split("\\s+");
        int[] tokenIds = new int[words.length];
        
        for (int i = 0; i < words.length; i++) {
            tokenIds[i] = tokenToId.getOrDefault(words[i], 1); // Default to <unk>
        }
        
        return tokenIds;
    }
    
    /**
     * Convert token IDs back to text
     */
    public String detokenize(int[] tokenIds) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < tokenIds.length; i++) {
            String token = idToToken.getOrDefault(tokenIds[i], "<unk>");
            if (!token.startsWith("<") || !token.endsWith(">")) { // Skip special tokens
                if (text.length() > 0) {
                    text.append(" ");
                }
                text.append(token);
            }
        }
        return text.toString();
    }
    
    /**
     * Get token ID for a given token
     */
    public int getTokenId(String token) {
        return tokenToId.getOrDefault(token, 1); // Default to <unk>
    }
    
    /**
     * Get token for a given ID
     */
    public String getToken(int tokenId) {
        return idToToken.getOrDefault(tokenId, "<unk>");
    }
    
    public int getVocabularySize() {
        return vocabularySize;
    }
    
    public int getEmbeddingDimension() {
        return embeddingDimension;
    }
}
