package com.maak.llm;

import com.maak.llm.config.TransformerConfig;
import com.maak.llm.embedding.PositionalEncoding;
import com.maak.llm.embedding.TokenEmbedding;
import com.maak.llm.math.Matrix;
import com.maak.llm.model.TransformerModel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class TransformerModelTest {

    private TransformerModel model;
    private TransformerConfig config;

    @BeforeEach
    void setUp() {
        config = new TransformerConfig();
        config.setVocabularySize(1000);
        config.setMaxSequenceLength(128);
        config.setEmbeddingDimension(256);
        config.setNumLayers(4);
        config.setNumHeads(8);
        config.setFeedForwardDimension(1024);
        config.setDropoutRate(0.1);
        config.setLayerNormEpsilon(1e-5);
        config.setActivationFunction("gelu");
        
        // Create dependencies manually for testing
        TokenEmbedding tokenEmbedding = new TokenEmbedding(config.getVocabularySize(), config.getEmbeddingDimension());
        PositionalEncoding positionalEncoding = new PositionalEncoding(config.getMaxSequenceLength(), config.getEmbeddingDimension());
        
        model = new TransformerModel(config, tokenEmbedding, positionalEncoding);
    }

    @Test
    void testModelInitialization() {
        assertNotNull(model);
        assertNotNull(model.getConfig());
        assertNotNull(model.getTokenEmbedding());
        assertEquals(config.getVocabularySize(), model.getConfig().getVocabularySize());
        assertEquals(config.getEmbeddingDimension(), model.getConfig().getEmbeddingDimension());
    }

    @Test
    void testForwardPass() {
        int[] inputTokens = {1, 2, 3, 4, 5};
        Matrix output = model.forward(inputTokens);
        
        assertNotNull(output);
        assertEquals(inputTokens.length, output.getRows());
        assertEquals(config.getVocabularySize(), output.getCols());
        
        // Check that output values are finite
        for (int i = 0; i < output.getRows(); i++) {
            for (int j = 0; j < output.getCols(); j++) {
                assertTrue(Double.isFinite(output.get(i, j)));
            }
        }
    }

    @Test
    void testTextGeneration() {
        String prompt = "hello world";
        String generated = model.generate(prompt, 10, 1.0);
        
        assertNotNull(generated);
        assertFalse(generated.trim().isEmpty());
        assertTrue(generated.contains(prompt) || generated.length() > prompt.length());
    }

    @Test
    void testPerplexityCalculation() {
        String text = "the quick brown fox";
        double perplexity = model.calculatePerplexity(text);
        
        assertTrue(perplexity > 0);
        assertTrue(Double.isFinite(perplexity));
    }

    @Test
    void testTokenization() {
        String text = "hello world test";
        int[] tokens = model.getTokenEmbedding().tokenize(text);
        
        assertNotNull(tokens);
        assertTrue(tokens.length > 0);
        
        String detokenized = model.getTokenEmbedding().detokenize(tokens);
        assertNotNull(detokenized);
    }

    @Test
    void testEmptyInput() {
        int[] emptyTokens = {};
        assertThrows(Exception.class, () -> model.forward(emptyTokens));
    }

    @Test
    void testLongSequence() {
        int[] longTokens = new int[config.getMaxSequenceLength() + 10];
        for (int i = 0; i < longTokens.length; i++) {
            longTokens[i] = i % config.getVocabularySize();
        }
        
        // Should handle sequences longer than max length gracefully
        assertDoesNotThrow(() -> model.forward(longTokens));
    }
}
