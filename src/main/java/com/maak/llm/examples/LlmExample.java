package com.maak.llm.examples;

import com.maak.llm.dto.GenerationRequest;
import com.maak.llm.dto.GenerationResponse;
import com.maak.llm.service.LlmService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

/**
 * Example application demonstrating LLM Transformer usage
 */
@SpringBootApplication(scanBasePackages = "com.maak.llm")
public class LlmExample {

    public static void main(String[] args) {
        SpringApplication.run(LlmExample.class, args);
    }

    @Bean
    CommandLineRunner demo(LlmService llmService) {
        return args -> {
            System.out.println("=== LLM Transformer Example ===\n");

            // Example 1: Basic text generation
            System.out.println("1. Basic Text Generation:");
            GenerationRequest request = new GenerationRequest("The future of artificial intelligence", 50, 1.0);
            GenerationResponse response = llmService.generateText(request);
            System.out.println("Prompt: " + response.getOriginalPrompt());
            System.out.println("Generated: " + response.getGeneratedText());
            System.out.println("Tokens: " + response.getTokensGenerated());
            System.out.println("Time: " + response.getProcessingTimeMs() + "ms\n");

            // Example 2: Different temperature settings
            System.out.println("2. Temperature Comparison:");
            String prompt = "Once upon a time";
            
            // Low temperature (more deterministic)
            request = new GenerationRequest(prompt, 30, 0.3);
            response = llmService.generateText(request);
            System.out.println("Low temp (0.3): " + response.getGeneratedText());
            
            // High temperature (more creative)
            request = new GenerationRequest(prompt, 30, 1.5);
            response = llmService.generateText(request);
            System.out.println("High temp (1.5): " + response.getGeneratedText() + "\n");

            // Example 3: Tokenization
            System.out.println("3. Tokenization Example:");
            String text = "Hello world, this is a test sentence.";
            int[] tokens = llmService.tokenize(text);
            String detokenized = llmService.detokenize(tokens);
            System.out.println("Original: " + text);
            System.out.println("Tokens: " + java.util.Arrays.toString(tokens));
            System.out.println("Detokenized: " + detokenized + "\n");

            // Example 4: Perplexity calculation
            System.out.println("4. Perplexity Calculation:");
            String[] testTexts = {
                "The quick brown fox jumps over the lazy dog",
                "This is a normal sentence with common words",
                "Supercalifragilisticexpialidocious is a very unusual word"
            };
            
            for (String testText : testTexts) {
                try {
                    double perplexity = llmService.calculatePerplexity(testText);
                    System.out.println("Text: \"" + testText + "\"");
                    System.out.println("Perplexity: " + String.format("%.2f", perplexity) + "\n");
                } catch (Exception e) {
                    System.out.println("Could not calculate perplexity for: " + testText + "\n");
                }
            }

            // Example 5: Model information
            System.out.println("5. Model Information:");
            LlmService.ModelInfo info = llmService.getModelInfo();
            System.out.println("Vocabulary Size: " + info.getVocabularySize());
            System.out.println("Embedding Dimension: " + info.getEmbeddingDimension());
            System.out.println("Number of Layers: " + info.getNumLayers());
            System.out.println("Number of Heads: " + info.getNumHeads());
            System.out.println("Max Sequence Length: " + info.getMaxSequenceLength());
            System.out.println("Version: " + info.getVersion() + "\n");

            // Example 6: Batch processing simulation
            System.out.println("6. Batch Processing Simulation:");
            String[] prompts = {
                "The weather today is",
                "In the year 2030",
                "Machine learning will",
                "The best way to learn"
            };

            long startTime = System.currentTimeMillis();
            for (int i = 0; i < prompts.length; i++) {
                request = new GenerationRequest(prompts[i], 20, 1.0);
                response = llmService.generateText(request);
                System.out.println((i + 1) + ". " + prompts[i] + " -> " + response.getGeneratedText());
            }
            long totalTime = System.currentTimeMillis() - startTime;
            System.out.println("Total batch time: " + totalTime + "ms");
            System.out.println("Average per prompt: " + (totalTime / prompts.length) + "ms\n");

            System.out.println("=== Example Complete ===");
        };
    }
}
