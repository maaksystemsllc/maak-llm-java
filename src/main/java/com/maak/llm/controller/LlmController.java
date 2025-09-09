package com.maak.llm.controller;

import com.maak.llm.dto.GenerationRequest;
import com.maak.llm.dto.GenerationResponse;
import com.maak.llm.service.LlmService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * REST Controller for LLM operations
 */
@Tag(name = "LLM Transformer", description = "Large Language Model Transformer API for text generation and processing")
@RestController
@RequestMapping("/api/llm")
@CrossOrigin(origins = "*")
public class LlmController {
    
    private static final Logger logger = LoggerFactory.getLogger(LlmController.class);
    
    private final LlmService llmService;
    
    @Autowired
    public LlmController(LlmService llmService) {
        this.llmService = llmService;
    }
    
    /**
     * Generate text synchronously
     */
    @Operation(
        summary = "Generate text synchronously",
        description = "Generate text using the LLM Transformer model with specified parameters. This endpoint processes the request synchronously and returns the generated text along with metadata.",
        requestBody = @io.swagger.v3.oas.annotations.parameters.RequestBody(
            description = "Text generation parameters",
            content = @Content(
                mediaType = "application/json",
                schema = @Schema(implementation = GenerationRequest.class),
                examples = @ExampleObject(
                    name = "Basic Generation",
                    value = "{\n  \"prompt\": \"The future of artificial intelligence\",\n  \"maxLength\": 100,\n  \"temperature\": 1.0,\n  \"topP\": 0.9\n}"
                )
            )
        )
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Text generated successfully",
            content = @Content(
                mediaType = "application/json",
                schema = @Schema(implementation = GenerationResponse.class)
            )
        ),
        @ApiResponse(
            responseCode = "400",
            description = "Invalid request parameters",
            content = @Content(mediaType = "application/json")
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Internal server error during text generation",
            content = @Content(mediaType = "application/json")
        )
    })
    @PostMapping("/generate")
    public ResponseEntity<GenerationResponse> generateText(@Valid @RequestBody GenerationRequest request) {
        try {
            logger.info("Received generation request for prompt length: {}", request.getPrompt().length());
            GenerationResponse response = llmService.generateText(request);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error in text generation", e);
            return ResponseEntity.internalServerError().build();
        }
    }
    
    /**
     * Generate text asynchronously
     */
    @Operation(
        summary = "Generate text asynchronously",
        description = "Generate text using the LLM Transformer model asynchronously. This endpoint returns immediately and processes the request in the background."
    )
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Async text generation started successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid request parameters"),
        @ApiResponse(responseCode = "500", description = "Internal server error")
    })
    @PostMapping("/generate/async")
    public CompletableFuture<ResponseEntity<GenerationResponse>> generateTextAsync(@Valid @RequestBody GenerationRequest request) {
        logger.info("Received async generation request for prompt length: {}", request.getPrompt().length());
        
        return llmService.generateTextAsync(request)
            .thenApply(ResponseEntity::ok)
            .exceptionally(throwable -> {
                logger.error("Error in async text generation", throwable);
                return ResponseEntity.internalServerError().build();
            });
    }
    
    /**
     * Calculate perplexity for given text
     */
    @PostMapping("/perplexity")
    public ResponseEntity<Map<String, Object>> calculatePerplexity(@RequestBody Map<String, String> request) {
        try {
            String text = request.get("text");
            if (text == null || text.trim().isEmpty()) {
                return ResponseEntity.badRequest().build();
            }
            
            double perplexity = llmService.calculatePerplexity(text);
            
            Map<String, Object> response = new HashMap<>();
            response.put("text", text);
            response.put("perplexity", perplexity);
            response.put("length", text.length());
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error calculating perplexity", e);
            return ResponseEntity.internalServerError().build();
        }
    }
    
    /**
     * Tokenize text
     */
    @PostMapping("/tokenize")
    public ResponseEntity<Map<String, Object>> tokenize(@RequestBody Map<String, String> request) {
        try {
            String text = request.get("text");
            if (text == null) {
                return ResponseEntity.badRequest().build();
            }
            
            int[] tokens = llmService.tokenize(text);
            
            Map<String, Object> response = new HashMap<>();
            response.put("text", text);
            response.put("tokens", tokens);
            response.put("token_count", tokens.length);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error tokenizing text", e);
            return ResponseEntity.internalServerError().build();
        }
    }
    
    /**
     * Detokenize token IDs back to text
     */
    @PostMapping("/detokenize")
    public ResponseEntity<Map<String, Object>> detokenize(@RequestBody Map<String, int[]> request) {
        try {
            int[] tokens = request.get("tokens");
            if (tokens == null) {
                return ResponseEntity.badRequest().build();
            }
            
            String text = llmService.detokenize(tokens);
            
            Map<String, Object> response = new HashMap<>();
            response.put("tokens", tokens);
            response.put("text", text);
            response.put("token_count", tokens.length);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            logger.error("Error detokenizing tokens", e);
            return ResponseEntity.internalServerError().build();
        }
    }
    
    /**
     * Get model information
     */
    @Operation(
        summary = "Get model information",
        description = "Retrieve detailed information about the LLM Transformer model configuration and capabilities."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Model information retrieved successfully",
            content = @Content(
                mediaType = "application/json",
                schema = @Schema(implementation = LlmService.ModelInfo.class)
            )
        ),
        @ApiResponse(responseCode = "500", description = "Internal server error")
    })
    @GetMapping("/info")
    public ResponseEntity<LlmService.ModelInfo> getModelInfo() {
        try {
            LlmService.ModelInfo info = llmService.getModelInfo();
            return ResponseEntity.ok(info);
        } catch (Exception e) {
            logger.error("Error getting model info", e);
            return ResponseEntity.internalServerError().build();
        }
    }
    
    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "healthy");
        response.put("service", "LLM Transformer");
        response.put("version", "1.0.0");
        return ResponseEntity.ok(response);
    }
    
    /**
     * Simple completion endpoint for quick testing
     */
    @Operation(
        summary = "Quick text completion",
        description = "Simple endpoint for quick text completion with URL parameters. Ideal for testing and simple integrations."
    )
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Text completion successful"),
        @ApiResponse(responseCode = "500", description = "Internal server error")
    })
    @GetMapping("/complete")
    public ResponseEntity<Map<String, String>> complete(
            @Parameter(description = "Input prompt for text completion", required = true)
            @RequestParam String prompt,
            @Parameter(description = "Maximum length of generated text", example = "50")
            @RequestParam(defaultValue = "50") int maxLength,
            @Parameter(description = "Sampling temperature (0.1-2.0)", example = "1.0")
            @RequestParam(defaultValue = "1.0") double temperature) {
        try {
            GenerationRequest request = new GenerationRequest(prompt, maxLength, temperature);
            GenerationResponse response = llmService.generateText(request);
            
            Map<String, String> result = new HashMap<>();
            result.put("prompt", prompt);
            result.put("completion", response.getGeneratedText());
            
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            logger.error("Error in completion", e);
            return ResponseEntity.internalServerError().build();
        }
    }
}
