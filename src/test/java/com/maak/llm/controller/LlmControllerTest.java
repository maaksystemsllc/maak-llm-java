package com.maak.llm.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.maak.llm.dto.GenerationRequest;
import com.maak.llm.dto.GenerationResponse;
import com.maak.llm.service.LlmService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(LlmController.class)
class LlmControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private LlmService llmService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void testGenerateText() throws Exception {
        GenerationRequest request = new GenerationRequest("Hello", 50, 1.0);
        GenerationResponse response = new GenerationResponse("Hello world", "Hello");
        response.setTokensGenerated(2);
        response.setProcessingTimeMs(100);

        when(llmService.generateText(any(GenerationRequest.class))).thenReturn(response);

        mockMvc.perform(post("/api/llm/generate")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.generatedText").value("Hello world"))
                .andExpect(jsonPath("$.originalPrompt").value("Hello"))
                .andExpect(jsonPath("$.tokensGenerated").value(2));
    }

    @Test
    void testGenerateTextAsync() throws Exception {
        GenerationRequest request = new GenerationRequest("Hello", 50, 1.0);
        GenerationResponse response = new GenerationResponse("Hello world", "Hello");
        
        when(llmService.generateTextAsync(any(GenerationRequest.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        mockMvc.perform(post("/api/llm/generate/async")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk());
    }

    @Test
    void testCalculatePerplexity() throws Exception {
        when(llmService.calculatePerplexity("test text")).thenReturn(15.5);

        mockMvc.perform(post("/api/llm/perplexity")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"text\":\"test text\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.perplexity").value(15.5))
                .andExpect(jsonPath("$.text").value("test text"));
    }

    @Test
    void testTokenize() throws Exception {
        when(llmService.tokenize("hello world")).thenReturn(new int[]{1, 2});

        mockMvc.perform(post("/api/llm/tokenize")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"text\":\"hello world\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.tokens[0]").value(1))
                .andExpect(jsonPath("$.tokens[1]").value(2))
                .andExpect(jsonPath("$.token_count").value(2));
    }

    @Test
    void testDetokenize() throws Exception {
        when(llmService.detokenize(new int[]{1, 2})).thenReturn("hello world");

        mockMvc.perform(post("/api/llm/detokenize")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"tokens\":[1,2]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.text").value("hello world"))
                .andExpect(jsonPath("$.token_count").value(2));
    }

    @Test
    void testGetModelInfo() throws Exception {
        LlmService.ModelInfo info = new LlmService.ModelInfo(1000, 256, 4, 8, 128, "v1.0");
        when(llmService.getModelInfo()).thenReturn(info);

        mockMvc.perform(get("/api/llm/info"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.vocabularySize").value(1000))
                .andExpect(jsonPath("$.embeddingDimension").value(256))
                .andExpect(jsonPath("$.numLayers").value(4));
    }

    @Test
    void testHealthCheck() throws Exception {
        mockMvc.perform(get("/api/llm/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("healthy"))
                .andExpect(jsonPath("$.service").value("LLM Transformer"));
    }

    @Test
    void testComplete() throws Exception {
        GenerationResponse response = new GenerationResponse("Hello world", "Hello");
        when(llmService.generateText(any(GenerationRequest.class))).thenReturn(response);

        mockMvc.perform(get("/api/llm/complete")
                .param("prompt", "Hello")
                .param("maxLength", "50")
                .param("temperature", "1.0"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prompt").value("Hello"))
                .andExpect(jsonPath("$.completion").value("Hello world"));
    }

    @Test
    void testInvalidRequest() throws Exception {
        GenerationRequest invalidRequest = new GenerationRequest("", -1, -1.0);

        mockMvc.perform(post("/api/llm/generate")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(invalidRequest)))
                .andExpect(status().isBadRequest());
    }
}
