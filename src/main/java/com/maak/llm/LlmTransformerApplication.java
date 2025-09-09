package com.maak.llm;

import com.maak.llm.config.TransformerConfig;
import com.maak.llm.embedding.PositionalEncoding;
import com.maak.llm.embedding.TokenEmbedding;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

@SpringBootApplication
@EnableAsync
public class LlmTransformerApplication {

    @Autowired
    private TransformerConfig transformerConfig;

    public static void main(String[] args) {
        SpringApplication.run(LlmTransformerApplication.class, args);
    }

    @Bean(name = "taskExecutor")
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(4);
        executor.setMaxPoolSize(8);
        executor.setQueueCapacity(500);
        executor.setThreadNamePrefix("LLM-");
        executor.initialize();
        return executor;
    }

    @Bean
    public TokenEmbedding tokenEmbedding() {
        return new TokenEmbedding(
            transformerConfig.getVocabularySize(),
            transformerConfig.getEmbeddingDimension()
        );
    }

    @Bean
    public PositionalEncoding positionalEncoding() {
        return new PositionalEncoding(
            transformerConfig.getMaxSequenceLength(),
            transformerConfig.getEmbeddingDimension()
        );
    }
}
