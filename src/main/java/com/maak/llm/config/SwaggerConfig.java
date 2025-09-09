package com.maak.llm.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * Swagger/OpenAPI configuration for LLM Transformer API
 */
@Configuration
public class SwaggerConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("LLM Transformer API")
                        .version("1.0.0")
                        .description("A comprehensive Large Language Model (LLM) Transformer implementation with REST API endpoints for text generation, tokenization, and model operations.")
                        .contact(new Contact()
                                .name("MAAK LLM Team")
                                .email("support@maak-llm.com")
                                .url("https://github.com/maak/llm-transformer"))
                        .license(new License()
                                .name("MIT License")
                                .url("https://opensource.org/licenses/MIT")))
                .servers(List.of(
                        new Server()
                                .url("http://localhost:8083")
                                .description("Development Server"),
                        new Server()
                                .url("https://api.maak-llm.com")
                                .description("Production Server")
                ));
    }
}
