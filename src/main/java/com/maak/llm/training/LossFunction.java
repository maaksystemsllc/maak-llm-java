package com.maak.llm.training;

import com.maak.llm.math.Matrix;
import org.springframework.stereotype.Component;

/**
 * Loss functions for training the transformer model
 */
@Component
public class LossFunction {
    
    /**
     * Cross-entropy loss for language modeling
     * @param predictions Model predictions (batch_size x seq_length x vocab_size)
     * @param targets Target token indices (batch_size x seq_length)
     * @return Loss value and gradients
     */
    public LossResult crossEntropyLoss(Matrix predictions, int[][] targets) {
        int batchSize = predictions.getRows();
        int seqLength = predictions.getCols() / targets[0].length;
        int vocabSize = predictions.getCols() / seqLength;
        
        double totalLoss = 0.0;
        Matrix gradients = new Matrix(predictions.getRows(), predictions.getCols());
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < targets[b].length; t++) {
                int targetToken = targets[b][t];
                
                // Get predictions for this position
                double[] logits = new double[vocabSize];
                for (int v = 0; v < vocabSize; v++) {
                    logits[v] = predictions.get(b, t * vocabSize + v);
                }
                
                // Apply softmax
                double[] probs = softmax(logits);
                
                // Calculate loss (negative log likelihood)
                totalLoss -= Math.log(Math.max(probs[targetToken], 1e-10));
                
                // Calculate gradients
                for (int v = 0; v < vocabSize; v++) {
                    double grad = probs[v];
                    if (v == targetToken) {
                        grad -= 1.0;
                    }
                    gradients.set(b, t * vocabSize + v, grad);
                }
            }
        }
        
        // Average loss over batch and sequence length
        double avgLoss = totalLoss / (batchSize * targets[0].length);
        
        return new LossResult(avgLoss, gradients);
    }
    
    /**
     * Softmax activation function
     */
    private double[] softmax(double[] logits) {
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            maxLogit = Math.max(maxLogit, logit);
        }
        
        double sum = 0.0;
        double[] probs = new double[logits.length];
        
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        
        return probs;
    }
    
    /**
     * Result of loss calculation
     */
    public static class LossResult {
        private final double loss;
        private final Matrix gradients;
        
        public LossResult(double loss, Matrix gradients) {
            this.loss = loss;
            this.gradients = gradients;
        }
        
        public double getLoss() {
            return loss;
        }
        
        public Matrix getGradients() {
            return gradients;
        }
    }
}
