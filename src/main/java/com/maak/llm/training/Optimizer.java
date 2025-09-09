package com.maak.llm.training;

import com.maak.llm.math.Matrix;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * Adam optimizer for training neural networks
 */
@Component
public class Optimizer {
    
    private final Map<String, Matrix> momentum = new HashMap<>();
    private final Map<String, Matrix> velocity = new HashMap<>();
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int step = 0;
    
    /**
     * Update parameters using Adam optimization
     */
    public void updateParameters(Map<String, Matrix> parameters, Map<String, Matrix> gradients, double learningRate) {
        step++;
        
        for (Map.Entry<String, Matrix> entry : parameters.entrySet()) {
            String paramName = entry.getKey();
            Matrix param = entry.getValue();
            Matrix grad = gradients.get(paramName);
            
            if (grad == null) continue;
            
            // Initialize momentum and velocity if not exists
            if (!momentum.containsKey(paramName)) {
                momentum.put(paramName, new Matrix(param.getRows(), param.getCols()));
                velocity.put(paramName, new Matrix(param.getRows(), param.getCols()));
            }
            
            Matrix m = momentum.get(paramName);
            Matrix v = velocity.get(paramName);
            
            // Update biased first moment estimate
            for (int i = 0; i < param.getRows(); i++) {
                for (int j = 0; j < param.getCols(); j++) {
                    double gradValue = grad.get(i, j);
                    double mValue = beta1 * m.get(i, j) + (1 - beta1) * gradValue;
                    double vValue = beta2 * v.get(i, j) + (1 - beta2) * gradValue * gradValue;
                    
                    m.set(i, j, mValue);
                    v.set(i, j, vValue);
                    
                    // Bias correction
                    double mHat = mValue / (1 - Math.pow(beta1, step));
                    double vHat = vValue / (1 - Math.pow(beta2, step));
                    
                    // Update parameter
                    double paramValue = param.get(i, j);
                    double update = learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                    param.set(i, j, paramValue - update);
                }
            }
        }
    }
    
    /**
     * Apply gradient clipping to prevent exploding gradients
     */
    public void clipGradients(Map<String, Matrix> gradients, double maxNorm) {
        double totalNorm = 0.0;
        
        // Calculate total gradient norm
        for (Matrix grad : gradients.values()) {
            for (int i = 0; i < grad.getRows(); i++) {
                for (int j = 0; j < grad.getCols(); j++) {
                    double value = grad.get(i, j);
                    totalNorm += value * value;
                }
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // Clip if necessary
        if (totalNorm > maxNorm) {
            double clipRatio = maxNorm / totalNorm;
            for (Matrix grad : gradients.values()) {
                for (int i = 0; i < grad.getRows(); i++) {
                    for (int j = 0; j < grad.getCols(); j++) {
                        grad.set(i, j, grad.get(i, j) * clipRatio);
                    }
                }
            }
        }
    }
    
    /**
     * Reset optimizer state
     */
    public void reset() {
        momentum.clear();
        velocity.clear();
        step = 0;
    }
    
    /**
     * Get current step count
     */
    public int getStep() {
        return step;
    }
}
