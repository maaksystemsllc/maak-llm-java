package com.maak.llm.math;

import java.util.Arrays;
import java.util.Random;

/**
 * Matrix operations for neural network computations
 */
public class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;
    
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }
    
    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }
    
    public static Matrix zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }
    
    public static Matrix ones(int rows, int cols) {
        Matrix matrix = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            Arrays.fill(matrix.data[i], 1.0);
        }
        return matrix;
    }
    
    public static Matrix random(int rows, int cols, double scale) {
        Matrix matrix = new Matrix(rows, cols);
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.data[i][j] = (random.nextGaussian() * scale);
            }
        }
        return matrix;
    }
    
    public static Matrix identity(int size) {
        Matrix matrix = new Matrix(size, size);
        for (int i = 0; i < size; i++) {
            matrix.data[i][i] = 1.0;
        }
        return matrix;
    }
    
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }
    
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions don't match for addition");
        }
        
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix subtract(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions don't match for subtraction");
        }
        
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix scale(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }
    
    public Matrix softmax() {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            double max = Arrays.stream(this.data[i]).max().orElse(0.0);
            double sum = 0.0;
            
            // Compute exponentials with numerical stability
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = Math.exp(this.data[i][j] - max);
                sum += result.data[i][j];
            }
            
            // Normalize
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] /= sum;
            }
        }
        return result;
    }
    
    public Matrix layerNorm(double epsilon) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            // Calculate mean
            double mean = Arrays.stream(this.data[i]).average().orElse(0.0);
            
            // Calculate variance
            double variance = Arrays.stream(this.data[i])
                .map(x -> Math.pow(x - mean, 2))
                .average().orElse(0.0);
            
            // Normalize
            double std = Math.sqrt(variance + epsilon);
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = (this.data[i][j] - mean) / std;
            }
        }
        return result;
    }
    
    public Matrix gelu() {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                double x = this.data[i][j];
                result.data[i][j] = 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
            }
        }
        return result;
    }
    
    public Matrix relu() {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = Math.max(0.0, this.data[i][j]);
            }
        }
        return result;
    }
    
    // Getters
    public double get(int row, int col) {
        return data[row][col];
    }
    
    public void set(int row, int col, double value) {
        data[row][col] = value;
    }
    
    public int getRows() {
        return rows;
    }
    
    public int getCols() {
        return cols;
    }
    
    public double[][] getData() {
        return data;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Matrix[").append(rows).append("x").append(cols).append("]:\n");
        for (int i = 0; i < Math.min(rows, 5); i++) {
            sb.append("[");
            for (int j = 0; j < Math.min(cols, 5); j++) {
                sb.append(String.format("%8.4f", data[i][j]));
                if (j < Math.min(cols, 5) - 1) sb.append(", ");
            }
            if (cols > 5) sb.append(", ...");
            sb.append("]\n");
        }
        if (rows > 5) sb.append("...\n");
        return sb.toString();
    }
}
