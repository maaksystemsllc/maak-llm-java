package com.maak.llm.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {

    @Test
    void testMatrixCreation() {
        Matrix matrix = new Matrix(3, 4);
        assertEquals(3, matrix.getRows());
        assertEquals(4, matrix.getCols());
    }

    @Test
    void testMatrixMultiplication() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        
        Matrix result = a.multiply(b);
        
        assertEquals(2, result.getRows());
        assertEquals(2, result.getCols());
        assertEquals(19, result.get(0, 0), 1e-10);
        assertEquals(22, result.get(0, 1), 1e-10);
        assertEquals(43, result.get(1, 0), 1e-10);
        assertEquals(50, result.get(1, 1), 1e-10);
    }

    @Test
    void testMatrixAddition() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        
        Matrix result = a.add(b);
        
        assertEquals(6, result.get(0, 0), 1e-10);
        assertEquals(8, result.get(0, 1), 1e-10);
        assertEquals(10, result.get(1, 0), 1e-10);
        assertEquals(12, result.get(1, 1), 1e-10);
    }

    @Test
    void testMatrixTranspose() {
        Matrix matrix = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix transposed = matrix.transpose();
        
        assertEquals(3, transposed.getRows());
        assertEquals(2, transposed.getCols());
        assertEquals(1, transposed.get(0, 0), 1e-10);
        assertEquals(4, transposed.get(0, 1), 1e-10);
        assertEquals(2, transposed.get(1, 0), 1e-10);
        assertEquals(5, transposed.get(1, 1), 1e-10);
    }

    @Test
    void testSoftmax() {
        Matrix matrix = new Matrix(new double[][]{{1, 2, 3}});
        Matrix softmax = matrix.softmax();
        
        // Check that probabilities sum to 1
        double sum = 0;
        for (int j = 0; j < softmax.getCols(); j++) {
            sum += softmax.get(0, j);
        }
        assertEquals(1.0, sum, 1e-10);
        
        // Check that all values are positive
        for (int j = 0; j < softmax.getCols(); j++) {
            assertTrue(softmax.get(0, j) > 0);
        }
    }

    @Test
    void testLayerNorm() {
        Matrix matrix = new Matrix(new double[][]{{1, 2, 3, 4}});
        Matrix normalized = matrix.layerNorm(1e-5);
        
        // Check that mean is approximately 0
        double mean = 0;
        for (int j = 0; j < normalized.getCols(); j++) {
            mean += normalized.get(0, j);
        }
        mean /= normalized.getCols();
        assertEquals(0.0, mean, 1e-10);
        
        // Check that variance is approximately 1
        double variance = 0;
        for (int j = 0; j < normalized.getCols(); j++) {
            variance += Math.pow(normalized.get(0, j) - mean, 2);
        }
        variance /= normalized.getCols();
        assertEquals(1.0, variance, 1e-5);
    }

    @Test
    void testGelu() {
        Matrix matrix = new Matrix(new double[][]{{-1, 0, 1}});
        Matrix gelu = matrix.gelu();
        
        // GELU(0) should be approximately 0
        assertEquals(0.0, gelu.get(0, 1), 1e-5);
        
        // GELU should be monotonic
        assertTrue(gelu.get(0, 0) < gelu.get(0, 1));
        assertTrue(gelu.get(0, 1) < gelu.get(0, 2));
    }

    @Test
    void testRelu() {
        Matrix matrix = new Matrix(new double[][]{{-1, 0, 1}});
        Matrix relu = matrix.relu();
        
        assertEquals(0.0, relu.get(0, 0), 1e-10);
        assertEquals(0.0, relu.get(0, 1), 1e-10);
        assertEquals(1.0, relu.get(0, 2), 1e-10);
    }

    @Test
    void testRandomMatrix() {
        Matrix random = Matrix.random(5, 5, 1.0);
        assertEquals(5, random.getRows());
        assertEquals(5, random.getCols());
        
        // Check that values are not all the same (very unlikely with random)
        boolean allSame = true;
        double firstValue = random.get(0, 0);
        for (int i = 0; i < random.getRows(); i++) {
            for (int j = 0; j < random.getCols(); j++) {
                if (Math.abs(random.get(i, j) - firstValue) > 1e-10) {
                    allSame = false;
                    break;
                }
            }
            if (!allSame) break;
        }
        assertFalse(allSame);
    }

    @Test
    void testInvalidOperations() {
        Matrix a = new Matrix(2, 3);
        Matrix b = new Matrix(4, 5);
        
        assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }
}
