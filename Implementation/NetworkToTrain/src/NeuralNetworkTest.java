import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for NeuralNetwork
 * @version 1.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
class NeuralNetworkTest {

    /**
     * Tests the predict method
     */
    @Test
    void testPredict() {

        Neuron h = new Neuron(new double[] {1.0, 1.0}, -1.5);
        Neuron o = new Neuron(new double[] {1.0, 1.0, -2.0}, -0.5);

        NeuralNetwork nn = new NeuralNetwork(h, o);

        double result = nn.predict(new double[] {0, 0});
        assertTrue(Math.abs(result - 0.2963268202) < 10e-9);

        result = nn.predict(new double[] {0, 1});
        assertTrue(Math.abs(result - 0.4365732065) < 10e-9);

        result = nn.predict(new double[] {1, 0});
        assertTrue(Math.abs(result - 0.4365732065) < 10e-9);

        result = nn.predict(new double[] {1, 1});
        assertTrue(Math.abs(result - 0.5634267935) < 10e-9);
    }
}
