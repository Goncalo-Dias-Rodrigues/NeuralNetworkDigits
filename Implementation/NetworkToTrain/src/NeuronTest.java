import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for Neuron
 * @version 1.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
class NeuronTest {

    /**
     * Tests the predict method
     */
    @Test
    void testPredict() {
        Neuron p = new Neuron(new double[] {1.0, 1.0}, -1.5);

        double result = p.predict(new double[] {0, 0});
        assertTrue(Math.abs(result - 0.1824255238) < 10e-9);

        result = p.predict(new double[] {0, 1});
        assertTrue(Math.abs(result - 0.3775406688) < 10e-9);

        result = p.predict(new double[] {1, 0});
        assertTrue(Math.abs(result - 0.3775406688) < 10e-9);

        result = p.predict(new double[] {1, 1});
        assertTrue(Math.abs(result - 0.6224593312) < 10e-9);
    }
}
