import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Classe de teste para a classe NeuralNetwork
 * @version 1.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
class NeuralNetworkTest {

    /**
     * Testa o metodo predict
     */
    @Test
    void testPredict() {

        Neuron h = new Neuron(new double[] {1.0,1.0},-1.5);
        Neuron o = new Neuron(new double[] {1.0,1.0,-2.0},-0.5);

        NeuralNetwork nn = new NeuralNetwork(h,o);

        double obtido = nn.predict(new double[] {0,0});

        assertTrue(Math.abs(obtido - 0.2963268202) < 10e-9);

        obtido = nn.predict(new double[] {0,1});

        assertTrue(Math.abs(obtido - 0.4365732065) < 10e-9);

        obtido = nn.predict(new double[] {1,0});

        assertTrue(Math.abs(obtido - 0.4365732065) < 10e-9);

        obtido = nn.predict(new double[] {1,1});

        assertTrue(Math.abs(obtido - 0.5634267935) < 10e-9);
    }
}






