import java.util.Arrays;
import java.io.*;

/**
 * Class that represents the neural network with two neurons
 * @version 2.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
public class NeuralNetwork {

    private Neuron h, o;


    /**
     * Neural Network Constructor
     * @param p1 Input perceptron
     * @param p2 Output perceptron
     */
    public NeuralNetwork(Neuron p1, Neuron p2) {
        this.h = p1;
        this.o = p2;
    }


    /**
     * Provides the prediction of the neural network according to the input
     * @param inputs input values
     * @return network prediction
     */
    public double predict(double[] inputs) {
        double y1 = h.predict(inputs); // output of hidden neuron is third input to second neuron

        double[] finalInputs = Arrays.copyOf(inputs, inputs.length + 1);
        finalInputs[finalInputs.length - 1] = y1;

        return o.predict(finalInputs); // output of output neuron
    }

}



















