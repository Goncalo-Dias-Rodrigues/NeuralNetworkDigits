/**
 * Class that represents a neuron
 * @version 2.1
 * @author by Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
public class Neuron {

    private double[] weights;
    private double bias;

    /**
     * Neuron constructor
     * @param weights weights
     * @param bias bias value
     */
    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    /**
     * Getter for weights
     * @return weights
     */
    public double[] getWeights() {
        return weights;
    }

    /**
     * Setter for weights
     * @param weights New weights
     */
    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    /**
     * Setter for bias
     * @param bias New bias
     */
    public void setBias(double bias) {
        this.bias = bias;
    }

    /**
     * Getter for bias
     * @return bias
     */
    public double getBias() {
        return bias;
    }

    /**
     * Calculates the neuron's prediction. It computes the sum of the input-weight products,
     * adds the bias, and passes the result through a sigmoid function.
     * @param inputs input values
     * @return neuron prediction
     */
    public double predict(double[] inputs) {
        double sum = bias; // initialize sum with bias (same as initializing at 0 and then adding bias)

        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i]; // sum w*x
        }

        return sigmoid(sum); // apply sigmoid
    }

    /**
     * Calculates the sigmoid function of a value
     * @param z input to the function
     * @return value between 0 and 1
     */
    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }
}
