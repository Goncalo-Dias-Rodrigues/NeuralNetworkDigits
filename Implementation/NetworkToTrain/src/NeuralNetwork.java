import java.util.Arrays;
import java.io.*;

/**
 * Class that represents the neural network with two neurons
 * @version 3.0
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
     * Calculates the accuracy of the network for a given set of inputs
     * @param testInputs test input set
     * @param testOutputs expected outputs for test
     * @return network accuracy
     */
    public double precision(double[][] testInputs, double[] testOutputs) {
        double correct = 0;

        for (int i = 0; i < testInputs.length; i++) {
            double output = Math.round(this.predict(testInputs[i]));
            if (Math.abs(output - testOutputs[i]) < 10e-9)
                correct++;
        }

        return correct / testInputs.length;
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

    /**
     * Trains the neural network using the exact backpropagation algorithm
     * @param trainInputs training input set
     * @param trainOutputs expected training outputs
     * @param testInputs test input set
     * @param testOutputs expected test outputs
     * @throws IOException file error
     */
    public void train(double[][] trainInputs, double[] trainOutputs, double[][] testInputs, double[] testOutputs) throws IOException {

        double learningRate = 0.3;

        int epoch = 0;
        double mseTrain;

        double mseTest = 0;
        double minMseTest = Double.MAX_VALUE;
        double previousMseTest = Double.MAX_VALUE;

        double[] bestWeightsH = new double[h.getWeights().length];
        double[] bestWeightsO = new double[o.getWeights().length];

        double bestBiasH = 0;
        double bestBiasO = 0;

        int tolerance = 0;

        BufferedWriter writer = new BufferedWriter(new FileWriter("results.csv"));

        while (true) {
            epoch++;
            double totalError = 0;

            for (int i = 0; i < trainInputs.length; i++) {

                double[] input = trainInputs[i];
                double target = trainOutputs[i];

                double output = this.predict(input);

                totalError += calculateError(output, target);

                double deltaO = (output - target) * dSigmoid(output);

                double outputH = h.predict(input);

                double deltaH = deltaO * o.getWeights()[400] * dSigmoid(outputH);

                for (int j = 0; j < h.getWeights().length; j++) {
                    h.getWeights()[j] -= learningRate * deltaH * input[j];
                }
                h.setBias(h.getBias() - learningRate * deltaH);

                for (int k = 0; k < o.getWeights().length; k++) {
                    if (k == 400) {
                        o.getWeights()[400] -= learningRate * deltaO * outputH;
                    } else {
                        o.getWeights()[k] -= learningRate * deltaO * input[k];
                    }
                }
                o.setBias(o.getBias() - learningRate * deltaO);
            }

            mseTrain = totalError / trainInputs.length;

            double totalTestError = 0;
            for (int i = 0; i < testInputs.length; i++) {
                double[] input = testInputs[i];
                double target = testOutputs[i];
                double output = this.predict(input);
                totalTestError += calculateError(output, target);
            }

            mseTest = totalTestError / testInputs.length;

            if (mseTest > previousMseTest) {
                tolerance++;
            } else {
                tolerance = 0;

                if (mseTest < minMseTest) {
                    System.arraycopy(h.getWeights(), 0, bestWeightsH, 0, h.getWeights().length);
                    System.arraycopy(o.getWeights(), 0, bestWeightsO, 0, o.getWeights().length);
                    bestBiasH = h.getBias();
                    bestBiasO = o.getBias();
                    minMseTest = mseTest;
                }
            }

            previousMseTest = mseTest;

            if (epoch % 100 == 0) {
                System.out.println("Epoch: " + epoch + " | MSE (train): " + mseTrain + " | MSE (test): " + mseTest);
            }

            writer.write(epoch + ";" + mseTrain + ";" + mseTest);
            writer.newLine();
            writer.flush();

            if (mseTrain <= 0.00001 || tolerance == 10) {

                h.setWeights(bestWeightsH);
                o.setWeights(bestWeightsO);
                h.setBias(bestBiasH);
                o.setBias(bestBiasO);

                System.out.print("Training completed at epoch ");
                if (tolerance == 10)
                    System.out.println(epoch - 10 + " (tolerance)");
                else
                    System.out.println(epoch);

                System.out.println("MSE (train): " + mseTrain + ", MSE (test): " + minMseTest);
                break;
            }
        }
    }

    /**
     * Calculates the mean squared error for a single input
     * @param output network output
     * @param target expected output
     * @return MSE
     */
    private double calculateError(double output, double target) {
        return 0.5 * Math.pow(output - target, 2);
    }

    /**
     * Derivative of the sigmoid function
     * @param x input to the function
     * @return sig'(x)
     */
    private double dSigmoid(double x) {
        return x * (1 - x);
    }
}
