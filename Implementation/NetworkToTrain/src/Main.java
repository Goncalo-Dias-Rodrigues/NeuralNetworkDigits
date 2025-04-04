import java.io.*;
import java.io.IOException;
import java.util.Arrays;

/**
 * Class responsible for user interaction
 * @version 1.0
 * @author Daniel Goncalves, Diogo Damasio, Goncalo Rodrigues
 */
public class Main {
    /**
     * Main program flow
     * @param args Command line arguments
     * @throws IOException file error
     */
    public static void main(String[] args) throws IOException {

        double[][] trainingInputs = null;
        double[] trainingOutputs = null;
        double[][] testInputs = null;
        double[] testOutputs = null;

        try {
            trainingInputs = loadInputs("src/dataset.csv", 640,"train"); // CSV file with image pixels
            trainingOutputs = loadOutputs("src/labels.csv",640,"train"); // CSV file with labels (0 or 1)
            testInputs = loadInputs("src/dataset.csv", 160,"test");
            testOutputs = loadOutputs("src/labels.csv", 160, "test");
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Applies data augmentation to training set
        double[][] augmentedTrainingInputs = augmentData(trainingInputs);
        double[] augmentedTrainingOutputs = new double[trainingOutputs.length * 2];

        // Duplicates the outputs
        System.arraycopy(trainingOutputs, 0, augmentedTrainingOutputs, 0, trainingOutputs.length);
        System.arraycopy(trainingOutputs, 0, augmentedTrainingOutputs, trainingOutputs.length, trainingOutputs.length);

        // Use augmented data for training
        trainingInputs = augmentedTrainingInputs;
        trainingOutputs = augmentedTrainingOutputs;

        double[] weights1 = new double[400];

        // Initialize weights and biases between 0 and 0.01
        for(int i = 0; i < 400; i++) {
            weights1[i] = Math.random() * 0.01;
        }
        double bias1 = Math.random() * 0.01;

        double[] weights2 = new double[401];

        for(int i = 0; i < 401; i++) {
            weights2[i] = Math.random() * 0.01;
        }
        double bias2 = Math.random() * 0.01;

        Neuron h = new Neuron(weights1, bias1);
        Neuron o = new Neuron(weights2, bias2);

        NeuralNetwork nn = new NeuralNetwork(h, o);

        nn.train(trainingInputs, trainingOutputs, testInputs, testOutputs);   // train the network

        System.out.printf("Accuracy: %.2f%%\n", nn.precision(testInputs, testOutputs) * 100);  // calculate accuracy

        // Print weights and biases
        System.out.println("Weights H\n");
        for(double w : h.getWeights()) {
            System.out.print(w + " , ");
        }

        System.out.println("\n\nBias H\n");
        System.out.println(h.getBias());

        System.out.println("\nWeights O\n");
        for(double w : o.getWeights()) {
            System.out.print(w + " , ");
        }

        System.out.println("\n\nBias O\n");
        System.out.println(o.getBias());
    }

    /**
     * Reads image pixels
     * @param filePath File containing the images
     * @param numberOfLines Number of lines to read (images)
     * @param mode train or test, depending on intended use
     * @return array of images
     * @throws IOException file error
     */
    private static double[][] loadInputs(String filePath, int numberOfLines, String mode) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;

        double[][] inputs = new double[numberOfLines][400];
        int startLine = 0;

        // if it's test data
        if (mode.equals("test")) {
            // Count total number of lines in the file
            int totalLines = 0;
            while (br.readLine() != null) {
                totalLines++;
            }

            // Close and reopen the BufferedReader to reset it
            br.close();
            br = new BufferedReader(new FileReader(filePath));

            // Define starting point for reading in "test" mode. If train, startLine = 0
            startLine = totalLines - numberOfLines;
        }

        // Skip to the start line (if test) — if train, startLine = 0 does nothing
        for (int i = 0; i < startLine; i++) {
            br.readLine(); // Skip lines until reaching the start
        }

        // Read the next lines and process
        int sampleIndex = 0;
        while (sampleIndex < numberOfLines) {
            line = br.readLine();
            String[] pixelValues = line.split(",");

            for (int i = 0; i < 400; i++) {
                double value = normalizeInput(pixelValues[i]);
                inputs[sampleIndex][i] = value;
            }
            sampleIndex++;
        }

        br.close();
        return inputs;
    }

    /**
     * Reads the image labels
     * @param filePath File containing the labels
     * @param numberOfLines Number of lines to read
     * @param mode train or test, depending on intended use
     * @return array of labels
     * @throws IOException file error
     */
    private static double[] loadOutputs(String filePath, int numberOfLines, String mode) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;

        double[] outputs = new double[numberOfLines];
        int startLine = 0;

        if (mode.equals("test")) {
            // Count total number of lines in the file
            int totalLines = 0;
            while (br.readLine() != null) {
                totalLines++;
            }

            // Close and reopen the BufferedReader to reset it
            br.close();
            br = new BufferedReader(new FileReader(filePath));

            // Define starting point for reading in "test" mode. If train, startLine = 0
            startLine = totalLines - numberOfLines;
        }

        // Skip to the start line (if test)
        for (int i = 0; i < startLine; i++) {
            br.readLine(); // Skip lines
        }

        // Read next lines and process
        int sampleIndex = 0;
        while (sampleIndex < numberOfLines) {
            line = br.readLine();
            outputs[sampleIndex++] = Double.parseDouble(line);
        }

        br.close();
        return outputs;
    }

    /**
     * Normalizes input between 0 and 1
     * @param input pixel value
     * @return [0,1]
     */
    private static double normalizeInput(String input) {
        double number = Double.parseDouble(input);

        if (number < 0) return 0;
        else if (number > 1) return 1;
        else return number;
    }

    /**
     * Data augmentation function
     * @param inputs dataset
     * @return new dataset after augmentation
     */
    private static double[][] augmentData(double[][] inputs) {
        int originalSize = inputs.length;
        int augmentedSize = originalSize * 2;
        double[][] augmentedInputs = new double[augmentedSize][400];

        // Copy originals
        System.arraycopy(inputs, 0, augmentedInputs, 0, originalSize);

        // Create new samples
        for (int i = 0; i < originalSize; i++) {
            double[] original = inputs[i];
            double[] augmented = augmentSample(original);
            augmentedInputs[originalSize + i] = augmented;
        }

        return augmentedInputs;
    }

    /**
     * Applies data augmentation to a sample
     * @param sample original image
     * @return new image
     */
    private static double[] augmentSample(double[] sample) {
        double[] augmented = Arrays.copyOf(sample, sample.length);
        augmented = rotateImage90Degrees(augmented, 20, 20); // Rotate 90 degrees
        return augmented;
    }

    /**
     * Rotates an image
     * @param original original image
     * @param width width
     * @param height height
     * @return rotated image
     */
    private static double[] rotateImage90Degrees(double[] original, int width, int height) {
        double[][] matrix = new double[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix[i][j] = original[i * width + j];
            }
        }

        double[][] rotated = new double[width][height];

        // 90-degree clockwise rotation: (i,j) -> (j, height - 1 - i)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                rotated[j][height - 1 - i] = matrix[i][j];
            }
        }

        double[] rotatedArray = new double[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                rotatedArray[i * height + j] = rotated[i][j];
            }
        }

        return rotatedArray;
    }
}
