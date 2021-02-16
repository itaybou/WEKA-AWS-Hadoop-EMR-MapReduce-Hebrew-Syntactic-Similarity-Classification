package classifier;

import opennlp.tools.ml.perceptron.PerceptronModel;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.Range;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.*;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

public class Classifier {
    private static final int K_FOLDS = 10;
    private static final String CLASSIFIER_SUMMARY_FILE = "classifier_summary";
    private static final String CLASSIFIER_PREDICTIONS_FILE = "classifier_predictions";

    public static void classify(String arffInputPath, String classifierOutputPath) {
        int k = K_FOLDS;

        try {
            DataSource source = new DataSource(arffInputPath);
            Instances classifierData = source.getDataSet();
            if(classifierData.numInstances() < K_FOLDS) {
                k = classifierData.numInstances();
            }

            System.out.println("\nClassifier Input Data Structure:\n===============================\n" + source.getStructure());
            System.out.println("Initializing Random Forest classifier with " + k + "-folds cross validation.");
            classifierData.setClassIndex(classifierData.numAttributes() - 1); // the classification attribute is last attribute in the ARFF file

            RandomForest randomForestClassifier = new RandomForest();
            Evaluation evaluation = new Evaluation(classifierData);

            System.out.println("Training & Evaluating Random Forest classifier.");
            // Train
            evaluation.crossValidateModel(randomForestClassifier, classifierData, k, new Random(1));

            System.out.println("Training & Evaluation completed. Outputting Summary and Prediction files.");
            writeResultsToFile(classifierOutputPath, evaluation);
            System.out.println("Classification and Evaluation done.");
        } catch (Exception e) {
            System.err.println("Failed to run classifier.\n" + e.getMessage());
        }
    }

    public static void writeResultsToFile(String outputPath, Evaluation eval)
    {
        String outputSummaryFile = String.format("%s/%s", outputPath, CLASSIFIER_SUMMARY_FILE);
        try (PrintWriter writer = new PrintWriter(outputSummaryFile, "UTF-8")) {
            writer.println((eval.toSummaryString(("\nWEKA Random Forest Classifier Results\n===============================================\n"), true)));
            writer.println("F1 Measure: " + eval.fMeasure(1));
            writer.println("Precision: " + eval.precision(1));
            writer.println("Recall: " + eval.recall(1) + "\n");
            writer.println("True-Positive rate: " + eval.truePositiveRate(1));
            writer.println("False-Positive rate: " + eval.falsePositiveRate(1));
            writer.println("True-Negative rate: " + eval.trueNegativeRate(1));
            writer.println("False-Negative rate: " + eval.falseNegativeRate(1) + "\n");
            writer.println(eval.toClassDetailsString());
            writer.println(eval.toMatrixString());
            writer.println("===============================================");
            System.out.println("Classification Summary file can be found at: " + outputSummaryFile);
        }
        catch (IOException e) {
            System.err.println("Error occurred while writing Weka results to output file.\n");
        } catch (Exception e) {
            System.err.println("Failed to output classifier results summary.\n");
        }
    }
}
