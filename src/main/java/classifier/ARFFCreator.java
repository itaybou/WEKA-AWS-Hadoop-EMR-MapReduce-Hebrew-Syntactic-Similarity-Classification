package classifier;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class ARFFCreator {
    public static final String ARFF_FILENAME = "word_pair_similarity.arff";
    private static Map<Integer, String> samplesToWordPair;

    public static String arffHeader =
            "% Lexeme Pairs Similarity Vectors\n\n" +
            "@relation similarity_vectors\n\n" +
            "% Plain frequency measures\n" +
            "@attribute plain_manhattan_distance\treal\n" +
            "@attribute plain_euclidean_distance\treal\n" +
            "@attribute plain_cosine_distance\treal\n" +
            "@attribute plain_jaccard_measure\treal\n" +
            "@attribute plain_dice_measure\treal\n" +
            "@attribute plain_jensen_shannon_divergence\treal\n" +
            "% Relative frequency measures\n" +
            "@attribute relative_manhattan_distance\treal\n" +
            "@attribute relative_euclidean_distance\treal\n" +
            "@attribute relative_cosine_distance\treal\n" +
            "@attribute relative_jaccard_measure\treal\n" +
            "@attribute relative_dice_measure\treal\n" +
            "@attribute relative_jensen_shannon_divergence\treal\n" +
            "% PMI frequency measures\n" +
            "@attribute pmi_manhattan_distance\treal\n" +
            "@attribute pmi_euclidean_distance\treal\n" +
            "@attribute pmi_cosine_distance\treal\n" +
            "@attribute pmi_jaccard_measure\treal\n" +
            "@attribute pmi_dice_measure\treal\n" +
            "@attribute pmi_jensen_shannon_divergence\treal\n" +
            "% T-Test frequency measures\n" +
            "@attribute t_test_manhattan_distance\treal\n" +
            "@attribute t_test_euclidean_distance\treal\n" +
            "@attribute t_test_cosine_distance\treal\n" +
            "@attribute t_test_jaccard_measure\treal\n" +
            "@attribute t_test_dice_measure\treal\n" +
            "@attribute t_test_jensen_shannon_divergence\treal\n\n" +
            "@attribute similar\t{FALSE,TRUE}\n\n" +
            "@data\n\n";


    public static String createARFF(String classifierInputPath, String classifierOutputPath) {
        samplesToWordPair = new HashMap<>();

        System.out.println("Creating WEKA classifier input ARFF file from word pair similarity vector files.");
        File outDir = new File(classifierOutputPath);
        if(!outDir.exists()) {
            outDir.mkdirs();
        }
        String arffPath = String.format("%s/%s", classifierOutputPath, ARFF_FILENAME);
        try(PrintWriter writer = new PrintWriter(arffPath)) {
            writer.println(arffHeader);
            File inputDir = new File(classifierInputPath);
            File[] vectorFiles = inputDir.listFiles();

            Integer sampleId = 0;
            for(File vectorFile : vectorFiles) {
                if(vectorFile.isFile()) {
                    String line;
                    try(BufferedReader reader = new BufferedReader(new FileReader(vectorFile))) {
                        while((line = reader.readLine()) != null) {
                            String[] lineParts = line.split("\t");
                            writer.println(String.format("%% %s", lineParts[0])); // Write comment with the word pair above the vector
                            samplesToWordPair.put(sampleId++, lineParts[0]);
                            writeVectorLine(writer, lineParts[1], lineParts[2]);
                        }
                    } catch (IOException e) {
                        System.err.println("Unable to read similarity vector file " + vectorFile.getName() + "\n" + e.getMessage());
                        return null;
                    }
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("ARFF file for WEKA classifier does not exist:\n" + e.getMessage());
            return null;
        }
        System.out.println("Classifier ARFF input file created successfully.");
        return arffPath;
    }

    private static void writeVectorLine(PrintWriter writer, String vector, String classification) {
        String clearedVector = vector.replaceAll("[\\[\\]\\s+]", "").replaceAll("NaN", String.valueOf(Double.MIN_VALUE));
        writer.println(String.format("%s,%s", clearedVector, classification.toUpperCase()));
    }

    public static Map<Integer, String> getSamplesToWordPair() {
        return samplesToWordPair;
    }
}
