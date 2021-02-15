package similarity_vectors;

import measures_association.AssociationMeasuresWritable;
import opennlp.tools.stemmer.PorterStemmer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import utils.SyntacticTextUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;

public class MeasuresVectorSimilarity {

  /**
   * Input shape:
   *    key: <lexeme>
   *    value: <<lexeme, feature, plain-frequency, relative-frequency, pmi, t-test>>
   * Output shape:
   *    key: <<lexeme1, lexeme2>, lexeme, feature> (<lexeme1, lexeme2> from golden standard)
   *    value: <<lexeme, feature, plain-frequency, relative-frequency, pmi, t-test>>
   */
  public static class MapperClass extends Mapper<Text, AssociationMeasuresWritable, LexemePairWritable, AssociationMeasuresWritable> {

    private Map<String, List<Text>> goldenStandard; // Golden Standard stemmed lexemes to relevant lexeme pairs
    private PorterStemmer stemmer;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      goldenStandard = new HashMap<>();
      stemmer = new PorterStemmer();
      if (context.getCacheFiles() != null && context.getCacheFiles().length > 0) {
        URI mappingFileUri = context.getCacheFiles()[0];
        if (mappingFileUri != null) {
          String filePath = mappingFileUri.toString().split("#")[1];
          parseGoldenStandardFile(String.format("./%s", filePath));
        }
      }
    }

    private void parseGoldenStandardFile(String path) {
      try (BufferedReader br = new BufferedReader(new FileReader(path))) {
        String line;
        while ((line = br.readLine()) != null) {
          String[] lineParts = line.split("\t");
          String[] lexemes = new String[2];
          System.arraycopy(lineParts, 0, lexemes, 0, 2);
          Arrays.sort(lexemes);
          Text pairText = SyntacticTextUtils.createLexemePair(lexemes[0], lexemes[1]);
          goldenStandard.computeIfAbsent(stemmer.stem(lexemes[0]).toLowerCase(Locale.ROOT), k -> new ArrayList<>()).add(pairText);
          goldenStandard.computeIfAbsent(stemmer.stem(lexemes[1]).toLowerCase(Locale.ROOT), k -> new ArrayList<>()).add(pairText);
        }
      } catch (IOException e) {
        System.err.println("Error reading golden standard classification file." + e.getMessage());
      }
    }

    @Override
    public void map(Text lexeme, AssociationMeasuresWritable assocMeasurements, Context context)
        throws IOException, InterruptedException {
      List<Text> lexemePairs = goldenStandard.getOrDefault(lexeme.toString(), null);
      if (lexemePairs != null) {
        for (Text lexemePair : lexemePairs) {
          context.write(
              new LexemePairWritable(lexemePair, lexeme, assocMeasurements.getFeature()),
              assocMeasurements);
        }
      }
    }
  }

  /**
   * Input shape:
   *    key: <<lexeme1, lexeme2>, lexeme, feature> (<lexeme1, lexeme2> from golden standard)
   *    value: <<lexeme, feature, plain-frequency, relative-frequency, pmi, t-test>>
   * Output shape:
   *    key: <<lexeme1, lexeme2>>
   *    value: Similarity vector of size 24 between lexeme1 and lexeme2
   *            containing the following positions:
   *            1 plain frequency vector - Manhattan distance
   *            2 plain frequency vector - Euclidean distance
   *            3 plain frequency vector - Cosine distance
   *            4 plain frequency vector - Jaccard measure
   *            5 plain frequency vector - Dice measure
   *            6 plain frequency vector - Jensen-Shannon divergence
   *            7 relative frequency vector - Manhattan distance
   *            8 relative frequency vector - Euclidean distance
   *            9 relative frequency vector - Cosine distance
   *            10 relative frequency vector - Jaccard measure
   *            11 relative frequency vector - Dice measure
   *            12 relative frequency vector - Jensen-Shannon divergence
   *            13 pointwise mutual information vector - Manhattan distance
   *            14 pointwise mutual information vector - Euclidean distance
   *            15 pointwise mutual information vector - Cosine distance
   *            16 pointwise mutual information vector - Jaccard measure
   *            17 pointwise mutual information vector - Dice measure
   *            18 pointwise mutual information vector - Jensen-Shannon divergence
   *            19 t-test statistic vector - Manhattan distance
   *            20 t-test statistic vector - Euclidean distance
   *            21 t-test statistic vector - Cosine distance
   *            22 t-test statistic vector - Jaccard measure
   *            23 t-test statistic vector - Dice measure
   *            24 t-test statistic vector- Jensen-Shannon divergence
   */
  public static class ReducerClass extends Reducer<LexemePairWritable, AssociationMeasuresWritable, Text, SimilarityVectorWritable> {

    private Map<String, Boolean> goldenStandard = new HashMap<>(); // Golden Standard word pairs to similarity classification
    private PorterStemmer stemmer;

    protected void setup(Context context) throws IOException {
      stemmer = new PorterStemmer();
      if (context.getCacheFiles() != null && context.getCacheFiles().length > 0) {
        URI mappingFileUri = context.getCacheFiles()[0];
        if (mappingFileUri != null) {
          String filePath = mappingFileUri.toString().split("#")[1];
          parseGoldenStandardFile(String.format("./%s", filePath));
        }
      }
    }

    private void parseGoldenStandardFile(String path) {
      try (BufferedReader br = new BufferedReader(new FileReader(path))) {
        String line;
        while ((line = br.readLine()) != null) {
          String[] lineParts = line.split("\t");
          String[] lexemes = new String[2];
          System.arraycopy(lineParts, 0, lexemes, 0, 2);
          Arrays.sort(lexemes);
          Text pairText = SyntacticTextUtils.createLexemePair(lexemes[0], lexemes[1]);
          goldenStandard.put(pairText.toString(), Boolean.parseBoolean(lineParts[2]));
        }
      } catch (IOException e) {
        System.err.println("Error reading golden standard classification file." + e.getMessage());
      }
    }

    @Override
    public void reduce(LexemePairWritable key, Iterable<AssociationMeasuresWritable> cooccurrenceVectors, Context context)
        throws IOException, InterruptedException {
      String[] pair = SyntacticTextUtils.splitPairTriplet(key.getPair());
      String[] stemmedPair = Arrays.stream(pair.clone()).map(stemmer::stem).toArray(String[]::new);
      AssociationMeasuresWritable current;
      AssociationMeasuresWritable previous = null;

      double plainFreqSubAbsSum = 0;
      double plainFreqSubSquaredSum = 0;
      double plainFreqMultSum = 0;
      double plainFreqMinSum = 0;
      double plainFreqMaxSum = 0;
      double plainFreqAddSum = 0;
      double plainKullbackLeiblerSum1 = 0;
      double plainKullbackLeiblerSum2 = 0;

      double relFreqSubAbsSum = 0;
      double relFreqSubSquaredSum = 0;
      double relFreqMultSum = 0;
      double relFreqMinSum = 0;
      double relFreqMaxSum = 0;
      double relFreqAddSum = 0;
      double relKullbackLeiblerSum1 = 0;
      double relKullbackLeiblerSum2 = 0;

      double pmiSubAbsSum = 0;
      double pmiSubSquaredSum = 0;
      double pmiMultSum = 0;
      double pmiMinSum = 0;
      double pmiMaxSum = 0;
      double pmiAddSum = 0;
      double pmiKullbackLeiblerSum1 = 0;
      double pmiKullbackLeiblerSum2 = 0;

      double tTestSubAbsSum = 0;
      double tTestSubSquaredSum = 0;
      double tTestMultSum = 0;
      double tTestMinSum = 0;
      double tTestMaxSum = 0;
      double tTestAddSum = 0;
      double tTestKullbackLeiblerSum1 = 0;
      double tTestKullbackLeiblerSum2 = 0;

      double sumSquarePlain1 = 0;
      double sumSquareRel1 = 0;
      double sumSquarePMI1 = 0;
      double sumSquareTtest1 = 0;

      double sumSquarePlain2 = 0;
      double sumSquareRel2 = 0;
      double sumSquarePMI2 = 0;
      double sumSquareTtest2 = 0;

      Iterator<AssociationMeasuresWritable> it = cooccurrenceVectors.iterator();
      while (it.hasNext()) {
        current = new AssociationMeasuresWritable(it.next());
        if (previous != null) {
          if (!previous.getLexeme().toString().equals(current.getLexeme().toString())
              && previous.getFeature().toString().equals(current.getFeature().toString())) {
            // Feature is present both of the vectors
            double plainFreq1 = previous.getPlainFrequency().get();
            double relFreq1 = previous.getRelativeFrequency().get();
            double pmi1 = previous.getPmi().get();
            double tTest1 = previous.gettTest().get();
            double plainFreq2 = current.getPlainFrequency().get();
            double relFreq2 = current.getRelativeFrequency().get();
            double pmi2 = current.getPmi().get();
            double tTest2 = current.gettTest().get();

            // Calculate sum(l[i]^2) for both vectors
            sumSquarePlain1 += Math.pow(plainFreq1, 2);
            sumSquareRel1 += Math.pow(relFreq1, 2);
            sumSquarePMI1 += Math.pow(pmi1, 2);
            sumSquareTtest1 += Math.pow(tTest1, 2);
            sumSquarePlain2 += Math.pow(plainFreq2, 2);
            sumSquareRel2 += Math.pow(relFreq2, 2);
            sumSquarePMI2 += Math.pow(pmi2, 2);
            sumSquareTtest2 += Math.pow(tTest2, 2);

            // Plain Frequency sums
            plainFreqSubAbsSum += Math.abs(plainFreq1 - plainFreq2);
            plainFreqSubSquaredSum += Math.pow((plainFreq1 - plainFreq2), 2);
            plainFreqMultSum += (plainFreq1 * plainFreq2);
            plainFreqMinSum += Math.min(plainFreq1, plainFreq2);
            plainFreqMaxSum += Math.max(plainFreq1, plainFreq2);
            plainFreqAddSum += (plainFreq1 + plainFreq2);
            // Kullback-Leibler divergence = sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))
            plainKullbackLeiblerSum1 +=
                Double.isNaN(Math.log((plainFreq1 / ((plainFreq1 + plainFreq2) / 2))))
                    ? 0
                    : (plainFreq1 * Math.log((plainFreq1 / ((plainFreq1 + plainFreq2) / 2))));
            // Kullback-Leibler divergence = sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2))
            plainKullbackLeiblerSum2 +=
                Double.isNaN(Math.log((plainFreq2 / ((plainFreq1 + plainFreq2) / 2))))
                    ? 0
                    : (plainFreq2 * Math.log((plainFreq2 / ((plainFreq1 + plainFreq2) / 2))));

            // Relative Frequency sums
            relFreqSubAbsSum += Math.abs(relFreq1 - relFreq2);
            relFreqSubSquaredSum += Math.pow((relFreq1 - relFreq2), 2);
            relFreqMultSum += (relFreq1 * relFreq2);
            relFreqMinSum += Math.min(relFreq1, relFreq2);
            relFreqMaxSum += Math.max(relFreq1, relFreq2);
            relFreqAddSum += (relFreq1 + relFreq2);
            // Kullback-Leibler divergence = sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))
            relKullbackLeiblerSum1 +=
                Double.isNaN(Math.log((relFreq1 / ((relFreq1 + relFreq2) / 2))))
                    ? 0
                    : (relFreq1 * Math.log((relFreq1 / ((relFreq1 + relFreq2) / 2))));
            // Kullback-Leibler divergence = sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2))
            relKullbackLeiblerSum2 +=
                Double.isNaN(Math.log((relFreq2 / ((relFreq1 + relFreq2) / 2))))
                    ? 0
                    : (relFreq2 * Math.log((relFreq2 / ((relFreq1 + relFreq2) / 2))));

            // PMI sums
            pmiSubAbsSum += Math.abs(pmi1 - pmi2);
            pmiSubSquaredSum += Math.pow((pmi1 - pmi2), 2);
            pmiMultSum += (pmi1 * pmi2);
            pmiMinSum += Math.min(pmi1, pmi2);
            pmiMaxSum += Math.max(pmi1, pmi2);
            pmiAddSum += (pmi1 + pmi2);
            // Kullback-Leibler divergence = sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))
            pmiKullbackLeiblerSum1 +=
                Double.isNaN(Math.log((pmi1 / ((pmi1 + pmi2) / 2))))
                    ? 0
                    : (pmi1 * Math.log((pmi1 / ((pmi1 + pmi2) / 2))));
            // Kullback-Leibler divergence = sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2))
            pmiKullbackLeiblerSum2 +=
                Double.isNaN(Math.log((pmi2 / ((pmi1 + pmi2) / 2))))
                    ? 0
                    : (pmi2 * Math.log((pmi2 / ((pmi1 + pmi2) / 2))));

            // T-test statistic sums
            tTestSubAbsSum += Math.abs(tTest1 - tTest2);
            tTestSubSquaredSum += Math.pow((tTest1 - tTest2), 2);
            tTestMultSum += (tTest1 * tTest2);
            tTestMinSum += Math.min(tTest1, tTest2);
            tTestMaxSum += Math.max(tTest1, tTest2);
            tTestAddSum += (tTest1 + tTest2);
            // Kullback-Leibler divergence = sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))
            tTestKullbackLeiblerSum1 +=
                Double.isNaN(Math.log((tTest1 / ((tTest1 + tTest2) / 2))))
                    ? 0
                    : (tTest1 * Math.log((tTest1 / ((tTest1 + tTest2) / 2))));
            // Kullback-Leibler divergence = sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2))
            tTestKullbackLeiblerSum2 +=
                Double.isNaN(Math.log((tTest2 / ((tTest1 + tTest2) / 2))))
                    ? 0
                    : (tTest2 * Math.log((tTest2 / ((tTest1 + tTest2) / 2))));

            current = null;
          } else {

            double plainFreq = previous.getPlainFrequency().get();
            double relFreq = previous.getRelativeFrequency().get();
            double pmi = previous.getPmi().get();
            double tTest = previous.gettTest().get();

            // Feature is present in only one of the vectors
            if (previous.getLexeme().toString().equals(stemmedPair[0])) { // Calculate sum(l[i]^2) for specific vector
              sumSquarePlain1 += Math.pow(plainFreq, 2);
              sumSquareRel1 += Math.pow(relFreq, 2);
              sumSquarePMI1 += Math.pow(pmi, 2);
              sumSquareTtest1 += Math.pow(tTest, 2);
              plainKullbackLeiblerSum1 += (plainFreq * Math.log((plainFreq / (plainFreq / 2))));
              relKullbackLeiblerSum1 += (relFreq * Math.log((relFreq / (relFreq / 2))));
              pmiKullbackLeiblerSum1 += (pmi * Math.log((pmi / (pmi / 2))));
              tTestKullbackLeiblerSum1 += (tTest * Math.log((tTest / (tTest / 2))));
            } else {
              sumSquarePlain2 += Math.pow(plainFreq, 2);
              sumSquareRel2 += Math.pow(relFreq, 2);
              sumSquarePMI2 += Math.pow(pmi, 2);
              sumSquareTtest2 += Math.pow(tTest, 2);
              plainKullbackLeiblerSum2 += (plainFreq * Math.log((plainFreq / (plainFreq / 2))));
              relKullbackLeiblerSum2 += (relFreq * Math.log((relFreq / (relFreq / 2))));
              pmiKullbackLeiblerSum2 += (pmi * Math.log((pmi / (pmi / 2))));
              tTestKullbackLeiblerSum2 += (tTest * Math.log((tTest / (tTest / 2))));
            }

            // Plain Frequency sums
            plainFreqSubAbsSum += plainFreq;
            plainFreqSubSquaredSum += Math.pow(plainFreq, 2);
            plainFreqMaxSum += plainFreq;
            plainFreqAddSum += plainFreq;

            // Relative Frequency sums
            relFreqSubAbsSum += relFreq;
            relFreqSubSquaredSum += Math.pow(relFreq, 2);
            relFreqMaxSum += relFreq;
            relFreqAddSum += relFreq;

            // PMI sums
            pmiSubAbsSum += pmi;
            pmiSubSquaredSum += Math.pow(pmi, 2);
            pmiMaxSum += pmi;
            pmiAddSum += pmi;

            // T-test statistic sums
            tTestSubAbsSum += tTest;
            tTestSubSquaredSum += Math.pow(tTest, 2);
            tTestMaxSum += tTest;
            tTestAddSum += tTest;
            if (!it.hasNext()) {
              plainFreq = current.getPlainFrequency().get();
              relFreq = current.getRelativeFrequency().get();
              pmi = current.getPmi().get();
              tTest = current.gettTest().get();

              // Feature is present in only one of the vectors
              if (current.getLexeme().toString().equals(stemmedPair[0])) { // Calculate sum(l[i]^2) for specific vector
                sumSquarePlain1 += Math.pow(plainFreq, 2);
                sumSquareRel1 += Math.pow(relFreq, 2);
                sumSquarePMI1 += Math.pow(pmi, 2);
                sumSquareTtest1 += Math.pow(tTest, 2);
                plainKullbackLeiblerSum1 += (plainFreq * Math.log((plainFreq / (plainFreq / 2))));
                relKullbackLeiblerSum1 += (relFreq * Math.log((relFreq / (relFreq / 2))));
                pmiKullbackLeiblerSum1 += (pmi * Math.log((pmi / (pmi / 2))));
                tTestKullbackLeiblerSum1 += (tTest * Math.log((tTest / (tTest / 2))));
              } else {
                sumSquarePlain2 += Math.pow(plainFreq, 2);
                sumSquareRel2 += Math.pow(relFreq, 2);
                sumSquarePMI2 += Math.pow(pmi, 2);
                sumSquareTtest2 += Math.pow(tTest, 2);
                plainKullbackLeiblerSum2 += (plainFreq * Math.log((plainFreq / (plainFreq / 2))));
                relKullbackLeiblerSum2 += (relFreq * Math.log((relFreq / (relFreq / 2))));
                pmiKullbackLeiblerSum2 += (pmi * Math.log((pmi / (pmi / 2))));
                tTestKullbackLeiblerSum2 += (tTest * Math.log((tTest / (tTest / 2))));
              }

              // Plain Frequency sums
              plainFreqSubAbsSum += plainFreq;
              plainFreqSubSquaredSum += Math.pow(plainFreq, 2);
              plainFreqMaxSum += plainFreq;
              plainFreqAddSum += plainFreq;

              // Relative Frequency sums
              relFreqSubAbsSum += relFreq;
              relFreqSubSquaredSum += Math.pow(relFreq, 2);
              relFreqMaxSum += relFreq;
              relFreqAddSum += relFreq;

              // PMI sums
              pmiSubAbsSum += pmi;
              pmiSubSquaredSum += Math.pow(pmi, 2);
              pmiMaxSum += pmi;
              pmiAddSum += pmi;

              // T-test statistic sums
              tTestSubAbsSum += tTest;
              tTestSubSquaredSum += Math.pow(tTest, 2);
              tTestMaxSum += tTest;
              tTestAddSum += tTest;
            }
          }
        }
        previous = current != null ? new AssociationMeasuresWritable(current) : null;
      }

      double[] similarityVector = new double[SimilarityVectorWritable.VECTOR_SIZE];

      // plain frequency vector - Manhattan distance - sum(abs(l1[i] - l2[i]))
      similarityVector[0] = plainFreqSubAbsSum;
      // plain frequency vector - Euclidean distance - sqrt(sum((l1[i] - l2[i])^2))
      similarityVector[1] = Math.sqrt(plainFreqSubSquaredSum);
      // plain frequency vector - Cosine distance - (sum(l1[i] * l2[i]) / (sqrt(sum(l1[i]^2)) * sqrt(sum(l2[i]^2))))
      similarityVector[2] =
          (plainFreqMultSum / (Math.sqrt(sumSquarePlain1) * Math.sqrt(sumSquarePlain2)));
      // plain frequency vector - Jaccard measure - (sum(min(l1[i], l2[i])) / sum(max(l1[i], l2[i])))
      similarityVector[3] = (plainFreqMinSum / plainFreqMaxSum);
      // plain frequency vector - Dice measure - ((2 * sum(min(l1[i], l2[i]))) / sum(l1[i] + l2[i]))
      similarityVector[4] = ((2 * plainFreqMinSum) / plainFreqAddSum);
      // plain frequency vector - Jensen-Shannon divergence - sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))) + sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2)))
      similarityVector[5] = (plainKullbackLeiblerSum1 + plainKullbackLeiblerSum2);

      // relative frequency vector - Manhattan distance - sum(abs(l1[i] - l2[i]))
      similarityVector[6] = relFreqSubAbsSum;
      // relative frequency vector - Euclidean distance - sqrt(sum((l1[i] - l2[i])^2))
      similarityVector[7] = Math.sqrt(relFreqSubSquaredSum);
      // relative frequency vector - Cosine distance - (sum(l1[i] * l2[i]) / (sqrt(sum(l1[i]^2)) * sqrt(sum(l2[i]^2))))
      similarityVector[8] =
          (relFreqMultSum / (Math.sqrt(sumSquareRel1) * Math.sqrt(sumSquareRel2)));
      // relative frequency vector - Jaccard measure - (sum(min(l1[i], l2[i])) / sum(max(l1[i], l2[i])))
      similarityVector[9] = (relFreqMinSum / relFreqMaxSum);
      // relative frequency vector - Dice measure - ((2 * sum(min(l1[i], l2[i]))) / sum(l1[i] + l2[i]))
      similarityVector[10] = ((2 * relFreqMinSum) / relFreqAddSum);
      // relative frequency vector - Jensen-Shannon divergence - sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))) + sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2)))
      similarityVector[11] = (relKullbackLeiblerSum1 + relKullbackLeiblerSum2);

      // pointwise mutual information vector - Manhattan distance - sum(abs(l1[i] - l2[i]))
      similarityVector[12] = pmiSubAbsSum;
      // pointwise mutual information vector - Euclidean distance - sqrt(sum((l1[i] - l2[i])^2))
      similarityVector[13] = Math.sqrt(pmiSubSquaredSum);
      // pointwise mutual information vector - Cosine distance - (sum(l1[i] * l2[i]) / (sqrt(sum(l1[i]^2)) * sqrt(sum(l2[i]^2))))
      similarityVector[14] = (pmiMultSum / (Math.sqrt(sumSquarePMI1) * Math.sqrt(sumSquarePMI2)));
      // pointwise mutual information vector - Jaccard measure - (sum(min(l1[i], l2[i])) / sum(max(l1[i], l2[i])))
      similarityVector[15] = (pmiMinSum / pmiMaxSum);
      // pointwise mutual information vector - Dice measure - ((2 * sum(min(l1[i], l2[i]))) / sum(l1[i] + l2[i]))
      similarityVector[16] = ((2 * pmiMinSum) / pmiAddSum);
      // pointwise mutual information vector - Jensen-Shannon divergence - sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))) + sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2)))
      similarityVector[17] = (pmiKullbackLeiblerSum1 + pmiKullbackLeiblerSum2);

      // t-test statistic vector - Manhattan distance - sum(abs(l1[i] - l2[i]))
      similarityVector[18] = tTestSubAbsSum;
      // t-test statistic vector - Euclidean distance - sqrt(sum((l1[i] - l2[i])^2))
      similarityVector[19] = Math.sqrt(tTestSubSquaredSum);
      // t-test statistic vector - Cosine distance - (sum(l1[i] * l2[i]) / (sqrt(sum(l1[i]^2)) * sqrt(sum(l2[i]^2))))
      similarityVector[20] =
          (tTestMultSum / (Math.sqrt(sumSquareTtest1) * Math.sqrt(sumSquareTtest2)));
      // t-test statistic vector - Jaccard measure - (sum(min(l1[i], l2[i])) / sum(max(l1[i], l2[i])))
      similarityVector[21] = (tTestMinSum / tTestMaxSum);
      // t-test statistic vector - Dice measure - ((2 * sum(min(l1[i], l2[i]))) / sum(l1[i] + l2[i]))
      similarityVector[22] = ((2 * tTestMinSum) / tTestAddSum);
      // t-test statistic vector- Jensen-Shannon divergence - sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))) + sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2)))
      similarityVector[23] = (tTestKullbackLeiblerSum1 + tTestKullbackLeiblerSum2);

      context.write(
          new Text(key.getPair().toString()),
          new SimilarityVectorWritable(
              goldenStandard.get(key.getPair().toString()),
              similarityVector,
              key.getPair().toString()));
    }
  }

  public static class PartitionerClass
      extends Partitioner<LexemePairWritable, AssociationMeasuresWritable> {

    @Override
    public int getPartition(
        LexemePairWritable key, AssociationMeasuresWritable value, int numPartitions) {
      return (key.hashCode() & 0xFFFFFFF) % numPartitions;
    }
  }

  public static class MeasuresVectorSimilarityComparator extends WritableComparator {

    protected MeasuresVectorSimilarityComparator() {
      super(LexemePairWritable.class, true);
    }

    @Override
    public int compare(WritableComparable w1, WritableComparable w2) {
      LexemePairWritable a = (LexemePairWritable) w1;
      LexemePairWritable b = (LexemePairWritable) w2;
      return a.compareTo(b);
    }
  }

  public static class MeasuresVectorSimilarityGroupingComparator extends WritableComparator {
    protected MeasuresVectorSimilarityGroupingComparator() {
      super(LexemePairWritable.class, true);
    }

    @Override
    public int compare(WritableComparable w1, WritableComparable w2) {
      LexemePairWritable a = (LexemePairWritable) w1;
      LexemePairWritable b = (LexemePairWritable) w2;
      return a.getPair()
          .toString()
          .toLowerCase(Locale.ROOT)
          .compareTo(b.getPair().toString().toLowerCase(Locale.ROOT)); // Group only by the pair of coocurrence vectors word pairs
    }
  }
}
