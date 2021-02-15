package similarity_vectors;

import measures_association.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

public class SimilarityVectorsRunner {
  private static String inputMeasuresPath;
  private static String outputBucketPath;
  private static boolean outputCooccurrenceVectors;
  private static String goldenStandardPath;
  private static final String LOG_PATH = "/log-files/";

  public static void main(String[] args) throws IOException, URISyntaxException {

    if (args.length < 4) {
      System.err.println(
          "Wrong argument count received.\nExpected <input-measures-path> <output-s3-path> <golden-standard-path>.");
      System.exit(1);
    }
    inputMeasuresPath = args[0];
    outputCooccurrenceVectors = Boolean.parseBoolean(args[1]);
    outputBucketPath = args[2];
    goldenStandardPath = args[3];

    // Measures of vector similarity
    Configuration vectorSimilarity = new Configuration();
    final Job calculateVectorSimilarity =
        Job.getInstance(vectorSimilarity, "Calculate Measures Of Vector Similarity");
    String calculateVectorSimilarityPath =
        createMeasuresOfSimilarityJob(calculateVectorSimilarity, inputMeasuresPath);
    waitForJobCompletion(calculateVectorSimilarity, calculateVectorSimilarityPath);

    if(outputCooccurrenceVectors) {
      // Output co-occurrence vectors - Used for testing purposes only
      Configuration cooccurrenceVectors = new Configuration();
      final Job createCooccurrenceVectors =
              Job.getInstance(cooccurrenceVectors, "Create Co-Occurrence Vectors - TESTS");
      String createCooccurrenceVectorsPath =
              createCooccurrenceVectorsJob(createCooccurrenceVectors, inputMeasuresPath);
      waitForJobCompletion(createCooccurrenceVectors, createCooccurrenceVectorsPath);
    }

    System.out.printf(
        "\nFinished all jobs successfully: output can be found in s3 path: %s%n",
        String.format("%s/result", outputBucketPath));
  }

  private static String setInputOutput(Job job, String inputPath, boolean isResult)
      throws IOException {
    if (inputPath != null) {
      FileInputFormat.addInputPath(job, new Path(inputPath));
    }
    job.setInputFormatClass(SequenceFileInputFormat.class);
    String outputPath =
        isResult
            ? String.format("%s/result", outputBucketPath)
            : String.format("%s/co-occurrence_vectors", outputBucketPath);
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    return outputPath;
  }

  private static String createMeasuresOfSimilarityJob(Job job, String filePath)
      throws URISyntaxException, IOException {
    job.setJarByClass(MeasuresVectorSimilarity.class);
    job.setMapperClass(MeasuresVectorSimilarity.MapperClass.class);
    job.setPartitionerClass(MeasuresVectorSimilarity.PartitionerClass.class);
    job.setReducerClass(MeasuresVectorSimilarity.ReducerClass.class);
    job.setSortComparatorClass(MeasuresVectorSimilarity.MeasuresVectorSimilarityComparator.class);
    job.setGroupingComparatorClass(
        MeasuresVectorSimilarity.MeasuresVectorSimilarityGroupingComparator.class);
    job.setMapOutputKeyClass(LexemePairWritable.class);
    job.setMapOutputValueClass(AssociationMeasuresWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(SimilarityVectorWritable.class);
    job.addCacheFile(new URI(String.format("%s#words", goldenStandardPath)));
    return setInputOutput(job, filePath, true);
  }

  private static String createCooccurrenceVectorsJob(Job job, String filePath)
          throws URISyntaxException, IOException {
    job.setJarByClass(CreateCooccurrenceVectors.class);
    job.setMapperClass(CreateCooccurrenceVectors.MapperClass.class);
    job.setPartitionerClass(CreateCooccurrenceVectors.PartitionerClass.class);
    job.setReducerClass(CreateCooccurrenceVectors.ReducerClass.class);
    job.setSortComparatorClass(CreateCooccurrenceVectors.CooccurrenceVectorsComparator.class);
    job.setGroupingComparatorClass(
            CreateCooccurrenceVectors.CooccurrenceVectorsGroupingComparator.class);
    job.setMapOutputKeyClass(TextPairWritable.class);
    job.setMapOutputValueClass(AssociationMeasuresWritable.class);
    job.setOutputKeyClass(TextPairWritable.class);
    job.setOutputValueClass(Text.class);
    job.addCacheFile(new URI(String.format("%s#words", goldenStandardPath)));
    return setInputOutput(job, filePath, false);
  }

  private static void waitForJobCompletion(final Job job, String outputPath) {
    String description = job.getJobName();
    System.out.printf("Started %s job.%n", description);
    try {
      if (job.waitForCompletion(true)) {
        System.out.printf(
            "%s finished successfully, output in S3 bucket %s.%n", description, outputPath);
      } else {
        System.out.printf("%s failed!, logs in S3 bucket at %s.%n", description, LOG_PATH);
        System.exit(1);
      }
    } catch (InterruptedException | IOException | ClassNotFoundException e) {
      System.err.printf(
          "Exception caught! EXCEPTION: %s\nLogs in S3 bucket at %s.%n", e.getMessage(), LOG_PATH);
      System.exit(1);
    }
  }
}
