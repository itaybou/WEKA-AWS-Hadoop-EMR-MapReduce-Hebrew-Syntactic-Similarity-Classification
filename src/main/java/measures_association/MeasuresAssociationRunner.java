package measures_association;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import java.io.IOException;
import java.util.stream.IntStream;

public class MeasuresAssociationRunner {
  private static String inputCorpusPath;
  private static String outputBucketPath;
  private static String goldenStandardPath;
  private static int corpusFileCount;
  private static final String LOG_PATH = "/log-files/";

  private static long totalLexemeCount;
  private static long totalFeatureCount;

  public static void main(String[] args) throws IOException {

    if (args.length < 3) {
      System.err.println(
          "Wrong argument count received.\nExpected <input-corpus-path> <output-s3-path> <corpus-files-count>.");
      System.exit(1);
    }
    inputCorpusPath = args[0];
    outputBucketPath = args[1];
    corpusFileCount = Integer.parseInt(args[2]);

    // Parse Syntactic Dependencies
    Configuration parseSyntacticDependencies = new Configuration();
    final Job syntacticDependencies =
        Job.getInstance(parseSyntacticDependencies, "Parse Syntactic Dependencies");
    String syntacticDependenciesPath =
        createSyntacticDependenciesJob(syntacticDependencies, inputCorpusPath);
    waitForJobCompletion(syntacticDependencies, syntacticDependenciesPath);

    Counters counters = syntacticDependencies.getCounters();
    totalLexemeCount = counters.findCounter(CounterTypes.LEXEME_COUNTER).getValue(); // count(L)
    totalFeatureCount = counters.findCounter(CounterTypes.FEATURE_COUNTER).getValue(); // count(F)

    // Order And Count Lexeme Feature
    Configuration orderAndCountLexemeFeatures = new Configuration();
    final Job lexemeFeaturesCount =
        Job.getInstance(orderAndCountLexemeFeatures, "Order And Count Lexeme Feature");
    String lexemeFeaturesCountPath =
        createOrderAndCountLexemeFeatureJob(lexemeFeaturesCount, syntacticDependenciesPath);
    waitForJobCompletion(lexemeFeaturesCount, lexemeFeaturesCountPath);

    // Calculate Measures of association with context
    Configuration calculateAssociationWithContext = new Configuration();
    calculateAssociationWithContext.setLong("countL", totalLexemeCount);
    calculateAssociationWithContext.setLong("countF", totalFeatureCount);
    final Job associationWithContext =
        Job.getInstance(
            calculateAssociationWithContext, "Calculate Measures Of Association With Context");
    String associationWithContextPath =
        createMeasuresOfAssocationJob(associationWithContext, lexemeFeaturesCountPath);
    waitForJobCompletion(associationWithContext, associationWithContextPath);


    System.out.printf(
        "\nFinished all jobs successfully: output can be found in s3 path: %s%n",
        String.format("%s/result", outputBucketPath));
  }

  private static String setInputOutput(Job job, String inputPath, boolean finished)
      throws IOException {
    if (inputPath != null) {
      FileInputFormat.addInputPath(job, new Path(inputPath));
    }
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
    String outputPath =
        finished
            ? String.format("%s/result", outputBucketPath)
            : String.format("%s/jobs/%s", outputBucketPath, job.getJobName());
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    return outputPath;
  }

  private static String setCorpusInputOutput(Job job, String inputPath) throws IOException {
    IntStream.range(0, corpusFileCount)
        .forEach(
            i -> {
              try {
                FileInputFormat.addInputPath(
                    job, new Path(String.format("%s/biarcs.%02d-of-99", inputPath, i)));
              } catch (IOException e) {
                e.printStackTrace();
              }
            });
    return setInputOutput(job, null, false);
  }

  private static String createSyntacticDependenciesJob(Job job, String filePath)
      throws IOException {
    job.setJarByClass(ParseSyntacticDependencies.class);
    job.setMapperClass(ParseSyntacticDependencies.MapperClass.class);
    job.setPartitionerClass(ParseSyntacticDependencies.PartitionerClass.class);
    job.setCombinerClass(ParseSyntacticDependencies.ReducerClass.class);
    job.setReducerClass(ParseSyntacticDependencies.ReducerClass.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(LongWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(LongWritable.class);
    return setCorpusInputOutput(job, filePath);
  }

  private static String createOrderAndCountLexemeFeatureJob(Job job, String filePath)
      throws IOException {
    job.setJarByClass(CountLexemeFeatures.class);
    job.setMapperClass(CountLexemeFeatures.MapperClass.class);
    job.setPartitionerClass(CountLexemeFeatures.PartitionerClass.class);
    job.setReducerClass(CountLexemeFeatures.ReducerClass.class);
    job.setSortComparatorClass(CountLexemeFeatures.CountLexemeFeaturesComparator.class);
    job.setGroupingComparatorClass(CountLexemeFeatures.CountLexemeFeaturesGroupingComparator.class);
    job.setMapOutputKeyClass(SyntacticPairWritable.class);
    job.setMapOutputValueClass(TextPairWritable.class);
    job.setOutputKeyClass(SyntacticPairWritable.class);
    job.setOutputValueClass(Text.class);
    return setInputOutput(job, filePath, false);
  }

  private static String createMeasuresOfAssocationJob(Job job, String filePath) throws IOException {
    job.setJarByClass(MeasuresAssociationContext.class);
    job.setMapperClass(MeasuresAssociationContext.MapperClass.class);
    job.setPartitionerClass(MeasuresAssociationContext.PartitionerClass.class);
    job.setReducerClass(MeasuresAssociationContext.ReducerClass.class);
    job.setSortComparatorClass(
        MeasuresAssociationContext.MeasuresAssociationContextComparator.class);
    job.setGroupingComparatorClass(
        MeasuresAssociationContext.MeasuresAssociationContextGroupingComparator.class);
    job.setMapOutputKeyClass(SyntacticPairWritable.class);
    job.setMapOutputValueClass(TextPairWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(AssociationMeasuresWritable.class);
    return setInputOutput(job, filePath, true);
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
