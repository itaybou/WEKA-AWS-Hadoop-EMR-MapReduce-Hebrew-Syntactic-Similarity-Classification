import classifier.ARFFCreator;
import classifier.Classifier;
import org.apache.commons.io.FileUtils;
import software.amazon.awssdk.core.sync.ResponseTransformer;
import software.amazon.awssdk.services.ec2.model.InstanceType;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.emr.EmrClient;
import software.amazon.awssdk.services.emr.model.*;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.*;
import software.amazon.awssdk.services.s3.paginators.ListObjectsV2Iterable;

import java.io.*;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Input syntactic corpus is taken from {@see https://storage.googleapis.com/books/syntactic-ngrams/index.html}
 * English-All Biarcs dataset.
 */
public class Main {
  private static final Region REGION = Region.US_EAST_1;
  private static final String USER_FILE_PATH = "inputs.txt";
  private static final String CLASSIFIER_INPUT_PATH = "similarity_vectors/output/result";

  private static String inBucket; // Get from user file
  private static String inputJarName; // Get from user file
  private static String inputGoldStandardFile; // Get from user file
  private static String corpusPath; // Get from user file
  private static String outBucket; // Get from user file
  private static String inputMeasuresBucket; // Get from user file
  private static String classifierOutputPath; // Get from user file
  private static String classifierInputPath; // Get from user file
  private static int instanceCount; // Get from user file
  private static boolean createAssociationMeasures; // Get from user file
  private static boolean runClassifier; // Get from user file
  private static boolean uploadJARAndGolden; // Get from user file
  private static boolean outputCooccurrenceVectors; // Get from user file
  private static int corpusFileCount; // Get from user file
  private static boolean finishedDelete; // Get from user file

  private static S3Client s3;

  public static void main(String[] args) throws InterruptedException {
    s3 = S3Client.builder().region(REGION).build();
    EmrClient emr = EmrClient.builder().region(REGION).build();
    List<StepConfig> steps = new ArrayList<>();

    if (readUserFile()) {
      if (!runWekaClassification()) {
        if (uploadJARAndGolden) {
          uploadJARAndGoldenToBucket();
        }

        if (createAssociationMeasures) {
          HadoopJarStepConfig hadoopJarStep =
              HadoopJarStepConfig.builder()
                  .jar(String.format("s3n://%s/%s", inBucket, inputJarName))
                  .mainClass("measures_association.MeasuresAssociationRunner")
                  .args(
                      corpusPath,
                      String.format("s3n://%s/measures_association/output/", outBucket),
                      String.valueOf(corpusFileCount))
                  .build();

          StepConfig stepConfig =
              StepConfig.builder()
                  .name("Create Syntactic Measures Of Association")
                  .hadoopJarStep(hadoopJarStep)
                  .actionOnFailure("TERMINATE_JOB_FLOW")
                  .build();
          steps.add(stepConfig);
        }

        HadoopJarStepConfig hadoopJarStep =
            HadoopJarStepConfig.builder()
                .jar(String.format("s3n://%s/%s", inBucket, inputJarName))
                .mainClass("similarity_vectors.SimilarityVectorsRunner")
                .args(
                    inputMeasuresBucket,
                    String.valueOf(outputCooccurrenceVectors),
                    String.format("s3n://%s/similarity_vectors/output/", outBucket),
                    String.format("s3n://%s/%s", inBucket, inputGoldStandardFile))
                .build();

        StepConfig stepConfig =
            StepConfig.builder()
                .name("Create Golden Standard Similarity Vectors")
                .hadoopJarStep(hadoopJarStep)
                .actionOnFailure("TERMINATE_JOB_FLOW")
                .build();
        steps.add(stepConfig);

        JobFlowInstancesConfig instances =
            JobFlowInstancesConfig.builder()
                .instanceCount(instanceCount)
                .masterInstanceType(InstanceType.M4_LARGE.toString())
                .slaveInstanceType(InstanceType.M4_LARGE.toString())
                .hadoopVersion("3.2.1")
                .keepJobFlowAliveWhenNoSteps(false)
                .placement(PlacementType.builder().availabilityZone("us-east-1a").build())
                .build();

        RunJobFlowRequest runFlowRequest =
            RunJobFlowRequest.builder()
                .name("Word Pair Syntactic Similarity Vectors")
                .instances(instances)
                .steps(steps)
                .releaseLabel("emr-6.2.0")
                .jobFlowRole("EMR_EC2_DefaultRole")
                .serviceRole("EMR_DefaultRole")
                .logUri(String.format("s3n://%s/log-files/", outBucket))
                .build();

        RunJobFlowResponse runFlowResponse = emr.runJobFlow(runFlowRequest);
        String jobFlowId = runFlowResponse.jobFlowId();

        System.out.printf("\n\nWord Pair Similarities job started with job ID: %s%n", jobFlowId);
        System.out.printf(
            "The following steps will run:\n%s\n\n",
            steps.stream().map(StepConfig::name).collect(Collectors.joining("\n")));

        SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss z");

        while (true) {
          DescribeClusterRequest clusterRequest =
              DescribeClusterRequest.builder().clusterId(jobFlowId).build();
          DescribeClusterResponse clusterResponse = emr.describeCluster(clusterRequest);
          ClusterState jobState = clusterResponse.cluster().status().state();

          System.out.printf(
              "Job Status: %s\t%s%n",
              jobState, formatter.format(new Date(System.currentTimeMillis())));
          switch (jobState) {
            case TERMINATED_WITH_ERRORS:
              System.err.println(
                  "Job terminated with errors. Downloading log files to output directory.\n");
              downloadOutputBucket(jobFlowId, true);
              Thread deletionThread = deleteOutputBucket();
              if (deletionThread != null) {
                deletionThread.join();
              }
              System.exit(1);
            case TERMINATING:
            case TERMINATED:
              System.out.println(
                  "Job completed successfully, downloading results output and log files.\n");
              downloadOutputBucket(jobFlowId, false);
              classifierInputPath = CLASSIFIER_INPUT_PATH;
              Thread deletionThreadTerminated = deleteOutputBucket();
              runWekaClassification();
              if(deletionThreadTerminated != null) {
                deletionThreadTerminated.join();
              }
              System.exit(0);
          }
          Thread.sleep(5000);
        }
      }
    }
  }

  private static boolean runWekaClassification() {
    if (runClassifier && classifierInputPath != null && new File(classifierInputPath).isDirectory()) {
      String arffInputPath = ARFFCreator.createARFF(classifierInputPath, classifierOutputPath);
      if(arffInputPath == null) {
        System.out.println("Failed to start classifier, ARFF File missing.");
        return false;
      }
      Classifier.classify(arffInputPath, classifierOutputPath);
      return true;
    }
    return false;
  }

  public static void uploadJARAndGoldenToBucket() {
    System.out.println("Uploading JAR file and Golden Standard file to S3 bucket: " + inBucket);
    try {
      PutObjectRequest putObjectRequest =
          PutObjectRequest.builder()
              .bucket(inBucket)
              .key(inputJarName)
              .acl(ObjectCannedACL.PUBLIC_READ_WRITE)
              .build();
      s3.putObject(putObjectRequest, Paths.get(inputJarName));

      putObjectRequest =
              PutObjectRequest.builder()
                      .bucket(inBucket)
                      .key(inputGoldStandardFile)
                      .acl(ObjectCannedACL.PUBLIC_READ_WRITE)
                      .build();
      s3.putObject(putObjectRequest, Paths.get(inputGoldStandardFile));
      System.out.println("Finished Uploading JAR file and Golden Standard file.");
    } catch (Exception e) {
      System.err.println("Failed to upload JAR file and Golden Standard file to provided S3 bucket.");
      System.err.println(e.getMessage());
    }
  }

  public static void downloadOutputBucket(String jobId, boolean onlyLog) {
    File outDir = new File("output");
    if (!outDir.isDirectory()) {
      outDir.mkdir();
    }
    if (!onlyLog) {
      System.out.println("Downloading output files from S3");
      ListObjectsV2Request outputListObjectRequest =
          ListObjectsV2Request.builder().bucket(outBucket).prefix("similarity_vectors/output/result").build();
      downloadBucketDirectory(outputListObjectRequest, false);

      if(outputCooccurrenceVectors) {
        System.out.println("Downloading co-occurrence vectors files from S3");
        outputListObjectRequest =
                ListObjectsV2Request.builder().bucket(outBucket).prefix("similarity_vectors/output/co-occurrence_vectors").build();
        downloadBucketDirectory(outputListObjectRequest, false);
      }
    }

    System.out.println("Download files completed.\nOutput is in the output folder created.");

    System.out.println("Downloading log files from S3");
    ListObjectsV2Request logListObjectRequest =
        ListObjectsV2Request.builder()
            .bucket(outBucket)
            .prefix(String.format("log-files/%s/steps", jobId))
            .build();
    downloadBucketDirectory(logListObjectRequest, true);
  }

  private static void downloadBucketDirectory(ListObjectsV2Request request, boolean addOutput) {
    ListObjectsV2Iterable listObjectResponse = s3.listObjectsV2Paginator(request);
    listObjectResponse.stream().forEach(page ->
      page.contents().forEach(object -> {
          if (object.size() > 0) {
            System.out.printf(
                "Downloading file to %s%s%n", addOutput ? "output/" : "", object.key());
            InputStream stream = getFileStream(outBucket, object.key());
            File file = new File(String.format("%s%s", addOutput ? "output/" : "", object.key()));
            try {
              assert stream != null;
              FileUtils.copyInputStreamToFile(stream, file);
            } catch (IOException e) {
              e.printStackTrace();
            }
          }
      }));
  }

  public static InputStream getFileStream(String bucketName, String key) {
    try {
      GetObjectRequest objectRequest =
          GetObjectRequest.builder().bucket(bucketName).key(key).build();
      return s3.getObject(objectRequest, ResponseTransformer.toBytes()).asInputStream();
    } catch (Exception e) {
      return null;
    }
  }

  public static void deleteObject(String bucketName, String key) {
      DeleteObjectRequest deleteObjectRequest =
              DeleteObjectRequest.builder().bucket(bucketName).key(key).build();
      s3.deleteObject(deleteObjectRequest);
  }

  public static Thread deleteOutputBucket() {
    if (finishedDelete) {
      System.out.println("Starting output bucket deletion thread.");
      Thread deletionThread =
          new Thread(
              () -> {
                try {
                  ListObjectsV2Request listObjectRequest =
                      ListObjectsV2Request.builder().bucket(outBucket).build();
                  ListObjectsV2Iterable listObjectResponse =
                      s3.listObjectsV2Paginator(listObjectRequest);
                  listObjectResponse.stream()
                      .forEach(
                          page ->
                              page.contents()
                                  .forEach(object -> deleteObject(outBucket, object.key())));
                  System.out.println("Output deletion thread finished.");
                } catch (Exception e) {
                  System.err.println("Delete output bucket failed.");
                  e.printStackTrace();
                }
              });
      deletionThread.start();
      return deletionThread;
      } else return null;
  }

  private static boolean readUserFile() {
    File userFile = new File(USER_FILE_PATH);

    try (BufferedReader reader = new BufferedReader(new FileReader(userFile))) {
      String input = reader.readLine();
      String[] inputDetails = input.split(" ");
      inBucket = inputDetails[0];
      inputJarName = inputDetails[1];
      inputGoldStandardFile = inputDetails[2];
      if(!(new File(inputGoldStandardFile).isFile())) {
        System.err.println(
                "Invalid golden standard word pair file.");
        return false;
      }
      uploadJARAndGolden = Boolean.parseBoolean(inputDetails[3]);
      corpusPath = reader.readLine();
      outBucket = reader.readLine();
      corpusFileCount = Integer.parseInt(reader.readLine());
      if (corpusFileCount < 1 || corpusFileCount > 100) {
        System.err.println(
            "Illegal corpus file count value, valid file count values are in range 1 <= <corpus-file-count> <= 100");
        return false;
      }
      instanceCount = Integer.parseInt(reader.readLine());
      if (instanceCount <= 0 || instanceCount >= 10) {
        System.err.println(
            "Illegal instance count provided, legal instance count values are in range 0 < <instance-count> < 10");
        return false;
      }
      String[] cooccurrencesVectorsLine = reader.readLine().split("\\s+");
      createAssociationMeasures = Boolean.parseBoolean(cooccurrencesVectorsLine[0]);
      inputMeasuresBucket =
          createAssociationMeasures
              ? String.format("s3n://%s/measures_association/output/result", outBucket)
              : cooccurrencesVectorsLine[1];
      outputCooccurrenceVectors = Boolean.parseBoolean(reader.readLine());
      String[] classifier = reader.readLine().split("\\s+");
      runClassifier = Boolean.parseBoolean(classifier[0]);
      classifierOutputPath = classifier[1];
      if(classifier.length == 3) {
        classifierInputPath = classifier[2];
      } else classifierInputPath = null;
      finishedDelete = Boolean.parseBoolean(reader.readLine());
      return true;
    } catch (Exception e) {
      System.err.println(
          "\nIllegal user file format.\n" +
                  "Expected the following input file format:\n\n" +
                  "<input-bucket> <input-jar-file-name> <input-golden-standard> <upload-jar-and-golden-standard>\n" +
                  "<corpus-input-path>\n" +
                  "<output-bucket>\n" +
                  "<corpus-files-count>\n" +
                  "<worker-instance-count>\n" +
                  "<calculate-measures (if false supply measures-path)> <measures-path>\n" +
                  "<output-co-occurrence-vectors>\n" +
                  "<run-classifier> <classifier-output-path> <optional-classifier-input-path>\n" +
                  "<delete-after-finished>");
      return false;
    }
  }
}
