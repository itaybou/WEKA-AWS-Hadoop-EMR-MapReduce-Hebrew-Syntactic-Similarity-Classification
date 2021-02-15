package similarity_vectors;

import measures_association.AssociationMeasuresWritable;
import measures_association.TextPairWritable;
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

public class CreateCooccurrenceVectors {

    /**
     * Input shape:
     *    key: <lexeme>
     *    value: <<lexeme, feature, plain-frequency, relative-frequency, pmi, t-test>>
     * Output shape:
     *    key: <lexeme, feature>
     *    value: <<lexeme, feature, plain-frequency, relative-frequency, pmi, t-test>>
     */
    public static class MapperClass extends Mapper<Text, AssociationMeasuresWritable, TextPairWritable, AssociationMeasuresWritable> {

        private Set<String> goldenStandardLexemes; // Golden Standard lexemes
        private PorterStemmer stemmer;

        @Override
        protected void setup(Context context) throws IOException {
            goldenStandardLexemes = new HashSet<>();
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
                    goldenStandardLexemes.add(stemmer.stem(lexemes[0]).toLowerCase(Locale.ROOT));
                    goldenStandardLexemes.add(stemmer.stem(lexemes[1]).toLowerCase(Locale.ROOT));
                }
            } catch (IOException e) {
                System.err.println("Error reading golden standard classification file." + e.getMessage());
            }
        }

        @Override
        public void map(Text lexeme, AssociationMeasuresWritable assocMeasurements, Context context)
                throws IOException, InterruptedException {
            if (goldenStandardLexemes.contains(lexeme.toString())) {
                context.write(
                        new TextPairWritable(lexeme, assocMeasurements.getFeature()),
                        assocMeasurements);
            }
        }
    }

    /**
     * Input shape:
     *    key: <lexeme, feature>
     *    value: <<lexeme, feature, plain-frequency, relative-frequency, pmi, t-test>>
     * Output shape:
     *    key: <lexeme, <PLAIN | RELATIVE | PMI | TTEST>>
     *    value: lexeme co-occurrence vector associated with the lexeme and the <PLAIN | RELATIVE | PMI | TTEST> measurement
     */
  public static class ReducerClass
      extends Reducer<TextPairWritable, AssociationMeasuresWritable, TextPairWritable, Text> {

      public enum VectorType {
          PLAIN,
          RELATIVE,
          PMI,
          TTEST
      }

      @Override
        public void reduce(TextPairWritable key, Iterable<AssociationMeasuresWritable> cooccurrenceVectors, Context context)
            throws IOException, InterruptedException {
            Iterator<AssociationMeasuresWritable> it = cooccurrenceVectors.iterator();
            StringBuilder plain = new StringBuilder();
            StringBuilder relative = new StringBuilder();
            StringBuilder pmi = new StringBuilder();
            StringBuilder tTest = new StringBuilder();

            while(it.hasNext()) {
                AssociationMeasuresWritable current = it.next();
                boolean hasNext = it.hasNext();
                String feature = current.getFeature().toString();
                plain.append(String.format("%s:%s", feature, current.getPlainFrequency().get())).append(hasNext ? "," : "");
                relative.append(String.format("%s:%s", feature, current.getRelativeFrequency().get())).append(hasNext ? "," : "");
                pmi.append(String.format("%s:%s", feature, current.getPmi().get())).append(hasNext ? "," : "");
                tTest.append(String.format("%s:%s", feature, current.gettTest().get())).append(hasNext ? "," : "");
            }
            context.write(new TextPairWritable(key.getFirst(), new Text(String.valueOf(VectorType.PLAIN))), new Text(plain.toString()));
            context.write(new TextPairWritable(key.getFirst(), new Text(String.valueOf(VectorType.RELATIVE))), new Text(relative.toString()));
            context.write(new TextPairWritable(key.getFirst(), new Text(String.valueOf(VectorType.PMI))), new Text(pmi.toString()));
            context.write(new TextPairWritable(key.getFirst(), new Text(String.valueOf(VectorType.TTEST))), new Text(tTest.toString()));
        }
    }

    public static class PartitionerClass extends Partitioner<TextPairWritable, AssociationMeasuresWritable> {

        @Override
        public int getPartition(TextPairWritable key, AssociationMeasuresWritable value, int numPartitions) {
            return  (key.getFirst().toString().hashCode() & 0xFFFFFFF) % numPartitions;
        }
    }

    public static class CooccurrenceVectorsComparator extends WritableComparator {

        protected CooccurrenceVectorsComparator() {
            super(TextPairWritable.class, true);
        }

        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            TextPairWritable a = (TextPairWritable) w1;
            TextPairWritable b = (TextPairWritable) w2;
            int compareLexeme = a.getFirst().toString().compareTo(b.getFirst().toString());
            return compareLexeme == 0 ? a.getSecond().toString().compareTo(b.getSecond().toString()) : compareLexeme;
        }
    }

    public static class CooccurrenceVectorsGroupingComparator extends WritableComparator {
        protected CooccurrenceVectorsGroupingComparator() {
            super(TextPairWritable.class, true);
        }

        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            TextPairWritable a = (TextPairWritable) w1;
            TextPairWritable b = (TextPairWritable) w2;
            return a.getFirst().toString().compareTo(b.getFirst().toString());
        }
    }
}
