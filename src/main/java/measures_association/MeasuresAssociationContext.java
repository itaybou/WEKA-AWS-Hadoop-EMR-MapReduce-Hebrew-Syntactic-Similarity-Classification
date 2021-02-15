package measures_association;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import utils.SyntacticTextUtils;

import java.io.IOException;

public class MeasuresAssociationContext {

    /**
     * Input shape:
     *    key: <<FEATURE, feature> | <LEXEME_FEATURE, feature>>
     *    value: <count(F=feature) | <lexeme, count(F=feature, L=lexeme), count(L=lexeme)>>
     * Output shape:
     *    key: <<FEATURE, feature> | <LEXEME_FEATURE, feature>>
     *    value: <<'F', count(F=feature)> | <'LF', <lexeme, count(F=feature, L=lexeme), count(L=lexeme)>>>
     */
    public static class MapperClass
            extends Mapper<SyntacticPairWritable, Text, SyntacticPairWritable, TextPairWritable> {

        @Override
        public void map(SyntacticPairWritable feature, Text featureInfo, Context context)
                throws IOException, InterruptedException {
            context.write(feature,
                    new TextPairWritable(
                            feature.getType() == SyntacticPairWritable.Type.FEATURE ? "F" : "LF",
                            featureInfo));
        }
    }

    /**
     * Input shape:
     *    key: <<FEATURE, feature> | <LEXEME_FEATURE, feature>>
     *    value: <<'F', count(F=feature)> | <'LF', <lexeme, count(F=feature, L=lexeme), count(L=lexeme)>>>
     * Output shape:
     *    key: <lexeme>
     *    value: <<lexeme, feature, plain-frequency, relative-frequency, pmi, t-test>>
     */
    public static class ReducerClass
            extends Reducer<SyntacticPairWritable, TextPairWritable, Text, AssociationMeasuresWritable> {

        private long totalLexemeCounter; // count(L)
        private long totalFeatureCounter; // count(F)

        protected void setup(Context context) {
            totalLexemeCounter = context.getConfiguration().getLong("countL", 0);
            totalFeatureCounter = context.getConfiguration().getLong("countF", 0);
        }

        private AssociationMeasuresWritable calculateAssociationMeasures(String lexeme, String feature, long countLexemeFeature, long countLexeme, long currentFeatureCount) {
            // count(L=lexeme, F=feature)
          long plainFrequency = countLexemeFeature;
            // P(feature|lexeme) = count(L=lexeme, F=feature) / count(L=lexeme)
          double relativeFrequency = ((double) countLexemeFeature / countLexeme);

            // P(lexeme, feature) = count(L=lexeme, F=feature) / count(L)
          double jointProbability = ((double) countLexemeFeature / totalLexemeCounter);
            // P(L=lexeme) = count(L=lexeme) / count(L)
          double lexemeProbability = ((double) countLexeme / totalLexemeCounter);
            // P(F=feature) = count(F=feature) / count(F)
          double featureProbability = ((double) currentFeatureCount / totalFeatureCounter);

            // pointwise mutual information = log_2(P(lexeme, feature) / P(L=lexeme) * P(F=feature))
          double pmi = Math.log(jointProbability / (lexemeProbability * featureProbability)) / Math.log(2);
            // t-test statistic = (P(lexeme, feature) - (P(L=lexeme) * P(F=feature))) / sqrt(P(L=lexeme) * P(F=feature))
          double tTest = ((jointProbability - (lexemeProbability * featureProbability)) / Math.sqrt((lexemeProbability * featureProbability)));

          return new AssociationMeasuresWritable(lexeme, feature, plainFrequency, relativeFrequency, pmi, tTest);
        }

        @Override
        public void reduce(
                SyntacticPairWritable feature, Iterable<TextPairWritable> featureInfo, Context context) throws IOException, InterruptedException {
            Long currentFeatureCount = null; // count(F=feature)
            for (TextPairWritable info : featureInfo) {
                String tag = info.getFirst().toString();
                if(tag.equals("F")) {
                  currentFeatureCount = Long.parseLong(info.getSecond().toString());
                }
                else if(currentFeatureCount != null) {
                  String[] triplet = SyntacticTextUtils.splitPairTriplet(info.getSecond());
                  String lexeme = triplet[0];
                  long countLexemeFeature = Long.parseLong(triplet[1]);
                  long countLexeme = Long.parseLong(triplet[2]);
                  context.write(new Text(lexeme), calculateAssociationMeasures(lexeme, feature.getElement().toString(), countLexemeFeature, countLexeme, currentFeatureCount));
                }
            }
        }
    }

    public static class PartitionerClass extends Partitioner<SyntacticPairWritable, TextPairWritable> {

        /** <LEXEME_FEATURE, feature> and <FEATURE, feature> will arrive to same reducer */
        @Override
        public int getPartition(SyntacticPairWritable key, TextPairWritable value, int numPartitions) {
            return (key.getElement().hashCode() & 0xFFFFFFF) % numPartitions;
        }
    }

    /** Sort the keys so that <FEATURE> tags will arrive before <LEXEME_FEATURE> tag for every equal feature */
    public static class MeasuresAssociationContextComparator extends WritableComparator {

        protected MeasuresAssociationContextComparator() {
            super(SyntacticPairWritable.class, true);
        }

        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            SyntacticPairWritable a = (SyntacticPairWritable) w1;
            SyntacticPairWritable b = (SyntacticPairWritable) w2;
            return a.compareTo(b);
        }
    }

    public static class MeasuresAssociationContextGroupingComparator extends WritableComparator {
        protected MeasuresAssociationContextGroupingComparator() {
            super(SyntacticPairWritable.class, true);
        }

        @Override
        public int compare(WritableComparable w1, WritableComparable w2) {
            SyntacticPairWritable a = (SyntacticPairWritable) w1;
            SyntacticPairWritable b = (SyntacticPairWritable) w2;
            return a.getElement().toString().compareTo(b.getElement().toString());
        }
    }
}
