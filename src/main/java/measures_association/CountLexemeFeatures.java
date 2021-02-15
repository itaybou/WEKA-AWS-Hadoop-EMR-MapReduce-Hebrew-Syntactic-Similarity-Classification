package measures_association;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import utils.SyntacticTextUtils;

import java.io.IOException;

public class CountLexemeFeatures {

  /**
   * Input shape:
   *    key: <lexeme | feature | <lexeme, feature>>
   *    value: value: <count(L=lexeme) | count(F=feature) | count(F=feature, L=lexeme)>
   * Output shape:
   *    key: <<LEXEME, lexeme> | <FEATURE, feature> | <LEXEME_FEATURE, lexeme>>
   *    value: <<'L', count(L=lexeme)> | <'F', count(F=feature)> | <'LF', <feature, count(F=feature, L=lexeme)>>>
   */
  public static class MapperClass
      extends Mapper<Text, LongWritable, SyntacticPairWritable, TextPairWritable> {

    @Override
    public void map(Text syntacticPart, LongWritable totalSum, Context context)
        throws IOException, InterruptedException {
      if (SyntacticTextUtils.isPair(syntacticPart)) {
        String[] pair = SyntacticTextUtils.splitPairTriplet(syntacticPart);
        String lexeme = pair[0];
        String feature = pair[1];
        context.write( // Emit <LEXEME_FEATURE, lexeme> -> <'LF', <feature, count(L=lexeme, F=feature)>>
            new SyntacticPairWritable(SyntacticPairWritable.Type.LEXEME_FEATURE, lexeme),
            new TextPairWritable(
                "LF", SyntacticTextUtils.createCountText(feature, totalSum.get())));
      } else if (SyntacticTextUtils.isFeature(syntacticPart)) {
        context.write( // Emit <FEATURE, feature> -> <'F', count(F=feature)>
            new SyntacticPairWritable(SyntacticPairWritable.Type.FEATURE, syntacticPart),
            new TextPairWritable("F", totalSum.toString()));
      } else {
        context.write( // Emit <LEXEME, lexeme> -> <'L', count(L=lexeme)>
            new SyntacticPairWritable(SyntacticPairWritable.Type.LEXEME, syntacticPart),
            new TextPairWritable("L", totalSum.toString()));
      }
    }
  }

  /**
   * Input shape:
   *    key: <<LEXEME, lexeme> | <FEATURE, feature> | <LEXEME_FEATURE, lexeme>>
   *    value: <<'L', count(L=lexeme)> | <'F', count(F=feature)> | <'LF', <feature, count(L=lexeme, F=feature)>>>
   * Output shape:
   *    key: <<FEATURE, feature> | <LEXEME_FEATURE, feature>>
   *    value: <count(F=feature) | <lexeme, count(F=feature, L=lexeme), count(L=lexeme)>>
   */
  public static class ReducerClass
      extends Reducer<SyntacticPairWritable, TextPairWritable, SyntacticPairWritable, Text> {

    @Override
    public void reduce(
        SyntacticPairWritable syntacticPart, Iterable<TextPairWritable> countPairs, Context context)
        throws IOException, InterruptedException {
      Long currentLexemeCount = null; // count(L=lexeme)

      for (TextPairWritable countPair : countPairs) {
        String tag = countPair.getFirst().toString();
        switch (tag) {
          case "L":
            currentLexemeCount = Long.parseLong(countPair.getSecond().toString());
            break;
          case "F":
            context.write( // Emit key: feature, value: count(F=feature)
                new SyntacticPairWritable(
                    SyntacticPairWritable.Type.FEATURE, syntacticPart.getElement()),
                countPair.getSecond());
            break;
          case "LF":
            if (currentLexemeCount != null) {
              String[] pair = SyntacticTextUtils.splitPairTriplet(countPair.getSecond());
              String feature = pair[0];
              long countLexemeFeature = Long.parseLong(pair[1]);
              context.write( // Emit key: feature, value: <lexeme, count(F=feature, L=lexeme), count(L=lexeme)>
                      new SyntacticPairWritable(SyntacticPairWritable.Type.LEXEME_FEATURE, feature),
                      SyntacticTextUtils.createLexemeFeatureTripletText(
                          syntacticPart.getElement().toString(),
                          countLexemeFeature,
                          currentLexemeCount));
            }
            break;
        }
      }
    }
  }

  public static class PartitionerClass extends Partitioner<SyntacticPairWritable, TextPairWritable> {

    /** <LEXEME_FEATURE, lexeme> and <LEXEME, lexeme> will arrive to same reducer */
    @Override
    public int getPartition(SyntacticPairWritable key, TextPairWritable value, int numPartitions) {
      return (key.getElement().hashCode() & 0xFFFFFFF) % numPartitions;
    }
  }

  /** Sort the keys so that <LEXEME | FEATURE> tags will arrive before <LEXEME_FEATURE> tag for every equal lexeme */
  public static class CountLexemeFeaturesComparator extends WritableComparator {

    protected CountLexemeFeaturesComparator() {
      super(SyntacticPairWritable.class, true);
    }

    @Override
    public int compare(WritableComparable w1, WritableComparable w2) {
      SyntacticPairWritable a = (SyntacticPairWritable) w1;
      SyntacticPairWritable b = (SyntacticPairWritable) w2;
      return a.compareTo(b);
    }
  }

  public static class CountLexemeFeaturesGroupingComparator extends WritableComparator {
    protected CountLexemeFeaturesGroupingComparator() {
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
