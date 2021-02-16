package measures_association;

import opennlp.tools.stemmer.PorterStemmer;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import utils.StopWordsIdentifier;
import utils.SyntacticTextUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

public class ParseSyntacticDependencies {

  /**
   * Input shape:
   *    key: <line-id>
   *    value: <head_word<TAB>syntactic-ngram<TAB>total_count<TAB>counts_by_year> (syntactic-ngram: [<word/pos-tag/dep-label/head-index>]),
   * Output shape:
   *    key: <lexeme | feature | <lexeme, feature>>
   *    value: <total_count>
   */
  public static class MapperClass extends Mapper<LongWritable, Text, Text, LongWritable> {

    private static final Pattern ENG_REGEX = Pattern.compile("[a-z-]+");
    private PorterStemmer stemmer;

    protected void setup(Context context) {
      stemmer = new PorterStemmer();
    }

    private boolean isLegalWord(String word) {
      return ENG_REGEX.matcher(word).matches() && !StopWordsIdentifier.isStopWord(word);
    }

    @Override
    public void map(LongWritable lineId, Text line, Context context)
        throws IOException, InterruptedException {
      Set<String> emittedLexemes = new HashSet<>();

      // head_word<TAB>syntactic-ngram<TAB>total_count<TAB>counts_by_year
      String[] split = line.toString().split("\t");
      String[] syntacticNgramSplits = split[1].split(" ");
      long totalCount = Long.parseLong(split[2]);
      LongWritable totalCountWritable = new LongWritable(totalCount);
      String[] words =
          Arrays.stream(syntacticNgramSplits)
              .map(s -> stemmer.stem(s.split("/")[0].toLowerCase(Locale.ROOT)))
              .toArray(String[]::new);

      for (int i = 0; i < syntacticNgramSplits.length; ++i) {
        // word/pos-tag/dep-label/head-index
        String[] ngramSplits = syntacticNgramSplits[i].split("/");

        if (ngramSplits.length == 4) {
          String featureWord = words[i];
          int headIndex = Integer.parseInt(ngramSplits[3]) - 1;
          if (headIndex < 0 || !isLegalWord(featureWord)) continue;

          // Emit lexemes count
          String lexeme = words[headIndex];
          if (!isLegalWord(lexeme)) continue;
          if (!emittedLexemes.contains(lexeme)) {
            context.write(new Text(lexeme), totalCountWritable);
            context.getCounter(CounterTypes.LEXEME_COUNTER).increment(totalCount);
            emittedLexemes.add(lexeme);
          }

          // Emit features count
          String dependency = ngramSplits[2];
          context.write(
                  SyntacticTextUtils.createFeatureText(featureWord, dependency), totalCountWritable);
          context.getCounter(CounterTypes.FEATURE_COUNTER).increment(totalCount);

          // Emit <lexeme, feature> pairs count
          context.write(
              SyntacticTextUtils.createLexemeFeatureText(lexeme, featureWord, dependency),
              totalCountWritable);
        }
      }
    }
  }

  /**
   * Input shape:
   *    key: <lexeme | feature | <lexeme, feature>>,
   *    value: value: [counts]
   * Output shape:
   *    key: <lexeme | feature | <lexeme, feature>>
   *    value: value: <count(L=lexeme) | count(F=feature) | count(F=feature, L=lexeme)>
   */
  public static class ReducerClass extends Reducer<Text, LongWritable, Text, LongWritable> {
    @Override
    public void reduce(Text syntacticPart, Iterable<LongWritable> counts, Context context)
        throws IOException, InterruptedException {
      long sum = 0;
      for (LongWritable count : counts) {
        sum += count.get();
      }
      context.write(syntacticPart, new LongWritable(sum));
    }
  }

  public static class PartitionerClass extends Partitioner<Text, LongWritable> {

    @Override
    public int getPartition(Text key, LongWritable value, int numPartitions) {
      return (key.hashCode() & 0xFFFFFFF) % numPartitions;
    }
  }
}
