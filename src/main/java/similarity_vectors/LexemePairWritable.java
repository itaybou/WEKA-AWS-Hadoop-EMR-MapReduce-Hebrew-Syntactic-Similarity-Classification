package similarity_vectors;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Locale;

public class LexemePairWritable implements WritableComparable<LexemePairWritable> {
  private Text pair;
  private Text lexeme;
  private Text feature;

  public LexemePairWritable() {
    this.pair = new Text();
    this.lexeme = new Text();
    this.feature = new Text();
  }

  public LexemePairWritable(Text pair, Text lexeme, Text feature) {
    this.pair = pair;
    this.lexeme = lexeme;
    this.feature = feature;
  }

  @Override
  public int compareTo(LexemePairWritable other) {
    int comparePair =
        pair.toString()
            .toLowerCase(Locale.ROOT)
            .compareTo(other.pair.toString().toLowerCase(Locale.ROOT));
    int compareFeature = feature.toString().compareTo(other.feature.toString());
    return comparePair == 0 ? compareFeature == 0 ? lexeme.toString().compareTo(other.lexeme.toString()) : compareFeature : comparePair;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    pair.write(dataOutput);
    lexeme.write(dataOutput);
    feature.write(dataOutput);
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    pair.readFields(dataInput);
    lexeme.readFields(dataInput);
    feature.readFields(dataInput);
  }

  @Override
  public int hashCode() {
    return pair.toString().toLowerCase(Locale.ROOT).hashCode();
  }

  @Override
  public String toString() {
    return String.format("%s\t%s\t%s", pair.toString(), lexeme.toString(), feature.toString());
  }

  public Text getPair() {
    return pair;
  }

  public Text getLexeme() {
    return lexeme;
  }

  public Text getFeature() {
    return feature;
  }
}
