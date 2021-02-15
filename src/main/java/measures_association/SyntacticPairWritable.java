package measures_association;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableUtils;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class SyntacticPairWritable implements WritableComparable<SyntacticPairWritable> {

  public enum Type {
    LEXEME,
    FEATURE,
    LEXEME_FEATURE
  }

  private Type type;
  private Text element;

  public SyntacticPairWritable() {
    type = Type.LEXEME;
    element = new Text();
  }

  public SyntacticPairWritable(Type type, String element) {
    this.type = type;
    this.element = new Text(element);
  }

  public SyntacticPairWritable(Type type, Text element) {
    this.type = type;
    this.element = element;
  }

  @Override
  public int compareTo(SyntacticPairWritable other) {
    int compareElements = element.compareTo(other.element);
    return compareElements != 0
        ? compareElements
        : type.equals(other.type)
            ? 0
            : type == Type.LEXEME_FEATURE ? 1 : -1; // ensure that <lexeme,feature> pairs come last
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    type = Type.valueOf(WritableUtils.readString(dataInput));
    element = new Text(WritableUtils.readString(dataInput));
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    WritableUtils.writeString(dataOutput, type.toString());
    WritableUtils.writeString(dataOutput, element.toString());
  }

  @Override
  public String toString() {
    return String.format("%s\t%s", type.toString(), element.toString());
  }

  @Override
  public int hashCode() {
    return element.toString().hashCode();
  }

  public Type getType() {
    return type;
  }

  public Text getElement() {
    return element;
  }
}
