package measures_association;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class AssociationMeasuresWritable implements WritableComparable<AssociationMeasuresWritable> {

    private Text lexeme;
    private Text feature;

    private DoubleWritable plainFrequency;
    private DoubleWritable relativeFrequency;
    private DoubleWritable pmi;
    private DoubleWritable tTest;

    public AssociationMeasuresWritable() {
        lexeme = new Text();
        feature = new Text();
        plainFrequency = new DoubleWritable();
        relativeFrequency = new DoubleWritable();
        pmi = new DoubleWritable();
        tTest = new DoubleWritable();
    }

    public AssociationMeasuresWritable(AssociationMeasuresWritable o) {
        lexeme = new Text(o.getLexeme().toString());
        feature = new Text(o.getFeature().toString());
        plainFrequency = new DoubleWritable(o.getPlainFrequency().get());
        relativeFrequency = new DoubleWritable(o.getRelativeFrequency().get());
        pmi = new DoubleWritable(o.getPmi().get());
        tTest = new DoubleWritable(o.gettTest().get());
    }

    public AssociationMeasuresWritable(String lexeme, String feature, long plainFrequency, double relativeFrequency, double pmi, double tTest) {
        this.lexeme = new Text(lexeme);
        this.feature = new Text(feature);
        this.plainFrequency = new DoubleWritable(plainFrequency);
        this.relativeFrequency = new DoubleWritable(relativeFrequency);
        this.pmi = new DoubleWritable(pmi);
        this.tTest = new DoubleWritable(tTest);
    }

    @Override
    public String toString() {
        return String.format("<%s,%s,%f,%f,%f,%f>", lexeme.toString(), feature.toString(), plainFrequency.get(), relativeFrequency.get(), pmi.get(), tTest.get());
    }

    @Override
    public int compareTo(AssociationMeasuresWritable o) {
        return feature.compareTo(o.feature);
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        lexeme.write(dataOutput);
        feature.write(dataOutput);
        plainFrequency.write(dataOutput);
        relativeFrequency.write(dataOutput);
        pmi.write(dataOutput);
        tTest.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        lexeme.readFields(dataInput);
        feature.readFields(dataInput);
        plainFrequency.readFields(dataInput);
        relativeFrequency.readFields(dataInput);
        pmi.readFields(dataInput);
        tTest.readFields(dataInput);
    }

    public Text getLexeme() {
        return lexeme;
    }

    public Text getFeature() {
        return feature;
    }

    public DoubleWritable getPlainFrequency() {
        return plainFrequency;
    }

    public DoubleWritable getRelativeFrequency() {
        return relativeFrequency;
    }

    public DoubleWritable getPmi() {
        return pmi;
    }

    public DoubleWritable gettTest() {
        return tTest;
    }
}
