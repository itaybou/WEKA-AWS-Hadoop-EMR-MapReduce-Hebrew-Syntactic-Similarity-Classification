package similarity_vectors;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

public class SimilarityVectorWritable implements WritableComparable<SimilarityVectorWritable> {

    public final static int VECTOR_SIZE = 24;

    boolean similar;
    double[] similarityVector;
    String pair;

    public SimilarityVectorWritable() {
        similar = false;
        similarityVector = new double[VECTOR_SIZE];
        pair = null;
    }

    public SimilarityVectorWritable(boolean similar, double[] similarityVector, String pair) {
        this.similar = similar;
        this.similarityVector = similarityVector;
        this.pair = pair;

//        for(int i = 0; i < VECTOR_SIZE; ++i) {
//            if(Double.isNaN(similarityVector[i])) {
//                similarityVector[i] = 0;
//            }
//        }
    }

    @Override
    public String toString() {
    return String.format("%s\t%b", Arrays.toString(similarityVector), similar);
    }

    @Override
    public int compareTo(SimilarityVectorWritable o) {
        return pair.compareTo(o.pair);
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeBoolean(similar);
        for(double vectorPosition : similarityVector) {
            dataOutput.writeDouble(vectorPosition);
        }
        dataOutput.writeChars(pair);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        double[] vector = new double[VECTOR_SIZE];
        similar = dataInput.readBoolean();
        for(int i = 0; i < VECTOR_SIZE; ++i) {
            vector[i] = dataInput.readDouble();
        }
        similarityVector = vector;
        pair = dataInput.readUTF();
    }
}
