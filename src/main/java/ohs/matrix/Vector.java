package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;
import java.util.Set;

import ohs.types.generic.Pair;
import ohs.utils.ByteSize;

/**
 * @author Heung-Seon Oh
 * 
 */

public interface Vector extends Serializable {

	public void add(double v);

	public void add(int i, double v);

	public void addAt(int loc, double v);

	public void addAt(int loc, int i, double v);

	public int argMax();

	public int argMin();

	public ByteSize byteSize();

	public Vector copy();

	public double[] copyValues();

	public int indexAt(int loc);

	public int[] indexes();

	public String info();

	public void keepAbove(double cutoff);

	public void keepTopN(int topN);

	public int location(int i);

	public double max();

	public double min();

	public double multiply(double factor);

	public double multiply(int i, double factor);

	public double multiplyAt(int loc, double factor);

	public double multiplyAt(int loc, int i, double factor);

	public double normalize();

	public double normalizeAfterSummation();

	public List<Pair<Integer, Double>> pairs();

	public double prob(int index);

	public double probAt(int loc);

	public void prune(final Set<Integer> toRemove);

	public void pruneExcept(final Set<Integer> toKeep);

	public Vector ranking();

	public void readObject(ObjectInputStream ois) throws Exception;

	public void readObject(String fileName) throws Exception;

	public double set(int i, double v);

	public void setAll(double v);

	public double setAt(int loc, double v);

	public double setAt(int loc, int i, double v);

	public void setIndexes(int[] is);

	public double setSum(double sum);

	public void setValues(double[] vs);

	public int size();

	public int sizeOfNonzero();

	public Vector subVector(int size);

	public Vector subVector(int start, int size);

	public Vector subVector(int[] is);

	public double sum();

	public double summation();

	public double value(int i);

	public double valueAt(int loc);

	public double[] values();

	public void writeObject(ObjectOutputStream oos) throws Exception;

	public void writeObject(String fileName) throws Exception;

}
