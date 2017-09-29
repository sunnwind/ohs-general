package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;

import ohs.utils.ByteSize;

public interface Matrix extends Serializable {

	public void add(double v);

	public int[] argMax();

	public int[] argMin();

	public ByteSize byteSize();

	public int colSize();

	public Vector column(int j);

	public int indexAt(int loc);

	public String info();

	public double max();

	public double min();

	public void multiply(double v);

	public void normalizeColumns();

	public void normalizeRows();

	public double prob(int i, int j);

	public void readObject(ObjectInputStream ois) throws Exception;

	public void readObject(String fileName) throws Exception;

	public Vector row(int i);

	public Vector rowAt(int loc);

	public int[] rowIndexes();

	public Matrix rows(int size);

	public Matrix rows(int start, int size);

	public Matrix rows(int[] is);

	public int rowSize();

	public void set(int i, int j, double v);

	public void setAll(double v);

	public void setRow(int i, Vector x);

	public void setRowAt(int loc, Vector x);

	public void setRows(List<Vector> rows);

	public int sizeOfEntries();

	public double sum();

	public Vector sumColumns();

	public Vector sumRows();

	public void swapRows(int i, int j);

	public Matrix transpose();

	public void unwrapValues();

	public double value(int i, int j);

	public double[][] values();

	public void writeObject(ObjectOutputStream oos) throws Exception;

	public void writeObject(String fileName) throws Exception;

}
