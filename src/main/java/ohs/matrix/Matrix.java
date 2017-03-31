package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import ohs.utils.ByteSize;

public interface Matrix extends Serializable {

	public void add(double v);

	public ByteSize byteSize();

	public int colSize();

	public double sum();

	public Vector column(int j);

	public int indexAt(int loc);

	public String info();

	public void multiply(double v);

	public void normalizeColumns();

	public void normalizeRows();

	public void swapRows(int i, int j);

	public double prob(int i, int j);

	public int[] argMax();

	public int[] argMin();

	public double max();

	public double min();

	public void readObject(ObjectInputStream ois) throws Exception;

	public void readObject(String fileName) throws Exception;

	public Vector row(int i);

	public Vector rowAt(int loc);

	public int[] rowIndexes();

	public Vector[] rows();

	public Vector[] rows(int start, int size);

	public Vector[] rows(int[] is);

	public Matrix rowsAsMatrix(int size);

	public Matrix rowsAsMatrix(int start, int size);

	public Matrix rowsAsMatrix(int[] is);

	public int rowSize();

	public void set(int i, int j, double v);

	public void setAll(double v);

	public void setRow(int i, Vector x);

	public void setRowAt(int loc, Vector x);

	public void setRows(Vector[] rows);

	public int sizeOfEntries();

	public Vector sumColumns();

	public Vector sumRows();

	public void unwrapValues();

	public double value(int i, int j);

	public double[][] values();

	public void writeObject(ObjectOutputStream oos) throws Exception;

	public void writeObject(String fileName) throws Exception;

}
