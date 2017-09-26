package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.ByteSize;

public class DenseTensor extends ArrayList<DenseMatrix> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7866477825982661823L;

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
	}

	private double[][][] vals;

	public DenseTensor() {

	}

	public DenseTensor(DenseMatrix[] rows) {
		ensureCapacity(rows.length);
		for (DenseMatrix row : rows) {
			add(row);
		}
		unwrapValues();
	}

	public DenseTensor(double[][][] vals) {
		ensureCapacity(vals.length);

		for (int i = 0; i < vals.length; i++) {
			add(new DenseMatrix(vals[i]));
		}
		this.vals = vals;
	}

	public DenseTensor(int size) {
		this(new double[size][size][size]);
	}

	public DenseTensor(int size1, int size2, int size3) {
		this(new double[size1][size2][size3]);
	}

	public DenseTensor(List<DenseMatrix> rows) {
		this(rows.toArray(new DenseMatrix[rows.size()]));
	}

	public DenseTensor(Matrix[] rows) {
		ensureCapacity(rows.length);
		for (int i = 0; i < rows.length; i++) {
			add((DenseMatrix) rows[i]);
		}
		unwrapValues();
	}

	public DenseTensor(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public DenseTensor(String fileName) throws Exception {
		readObject(fileName);
	}

	public void add(double value) {
		for (int i = 0; i < size(); i++) {
			get(i).add(value);
		}
	}

	public void add(int i, int j, double value) {
		get(i).get(j).add(value);
	}

	public ByteSize byteSize() {
		long bytes = 0;
		for (DenseMatrix row : this) {
			bytes += row.byteSize().getBytes();
		}
		return new ByteSize(bytes);
	}

	public int colSize() {
		return get(0).size();
	}

	public int indexAt(int loc) {
		return loc;
	}

	public String info() {
		IntegerArray minSizes = new IntegerArray(3);
		IntegerArray maxSizes = new IntegerArray(3);

		minSizes.add(size());
		maxSizes.add(size());

		int min2 = Integer.MAX_VALUE;
		int min3 = Integer.MAX_VALUE;

		int max2 = -Integer.MAX_VALUE;
		int max3 = -Integer.MAX_VALUE;

		for (DenseMatrix a : this) {
			min2 = Math.min(min2, a.size());
			max2 = Math.max(max2, a.size());

			for (DenseVector b : a) {
				min3 = Math.min(min3, b.size());
				max3 = Math.max(max3, b.size());
			}
		}

		StringBuffer sb = new StringBuffer();
		sb.append("[DenseTensor Info]\n");
		sb.append(String.format("min size:\t%d\n", minSizes.toString()));
		sb.append(String.format("max size:\t%d\n", maxSizes.toString()));
		sb.append(String.format("entry size:\t%d\n", sizeOfEntries()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void multiply(double factor) {
		for (int i = 0; i < size(); i++) {
			get(i).multiply(factor);
		}
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ensureCapacity(size);
		for (int i = 0; i < size; i++) {
			add(new DenseMatrix(ois));
		}
		unwrapValues();
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public DenseMatrix row(int i) {
		return get(i);
	}

	public DenseTensor rows(int start, int size) {
		DenseMatrix[] ret = new DenseMatrix[size];
		for (int i = start, j = 0; i < start + size; i++, j++) {
			ret[j] = get(i);
		}
		return new DenseTensor(ret);
	}

	public int rowSize() {
		return size();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (DenseMatrix row : this) {
			ret += row.sizeOfEntries();
		}
		return ret;
	}

	public void swapRows(int i, int j) {
		DenseMatrix a = get(i);
		DenseMatrix b = get(j);

		set(i, b);
		set(j, a);
	}

	public void unwrapValues() {
		vals = new double[size()][][];
		for (int i = 0; i < vals.length; i++) {
			vals[i] = get(i).values();
		}
	}

	public double value(int i, int j, int k) {
		return row(i).row(j).value(k);
	}

	public double[][][] values() {
		return vals;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(size());
		for (DenseMatrix row : this) {
			row.writeObject(oos);
		}
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
