package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.utils.ByteSize;

public class DenseMatrix extends ArrayList<DenseVector> implements Matrix {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7866477825982661823L;

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		double[][] a = { { 0, 1 }, { 2, 3 } };

		DenseMatrix m = new DenseMatrix(a);

		System.out.println(m.toString());
		System.out.println();
	}

	private double[][] vals;

	public DenseMatrix() {

	}

	public DenseMatrix(DenseVector[] rows) {
		ensureCapacity(rows.length);
		for (DenseVector row : rows) {
			add(row);
		}
		unwrapValues();
	}

	public DenseMatrix(double[][] vals) {
		ensureCapacity(vals.length);

		for (int i = 0; i < vals.length; i++) {
			add(new DenseVector(vals[i]));
		}
		this.vals = vals;
	}

	public DenseMatrix(int size) {
		this(new double[size][size]);
	}

	public DenseMatrix(int row_size, int col_size) {
		this(new double[row_size][col_size]);
	}

	public DenseMatrix(List<DenseVector> rows) {
		this(rows.toArray(new DenseVector[rows.size()]));
	}

	public DenseMatrix(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public DenseMatrix(String fileName) throws Exception {
		readObject(fileName);
	}

	public DenseMatrix(Vector[] rows) {
		ensureCapacity(rows.length);
		for (int i = 0; i < rows.length; i++) {
			add((DenseVector) rows[i]);
		}
		unwrapValues();
	}

	@Override
	public void add(double value) {
		for (int i = 0; i < size(); i++) {
			get(i).add(value);
		}
	}

	public void add(int i, int j, double value) {
		get(i).add(j, value);
	}

	@Override
	public int[] argMax() {
		return ArrayMath.argMax(vals);
	}

	@Override
	public int[] argMin() {
		return ArrayMath.argMin(vals);
	}

	public ByteSize byteSize() {
		long bytes = 0;
		for (DenseVector row : this) {
			bytes += row.byteSize().getBytes();
		}
		return new ByteSize(bytes);
	}

	private boolean checkRange(int i) {
		boolean ret = true;
		if (i < 0 || i >= rowSize()) {
			ret = false;
		}
		return ret;
	}

	@Override
	public int colSize() {
		return get(0).size();
	}

	@Override
	public DenseVector column(int j) {
		DenseVector ret = new DenseVector(rowSize());
		for (int i = 0; i < rowSize(); i++) {
			ret.add(i, row(i).value(j));

		}
		return ret;
	}

	public DenseMatrix copy() {
		return copy(false);
	}

	public DenseMatrix copy(boolean shallow_copy) {
		DenseVector[] rows = new DenseVector[rowSize()];
		for (int i = 0; i < rowSize(); i++) {
			rows[i] = row(i).copy(shallow_copy);
		}
		return new DenseMatrix(rows);
	}

	@Override
	public int indexAt(int loc) {
		return loc;
	}

	@Override
	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[DenseMatrix Info]\n");
		sb.append(String.format("rows:\t[%d]\n", rowSize()));
		sb.append(String.format("cols:\t[%d]\n", colSize()));
		sb.append(String.format("size:\t[%d]\n", sizeOfEntries()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	@Override
	public double max() {
		return ArrayMath.max(vals);
	}

	@Override
	public double min() {
		return ArrayMath.min(vals);
	}

	@Override
	public void multiply(double factor) {
		for (int i = 0; i < size(); i++) {
			get(i).multiply(factor);
		}
	}

	@Override
	public void normalizeColumns() {
		DenseVector colSums = sumColumns();
		for (int i = 0; i < rowSize(); i++) {
			DenseVector row = row(i);
			for (int j = 0; j < row.size(); j++) {
				double sum = colSums.value(j);
				if (sum != 0) {
					row.multiply(j, 1f / sum);
				}
			}
		}
	}

	@Override
	public void normalizeRows() {
		for (int i = 0; i < rowSize(); i++) {
			row(i).normalizeAfterSummation();
		}
	}

	@Override
	public double prob(int i, int j) {
		return row(i).prob(j);
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ensureCapacity(size);
		for (int i = 0; i < size; i++) {
			add(new DenseVector(ois));
		}
		unwrapValues();
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	@Override
	public DenseVector row(int i) {
		checkRange(i);
		return get(i);
	}

	@Override
	public DenseVector rowAt(int loc) {
		return row(loc);
	}

	@Override
	public int[] rowIndexes() {
		new UnsupportedOperationException("unsupported");
		return null;
	}

	@Override
	public DenseMatrix rows(int size) {
		return new DenseMatrix(rows(0, size));
	}

	@Override
	public DenseMatrix rows(int start, int size) {
		DenseVector[] ret = new DenseVector[size];
		for (int i = start, j = 0; i < start + size; i++, j++) {
			ret[j] = get(i);
		}
		return new DenseMatrix(ret);
	}

	@Override
	public DenseMatrix rows(int[] is) {
		DenseVector[] ret = new DenseVector[is.length];
		for (int i = 0; i < is.length; i++) {
			int loc = is[i];
			ret[i] = get(loc);
		}
		return new DenseMatrix(ret);
	}

	@Override
	public int rowSize() {
		return size();
	}

	public void set(double value) {
		for (DenseVector row : this) {
			row.setAll(value);
		}
	}

	@Override
	public void set(int i, int j, double value) {
		get(i).set(j, value);
	}

	@Override
	public void setAll(double value) {
		ArrayUtils.setAll(vals, value);
		// ArrayUtils.setAllByThreads(vals, value, ArrayUtils.THREAD_SIZE);
		for (DenseVector row : this) {
			row.summation();
		}
	}

	public void setColumn(int j, Vector x) {
		for (int i = 0; i < size(); i++) {
			get(i).set(j, x.value(i));
		}
	}

	@Override
	public void setRow(int i, Vector x) {
		set(i, (DenseVector) x);
		vals[i] = x.values();
	}

	@Override
	public void setRowAt(int loc, Vector x) {
		setRow(loc, x);
	}

	@Override
	public void setRows(List<Vector> rows) {
		this.clear();
		this.ensureCapacity(rows.size());

		// for (Vector row : rows) {
		// add((DenseVector) row);
		// }

		addAll((Collection<? extends DenseVector>) rows);

		unwrapValues();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (DenseVector row : this) {
			ret += row.size();
		}
		return ret;
	}

	public void setValues(double[][] vals) {
		clear();
		ensureCapacity(vals.length);

		this.vals = vals;
		for (int i = 0; i < vals.length; i++) {
			add(new DenseVector(vals[i]));
		}
	}

	@Override
	public double sum() {
		return sumRows().sum();
	}

	@Override
	public DenseVector sumColumns() {
		DenseVector ret = new DenseVector(colSize());
		for (int i = 0; i < rowSize(); i++) {
			DenseVector row = row(i);
			for (int j = 0; j < row.size(); j++) {
				ret.add(j, row.value(j));
			}
		}
		return ret;
	}

	@Override
	public DenseVector sumRows() {
		DenseVector ret = new DenseVector(rowSize());
		for (int i = 0; i < rowSize(); i++) {
			DenseVector dv = row(i);
			dv.summation();
			ret.add(i, dv.sum());
		}
		return ret;
	}

	@Override
	public void swapRows(int i, int j) {
		DenseVector a = get(i);
		DenseVector b = get(j);

		set(i, b);
		set(j, a);
	}

	public DenseVector toDenseVector() {
		int size = 0;
		for (DenseVector dv : this) {
			size += dv.size();
		}
		DenseVector ret = new DenseVector(size);
		int j = 0;
		for (DenseVector dv : this) {
			for (int i = 0; i < dv.size(); i++) {
				ret.add(j++, dv.value(i));
			}
		}
		return ret;
	}

	public String toString() {
		return toString(20, 20);
	}

	public String toString(int row_size, int col_size) {
		StringBuffer sb = new StringBuffer();
		sb.append(info() + "\n");

		for (int i = 0, j = 0; i < rowSize(); i++) {
			if (j == row_size) {
				break;
			}
			DenseVector r = get(i);

			if (r.sizeOfNonzero() > 0) {
				sb.append(String.format("%dth: %s\n", i, r.toString(col_size, true, null)));
				j++;
			}

		}

		return sb.toString().trim();
	}

	@Override
	public DenseMatrix transpose() {
		return new DenseMatrix(ArrayMath.transpose(vals));
	}

	public void unwrapValues() {
		vals = new double[size()][];
		for (int i = 0; i < vals.length; i++) {
			vals[i] = get(i).values();
		}
	}

	public double value(int i, int j) {
		return row(i).value(j);
	}

	@Override
	public double[][] values() {
		return vals;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(size());
		for (DenseVector row : this) {
			row.writeObject(oos);
		}
	}

	@Override
	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
