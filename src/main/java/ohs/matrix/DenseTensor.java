package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.io.FileUtils;
import ohs.types.number.IntegerArray;
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

	public void add(double v) {
		for (DenseMatrix a : this) {
			a.add(v);
		}
	}

	public void add(int i, int j, double value) {
		get(i).get(j).add(value);
	}

	public ByteSize byteSize() {
		long bytes = 0;
		for (DenseMatrix a : this) {
			bytes += a.byteSize().getBytes();
		}
		return new ByteSize(bytes);
	}

	public DenseTensor copy(boolean shallow_copy) {
		DenseTensor ret = new DenseTensor();
		ret.ensureCapacity(size());
		for (int i = 0; i < size(); i++) {
			ret.add(get(i).copy(shallow_copy));
		}
		ret.unwrapValues();
		return ret;
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

		minSizes.add(min2);
		minSizes.add(min3);

		maxSizes.add(max2);
		maxSizes.add(max3);

		StringBuffer sb = new StringBuffer();
		sb.append("[DenseTensor Info]\n");
		sb.append(String.format("min size:\t(%d, %d, %d)\n", minSizes.get(0), minSizes.get(1), minSizes.get(2)));
		sb.append(String.format("max size:\t(%d, %d, %d)\n", maxSizes.get(0), maxSizes.get(1), maxSizes.get(2)));
		sb.append(String.format("entry size:\t%d\n", sizeOfEntries()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void multiply(double factor) {
		for (DenseMatrix a : this) {
			a.multiply(factor);
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

	public void setAll(double v) {
		for (DenseMatrix a : this) {
			a.setAll(v);
		}
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (DenseMatrix a : this) {
			ret += a.sizeOfEntries();
		}
		return ret;
	}

	public int sizeOfInnerVectors() {
		int ret = 0;
		for (DenseMatrix a : this) {
			ret += a.rowSize();
		}
		return ret;
	}

	public DenseTensor subTensor(int i, int j) {
		return new DenseTensor(subList(i, j));
	}

	public DenseTensor subTensor(int[] is) {
		DenseTensor ret = new DenseTensor();
		ret.ensureCapacity(is.length);

		for (int i = 0; i < is.length; i++) {
			ret.add(get(is[i]));
		}
		return ret;
	}

	public void swapRows(int i, int j) {
		DenseMatrix a = get(i);
		DenseMatrix b = get(j);
		set(i, b);
		set(j, a);
	}

	public String toString() {
		return toString(20, 20);
	}

	public String toString(int num_vecs, int vec_size) {
		StringBuffer sb = new StringBuffer();
		sb.append(info());

		boolean stop = false;

		for (int i = 0, u = 0; i < size(); i++) {
			DenseMatrix dm = get(i);

			for (int j = 0; j < dm.rowSize(); j++) {
				if (u++ == num_vecs) {
					stop = true;
					break;
				}

				DenseVector dv = dm.row(j);
				sb.append(String.format("\n(%d, %d, %d)\t[", i, j, dv.size()));

				for (int k = 0; k < dv.size(); k++) {
					double v = dv.value(k);

					if (v % 1 == 0) {
						sb.append(Integer.toString((int) v));
					} else {
						sb.append(Double.toString(dv.value(k)));
					}

					if (k == vec_size) {
						sb.append(" ...");
						break;
					}

					if (k != dv.size() - 1) {
						sb.append(" ");
					}
				}
				sb.append("]");
			}

			if (stop) {
				break;
			}
		}

		return sb.toString();
	}

	public void unwrapValues() {
		if (vals == null || vals.length != size()) {
			vals = new double[size()][][];
		}

		for (int i = 0; i < vals.length; i++) {
			vals[i] = get(i).values();
		}
	}

	public double value(int i, int j, int k) {
		return get(i).row(j).value(k);
	}

	public double[][][] values() {
		return vals;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(size());
		for (DenseMatrix m : this) {
			m.writeObject(oos);
		}
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
