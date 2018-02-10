package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ohs.io.FileUtils;
import ohs.types.number.IntegerArray;
import ohs.utils.ByteSize;

public class DenseTensor4D extends ArrayList<DenseTensor> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7866477825982661823L;

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
	}

	private double[][][][] vals;

	public DenseTensor4D() {

	}

	public DenseTensor4D(DenseTensor[] a) {
		this(Arrays.asList(a));
	}

	public DenseTensor4D(double[][][][] vals) {
		ensureCapacity(vals.length);

		for (int i = 0; i < vals.length; i++) {
			add(new DenseTensor(vals[i]));
		}
		this.vals = vals;
	}

	public DenseTensor4D(int size) {
		this(new double[size][size][size][size]);
	}

	public DenseTensor4D(int size1, int size2, int size3, int size4) {
		this(new double[size1][size2][size3][size4]);
	}

	public DenseTensor4D(List<DenseTensor> a) {
		ensureCapacity(a.size());
		for (int i = 0; i < a.size(); i++) {
			add(a.get(i));
		}
		unwrapValues();
	}

	public DenseTensor4D(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public DenseTensor4D(String fileName) throws Exception {
		readObject(fileName);
	}

	public void add(double v) {
		for (DenseTensor a : this) {
			a.add(v);
		}
	}

	public void add(int i, int j, double value) {
		get(i).get(j).add(value);
	}

	public ByteSize byteSize() {
		long bytes = 0;
		for (DenseTensor a : this) {
			bytes += a.byteSize().getBytes();
		}
		return new ByteSize(bytes);
	}

	public DenseTensor4D copy(boolean shallow_copy) {
		DenseTensor4D ret = new DenseTensor4D();
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

		int size = size();
		int min2 = Integer.MAX_VALUE;
		int min3 = Integer.MAX_VALUE;
		int min4 = Integer.MAX_VALUE;

		int max2 = -Integer.MAX_VALUE;
		int max3 = -Integer.MAX_VALUE;
		int max4 = -Integer.MAX_VALUE;

		for (DenseTensor a : this) {
			min2 = Math.min(min2, a.size());
			max2 = Math.max(max2, a.size());

			for (DenseMatrix b : a) {
				min3 = Math.min(min3, b.size());
				max3 = Math.max(max3, b.size());

				for (DenseVector c : b) {
					min4 = Math.min(min4, c.size());
					max4 = Math.max(max4, c.size());
				}
			}
		}

		StringBuffer sb = new StringBuffer();
		sb.append("[DenseTensor Info]\n");
		sb.append(String.format("min size:\t(%d, %d, %d, %d)\n", size, min2, min3, min4));
		sb.append(String.format("max size:\t(%d, %d, %d, %d)\n", size, max2, max3, max4));
		sb.append(String.format("entry size:\t%d\n", sizeOfEntries()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void multiply(double factor) {
		for (DenseTensor a : this) {
			a.multiply(factor);
		}
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ensureCapacity(size);
		for (int i = 0; i < size; i++) {
			add(new DenseTensor(ois));
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
		for (DenseTensor a : this) {
			a.setAll(v);
		}
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (DenseTensor a : this) {
			ret += a.sizeOfEntries();
		}
		return ret;
	}

	public int sizeOfInnerVectors() {
		int ret = 0;
		for (DenseTensor a : this) {
			ret += a.sizeOfInnerVectors();
		}
		return ret;
	}

	public DenseTensor4D subTensor(int[] is) {
		DenseTensor4D ret = new DenseTensor4D();
		ret.ensureCapacity(is.length);

		for (int i = 0; i < is.length; i++) {
			ret.add(get(is[i]));
		}
		return ret;
	}

	public DenseTensor4D subTensor4D(int i, int j) {
		return new DenseTensor4D(subList(i, j));
	}

	public void swapRows(int i, int j) {
		DenseTensor a = get(i);
		DenseTensor b = get(j);
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
			DenseTensor a = get(i);
			for (int j = 0; j < a.size(); j++) {
				DenseMatrix b = a.get(j);
				for (int k = 0; k < b.rowSize(); k++) {
					if (u++ == num_vecs) {
						stop = true;
						break;
					}
					DenseVector dv = b.row(k);
					sb.append(String.format("\n(%d, %d, %d, %d)\t[", i, j, k, dv.size()));

					for (int l = 0; l < dv.size(); l++) {
						double v = dv.value(l);

						if (v % 1 == 0) {
							sb.append(Integer.toString((int) v));
						} else {
							sb.append(Double.toString(dv.value(l)));
						}

						if (l == vec_size) {
							sb.append(" ...");
							break;
						}

						if (l != dv.size() - 1) {
							sb.append(" ");
						}
					}
					sb.append("]");
				}

				if (stop) {
					break;
				}
			}
			if (stop) {
				break;
			}
		}

		return sb.toString();
	}

	public void unwrapValues() {
		if (vals == null || vals.length != size()) {
			vals = new double[size()][][][];
		}

		for (

				int i = 0; i < vals.length; i++) {
			vals[i] = get(i).values();
		}
	}

	public double value(int i, int j, int k, int l) {
		return get(i).get(j).value(k, l);
	}

	public double[][][][] values() {
		return vals;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(size());
		for (DenseTensor m : this) {
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
