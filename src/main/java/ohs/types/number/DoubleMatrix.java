package ohs.types.number;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import ohs.io.FileUtils;
import ohs.utils.ByteSize;

public class DoubleMatrix extends ArrayList<DoubleArray> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3358006286038663771L;

	public DoubleMatrix() {
	}

	public DoubleMatrix(Collection<DoubleArray> a) {
		addAll(a);
	}

	public DoubleMatrix(double[][] a) {
		ensureCapacity(a.length);
		for (int i = 0; i < a.length; i++) {
			add(new DoubleArray(a[i]));
		}
	}

	public DoubleMatrix(int size) {
		super(size);
	}

	public DoubleMatrix(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public void add(int i, double v) {
		ensure(i).add(v);
	}

	public ByteSize byteSize() {
		return new ByteSize(Double.BYTES * sizeOfEntries());
	}

	public DoubleMatrix clone() {
		DoubleMatrix ret = new DoubleMatrix(size());
		for (DoubleArray a : this) {
			ret.add(a.clone());
		}
		return ret;
	}

	public double[][] elementData() {
		double[][] ret = new double[size()][];
		for (int i = 0; i < size(); i++) {
			ret[i] = get(i).values();
		}
		return ret;
	}

	public DoubleArray ensure(int i) {
		if (i >= size()) {
			int new_size = (i - size()) + 1;
			for (int k = 0; k < new_size; k++) {
				add(new DoubleArray());
			}
		}
		return get(i);
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[Info]\n");
		sb.append(String.format("size:\t%d\n", size()));
		sb.append(String.format("size of entries:\t%d\n", sizeOfEntries()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ensureCapacity(size);
		for (int i = 0; i < size; i++) {
			add(new DoubleArray(ois));
		}
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (DoubleArray a : this) {
			ret += a.size();
		}
		return ret;
	}

	public DoubleMatrix subMatrix(int i, int j) {
		return new DoubleMatrix(subList(i, j));
	}

	public DoubleArray toDoubleArray() {
		DoubleArray ret = new DoubleArray(sizeOfEntries());
		for (DoubleArray a : this) {
			ret.addAll(a);
		}
		return ret;
	}

	public String toString() {
		return toString(20);
	}

	public String toString(int size) {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("size: %d", size()));
		int min_size = Math.min(size, size());
		for (int i = 0; i < min_size; i++) {
			sb.append(String.format("\n%d: %s", i, get(i)));
		}

		return sb.toString();
	}

	public void trimToSize() {
		super.trimToSize();

		for (DoubleArray a : this) {
			a.trimToSize();
		}
	}

	public double[][] values() {
		double[][] ret = new double[size()][];
		for (int i = 0; i < size(); i++) {
			ret[i] = get(i).values();
		}
		return ret;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(size());
		for (int i = 0; i < size(); i++) {
			get(i).writeObject(oos);
		}
	}

	public void writeObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

}
