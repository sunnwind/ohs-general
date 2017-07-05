package ohs.types.number;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import ohs.io.FileUtils;
import ohs.utils.ByteSize;

public class DoubleArrayMatrix extends ArrayList<DoubleArray> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3358006286038663771L;

	public DoubleArrayMatrix(Collection<DoubleArray> a) {
		addAll(a);
	}

	public DoubleArrayMatrix(double[][] a) {
		ensureCapacity(a.length);
		for (int i = 0; i < a.length; i++) {
			add(new DoubleArray(a[i]));
		}
	}

	public DoubleArrayMatrix(int size) {
		super(size);
	}

	public DoubleArrayMatrix(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public ByteSize byteSize() {
		return new ByteSize(Double.BYTES * sizeOfEntries());
	}

	public DoubleArrayMatrix clone() {
		DoubleArrayMatrix ret = new DoubleArrayMatrix(size());
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

	public DoubleArrayMatrix subMatrix(int i, int j) {
		return new DoubleArrayMatrix(subList(i, j));
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
