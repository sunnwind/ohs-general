package ohs.types.number;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import ohs.io.FileUtils;
import ohs.utils.ByteSize;

public class IntegerArrayMatrix extends ArrayList<IntegerArray> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1810717124466759948L;

	public IntegerArrayMatrix() {
		super();
	}

	public IntegerArrayMatrix(Collection<IntegerArray> a) {
		addAll(a);
	}

	public IntegerArrayMatrix(int size) {
		super(size);
	}

	public IntegerArrayMatrix(int[][] a) {
		ensureCapacity(a.length);
		for (int i = 0; i < a.length; i++) {
			add(new IntegerArray(a[i]));
		}
	}

	public IntegerArrayMatrix(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public void add(int i, int j) {
		ensure(i);
		get(i).add(j);
	}

	public ByteSize byteSize() {
		return new ByteSize(Integer.BYTES * sizeOfEntries());
	}

	public IntegerArrayMatrix clone() {
		IntegerArrayMatrix ret = new IntegerArrayMatrix(size());
		for (IntegerArray a : this) {
			ret.add(a.clone());
		}
		return ret;
	}

	public void ensure(int i) {
		if (i >= size()) {
			int new_size = (i - size()) + 1;
			for (int k = 0; k < new_size; k++) {
				add(new IntegerArray());
			}
		}
	}

	public String info() {
		int min = Integer.MAX_VALUE;
		int max = -Integer.MAX_VALUE;

		for (IntegerArray a : this) {
			min = Math.min(min, a.size());
			max = Math.max(max, a.size());
		}

		StringBuffer sb = new StringBuffer();
		sb.append("[Info]\n");
		sb.append(String.format("size:\t%d\n", size()));
		sb.append(String.format("entry size:\t%d\n", sizeOfEntries()));
		sb.append(String.format("min array size:\t%d\n", min));
		sb.append(String.format("max array size:\t%d\n", max));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ensureCapacity(size);
		for (int i = 0; i < size; i++) {
			add(new IntegerArray(ois));
		}
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (IntegerArray a : this) {
			ret += a.size();
		}
		return ret;
	}

	public IntegerArrayMatrix subMatrix(int i, int j) {
		return new IntegerArrayMatrix(subList(i, j));
	}

	public String toString() {
		return toString(30);
	}

	public String toString(int size) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size(); i++) {
			if (i == size) {
				sb.append(String.format("\n..."));
				break;
			}
			sb.append(String.format("\n%d: %s", i, get(i).toString()));
		}
		return sb.toString();
	}

	public void trimToSize() {
		super.trimToSize();

		for (IntegerArray a : this) {
			a.trimToSize();
		}
	}

	public int[][] values() {
		int[][] ret = new int[size()][];
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
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
