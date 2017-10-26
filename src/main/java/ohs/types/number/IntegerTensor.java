package ohs.types.number;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import ohs.io.FileUtils;
import ohs.utils.ByteSize;

public class IntegerTensor extends ArrayList<IntegerMatrix> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1810717124466759948L;

	public IntegerTensor() {
		super();
	}

	public IntegerTensor(Collection<IntegerMatrix> a) {
		addAll(a);
	}

	public IntegerTensor(int size) {
		super(size);
	}

	public IntegerTensor(int[][][] a) {
		ensureCapacity(a.length);
		for (int i = 0; i < a.length; i++) {
			add(new IntegerMatrix(a[i]));
		}
	}

	public IntegerTensor(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public void add(int i, int j, int v) {
		ensure(i).ensure(j).add(v);
	}

	public ByteSize byteSize() {
		return new ByteSize(Integer.BYTES * sizeOfEntries());
	}

	public IntegerTensor clone() {
		IntegerTensor ret = new IntegerTensor(size());
		for (IntegerMatrix a : this) {
			ret.add(a.clone());
		}
		return ret;
	}

	public IntegerMatrix ensure(int i) {
		if (i >= size()) {
			int new_size = (i - size()) + 1;
			for (int k = 0; k < new_size; k++) {
				add(new IntegerMatrix());
			}
		}
		return get(i);
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

		for (IntegerMatrix a : this) {
			min2 = Math.min(min2, a.size());
			max2 = Math.max(max2, a.size());

			for (IntegerArray b : a) {
				min3 = Math.min(min3, b.size());
				max3 = Math.max(max3, b.size());
			}
		}

		StringBuffer sb = new StringBuffer();
		sb.append("[Info]\n");
		sb.append(String.format("min size:\t%d\n", minSizes.toString()));
		sb.append(String.format("max size:\t%d\n", maxSizes.toString()));
		sb.append(String.format("entry size:\t%d\n", sizeOfEntries()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ensureCapacity(size);
		for (int i = 0; i < size; i++) {
			add(new IntegerMatrix(ois));
		}
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (IntegerMatrix a : this) {
			ret += a.sizeOfEntries();
		}
		return ret;
	}

	public int sizeOfIntegerArrays() {
		int ret = 0;
		for (IntegerMatrix a : this) {
			ret += a.size();
		}
		return ret;
	}

	public IntegerTensor subTensor(int i, int j) {
		return new IntegerTensor(subList(i, j));
	}

	public IntegerTensor subTensor(int[] is) {
		IntegerTensor ret = new IntegerTensor(is.length);
		for (int i : is) {
			ret.add(get(i));
		}
		return ret;
	}

	public IntegerArray toIntegerArray() {
		IntegerArray ret = new IntegerArray(sizeOfEntries());
		for (IntegerMatrix a : this) {
			for (IntegerArray b : a) {
				ret.addAll(b);
			}
		}
		return ret;
	}

	public String toString() {
		return toString(30);
	}

	public String toString(int size) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size(); i++) {
			sb.append(String.format("%d:\n%s", i, get(i).toString()));
			if (i == size) {
				sb.append(String.format("\n..."));
				break;
			} else {
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	public void trimToSize() {
		super.trimToSize();
		for (IntegerMatrix a : this) {
			a.trimToSize();
		}
	}

	public int[][][] values() {
		int[][][] ret = new int[size()][][];
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
