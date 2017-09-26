package ohs.types.number;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import ohs.io.FileUtils;
import ohs.utils.ByteSize;

public class LongMatrix extends ArrayList<LongArray> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1810717124466759948L;

	public LongMatrix() {
		super();
	}

	public LongMatrix(Collection<LongArray> a) {
		addAll(a);
	}

	public LongMatrix(int size) {
		super(size);
	}

	public LongMatrix(long[][] a) {
		ensureCapacity(a.length);
		for (int i = 0; i < a.length; i++) {
			add(new LongArray(a[i]));
		}
	}

	public LongMatrix(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public void add(int i, long v) {
		ensure(i).add(v);
	}

	public ByteSize byteSize() {
		return new ByteSize(Integer.BYTES * sizeOfEntries());
	}

	public LongMatrix clone() {
		LongMatrix ret = new LongMatrix(size());
		for (LongArray a : this) {
			ret.add(a.clone());
		}
		return ret;
	}

	public LongArray ensure(int i) {
		if (i >= size()) {
			int new_size = (i - size()) + 1;
			for (int k = 0; k < new_size; k++) {
				add(new LongArray());
			}
		}
		return get(i);
	}

	public String info() {
		int min = Integer.MAX_VALUE;
		int max = -Integer.MAX_VALUE;

		for (LongArray a : this) {
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
			add(new LongArray(ois));
		}
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (LongArray a : this) {
			ret += a.size();
		}
		return ret;
	}

	public LongMatrix subMatrix(int i, int j) {
		return new LongMatrix(subList(i, j));
	}

	public LongArray toLongArray() {
		LongArray ret = new LongArray(sizeOfEntries());
		for (LongArray a : this) {
			ret.addAll(a);
		}
		return ret;
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

		for (LongArray a : this) {
			a.trimToSize();
		}
	}

	public long[][] values() {
		long[][] ret = new long[size()][];
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
