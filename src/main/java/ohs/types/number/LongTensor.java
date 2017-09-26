package ohs.types.number;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import ohs.io.FileUtils;
import ohs.utils.ByteSize;

public class LongTensor extends ArrayList<LongMatrix> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1810717124466759948L;

	public LongTensor() {
		super();
	}

	public LongTensor(Collection<LongMatrix> a) {
		addAll(a);
	}

	public LongTensor(int size) {
		super(size);
	}

	public LongTensor(long[][][] a) {
		ensureCapacity(a.length);
		for (int i = 0; i < a.length; i++) {
			add(new LongMatrix(a[i]));
		}
	}

	public LongTensor(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public void add(int i, int j, long v) {
		ensure(i).ensure(j).add(v);
	}

	public ByteSize byteSize() {
		return new ByteSize(Long.BYTES * sizeOfEntries());
	}

	public LongTensor clone() {
		LongTensor ret = new LongTensor(size());
		for (LongMatrix a : this) {
			ret.add(a.clone());
		}
		return ret;
	}

	public LongMatrix ensure(int i) {
		if (i >= size()) {
			int new_size = (i - size()) + 1;
			for (int k = 0; k < new_size; k++) {
				add(new LongMatrix());
			}
		}
		return get(i);
	}

	public String info() {

		LongArray minSizes = new LongArray(3);
		LongArray maxSizes = new LongArray(3);

		minSizes.add(size());
		maxSizes.add(size());

		int min2 = Integer.MAX_VALUE;
		int min3 = Integer.MAX_VALUE;

		int max2 = -Integer.MAX_VALUE;
		int max3 = -Integer.MAX_VALUE;

		for (LongMatrix a : this) {
			min2 = Math.min(min2, a.size());
			max2 = Math.max(max2, a.size());

			for (LongArray b : a) {
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
			add(new LongMatrix(ois));
		}
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public int sizeOfEntries() {
		int ret = 0;
		for (LongMatrix a : this) {
			ret += a.sizeOfEntries();
		}
		return ret;
	}

	public LongTensor subTensor(int i, int j) {
		return new LongTensor(subList(i, j));
	}

	public LongTensor subTensor(int[] is) {
		LongTensor ret = new LongTensor(is.length);
		for (int i : is) {
			ret.add(get(i));
		}
		return ret;
	}

	public LongArray toLongArray() {
		LongArray ret = new LongArray(sizeOfEntries());
		for (LongMatrix a : this) {
			for (LongArray b : a) {
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
		for (LongMatrix a : this) {
			a.trimToSize();
		}
	}

	public long[][][] values() {
		long[][][] ret = new long[size()][][];
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
