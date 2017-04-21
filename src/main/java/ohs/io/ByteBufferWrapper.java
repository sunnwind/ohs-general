package ohs.io;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.types.number.ShortArray;
import ohs.utils.Generics;

public class ByteBufferWrapper {

	private static final int DEFAULT_CAPACITY = 10000;

	private static final byte[] EMPTY_VALUES = {};

	private static final byte[] DEFAULT_CAPACITY_EMPTY_VALUES = {};

	private static int hugeCapacity(int minCapacity) {
		if (minCapacity < 0) // overflow
			throw new OutOfMemoryError();
		return (minCapacity > ByteArray.MAX_ARRAY_SIZE) ? Byte.MAX_VALUE : ByteArray.MAX_ARRAY_SIZE;
	}

	private byte[] vals;

	private ByteBuffer buf;

	public ByteBufferWrapper() {
		this(EMPTY_VALUES);
	}

	public ByteBufferWrapper(byte[] a) {
		this.vals = a;
		buf = ByteBuffer.wrap(a);
	}

	public ByteBufferWrapper(ByteArray a) {
		this(a.length() == a.size() ? a.values() : a.subArray(0, a.size()).values());
	}

	public ByteBufferWrapper(int size) {
		this(new byte[size]);
	}

	public ByteBufferWrapper(long size) {
		this(new byte[(int) size]);
	}

	public int capacity() {
		return buf.capacity();
	}

	public void clear() {
		buf.clear();
	}

	public void ensureCapacity(int minCapacity) {
		int minExpand = (vals != DEFAULT_CAPACITY_EMPTY_VALUES) ? 0 : DEFAULT_CAPACITY;

		if (minCapacity > minExpand) {
			ensureExplicitCapacity(minCapacity);
		}
	}

	private void ensureCapacityInternal(int minCapacity) {
		if (vals == DEFAULT_CAPACITY_EMPTY_VALUES) {
			minCapacity = Math.max(DEFAULT_CAPACITY, minCapacity);
		}

		ensureExplicitCapacity(minCapacity);
	}

	private void ensureExplicitCapacity(int minCapacity) {

		// overflow-conscious code
		if (minCapacity - vals.length > 0)
			grow(minCapacity);
	}

	public ByteBuffer getByteBuffer() {
		return buf;
	}

	private void grow(int minCapacity) {
		// overflow-conscious code

		int oldCapacity = vals.length;
		int newCapacity = oldCapacity + (oldCapacity >> 1);
		if (newCapacity - minCapacity < 0)
			newCapacity = minCapacity;
		if (newCapacity - ByteArray.MAX_ARRAY_SIZE > 0)
			newCapacity = hugeCapacity(minCapacity);
		// minCapacity is usually close to size, so this is a win:
		try {
			vals = Arrays.copyOf(vals, newCapacity);
		} catch (Exception e) {
			e.printStackTrace();
		}

		wrap();
	}

	public int position() {
		return buf.position();
	}

	public void position(int pos) {
		buf.position(pos);
	}

	public boolean readBoolean() {
		return readByte() == 1 ? true : false;
	}

	public boolean[] readBooleans() {
		ByteArray a = readByteArray();
		boolean[] b = new boolean[a.size()];
		for (int i = 0; i < a.size(); i++) {
			b[i] = a.get(i) == 1 ? true : false;
		}
		return b;
	}

	public byte readByte() {
		return buf.get();
	}

	public ByteArray readByteArray() {
		int size = buf.getInt();
		byte[] ret = new byte[size];
		buf.get(ret);
		return new ByteArray(ret);
	}

	public ByteArrayMatrix readByteArrayMatrix() {
		int size = buf.getInt();
		ByteArrayMatrix ret = new ByteArrayMatrix(size);
		for (int i = 0; i < size; i++) {
			ret.add(readByteArray());
		}
		return ret;
	}

	public List<IntegerArrayMatrix> readByteArrayMatrixList() {
		int size = buf.getInt();
		List<IntegerArrayMatrix> ret = Generics.newArrayList(size);
		for (int i = 0; i < size; i++) {
			ret.add(readIntegerArrayMatrix());
		}
		return ret;
	}

	public double readDouble() {
		return buf.getDouble();
	}

	public DoubleArray readDoubleArray() {
		int size = buf.getInt();
		double[] ret = new double[size];
		for (int i = 0; i < size; i++) {
			ret[i] = buf.getDouble();
		}
		return new DoubleArray(ret);
	}

	public int readInteger() {
		return buf.getInt();
	}

	public IntegerArray readIntegerArray() {
		int size = buf.getInt();
		int[] ret = new int[size];
		for (int i = 0; i < size; i++) {
			ret[i] = buf.getInt();
		}
		return new IntegerArray(ret);
	}

	public IntegerArrayMatrix readIntegerArrayMatrix() {
		int size = buf.getInt();
		IntegerArrayMatrix ret = new IntegerArrayMatrix(size);
		for (int i = 0; i < size; i++) {
			ret.add(readIntegerArray());
		}
		return ret;
	}

	public long readLong() {
		return buf.getLong();
	}

	public LongArray readLongArray() {
		int size = buf.getInt();
		long[] ret = new long[size];
		for (int i = 0; i < size; i++) {
			ret[i] = buf.getLong();
		}
		return new LongArray(ret);
	}

	public ShortArray readShortArray() {
		int size = buf.getInt();
		short[] ret = new short[size];
		for (int i = 0; i < size; i++) {
			ret[i] = buf.getShort();
		}
		return new ShortArray(ret);
	}

	public String readString() {
		return new String(readByteArray().values());
	}

	public ByteArray toByteArray() {
		return new ByteArray(buf.array()).subArray(0, buf.position());
	}

	public String toString() {
		return buf.toString();
	}

	private void wrap() {
		int pos_old = buf.position();
		buf = ByteBuffer.wrap(vals);
		buf.position(pos_old);
	}

	public void write(boolean a) {
		write((byte) (a ? 1 : 0));
	}

	public void write(boolean[] a) {
		byte[] b = new byte[a.length];
		for (int i = 0; i < a.length; i++) {
			b[i] = (byte) (a[i] ? 1 : 0);
		}
		write(new ByteArray(b));
	}

	public void write(byte a) {
		ensureCapacityInternal(buf.position() + 1);
		buf.put(a);
	}

	public void write(ByteArray a) {
		ensureCapacityInternal(buf.position() + a.size() + Integer.BYTES);
		buf.putInt(a.size());
		if (a.size() == a.length()) {
			buf.put(a.values());
		} else {
			for (byte b : a) {
				buf.put(b);
			}
		}
	}

	public void write(ByteArrayMatrix a) {
		ensureCapacityInternal(buf.position() + Integer.BYTES);
		buf.putInt(a.size());

		for (ByteArray b : a) {
			write(b);
		}
	}

	public void write(double a) {
		ensureCapacityInternal(buf.position() + Double.BYTES);
		buf.putDouble(a);
	}

	public void write(long a) {
		ensureCapacityInternal(buf.position() + Long.BYTES);
		buf.putLong(a);
	}

	public void write(DoubleArray a) {
		ensureCapacityInternal(buf.position() + a.size() * Double.BYTES + Integer.BYTES);
		buf.putInt(a.size());
		for (double b : a) {
			buf.putDouble(b);
		}
	}

	public void write(int a) {
		ensureCapacityInternal(buf.position() + Integer.BYTES);
		buf.putInt(a);
	}

	public void write(IntegerArray a) {
		ensureCapacityInternal(buf.position() + a.size() * Integer.BYTES + Integer.BYTES);
		buf.putInt(a.size());
		for (int b : a) {
			buf.putInt(b);
		}
	}

	public void write(IntegerArrayMatrix a) {
		ensureCapacityInternal(buf.position() + Integer.BYTES);
		buf.putInt(a.size());
		for (IntegerArray b : a) {
			write(b);
		}
	}

	public void write(List<IntegerArrayMatrix> a) {
		ensureCapacityInternal(buf.position() + Integer.BYTES);
		buf.putInt(a.size());
		for (IntegerArrayMatrix b : a) {
			write(b);
		}
	}

	public void write(LongArray a) {
		ensureCapacityInternal(buf.position() + a.size() * Long.BYTES + Integer.BYTES);
		buf.putInt(a.size());
		for (long b : a) {
			buf.putLong(b);
		}
	}

	public void write(String a) {
		ensureCapacityInternal(buf.position() + a.length() + Integer.BYTES);
		buf.putInt(a.length());
		buf.put(a.getBytes());
	}

}
