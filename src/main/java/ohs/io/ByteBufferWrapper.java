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

	private static final int MAX_ARRAY_SIZE = ByteArray.MAX_ARRAY_SIZE;

	private static int hugeCapacity(int minCapacity) {
		if (minCapacity < 0) // overflow
			throw new OutOfMemoryError();
		return (minCapacity > MAX_ARRAY_SIZE) ? Integer.MAX_VALUE : MAX_ARRAY_SIZE;
	}

	private ByteBuffer bb;

	public ByteBufferWrapper() {
		this(new ByteArray());
	}

	public ByteBufferWrapper(ByteBuffer a) {
		this(a.array());
	}

	public ByteBufferWrapper(byte[] a) {
		this(new ByteArray(a));
	}

	public ByteBufferWrapper(ByteArray a) {
		bb = ByteBuffer.wrap(a.values());
	}

	public ByteBufferWrapper(int size) {
		this(new byte[size]);
	}

	public ByteBufferWrapper(long size) {
		this(new byte[(int) size]);
	}

	public int capacity() {
		return bb.capacity();
	}

	public void clear() {
		bb.clear();

	}

	private void ensureCapacity(int minCapacity) {
		byte[] buf = bb.array();
		if (minCapacity - buf.length > 0)
			grow(minCapacity);
	}

	public ByteArray getByteArray() {
		return new ByteArray(bb.array());
	}

	public ByteBuffer getByteBuffer() {
		return bb;
	}

	private void grow(int minCapacity) {
		byte[] buf = bb.array();

		// overflow-conscious code
		int oldCapacity = buf.length;
		int newCapacity = oldCapacity << 1;
		if (newCapacity - minCapacity < 0)
			newCapacity = minCapacity;
		if (newCapacity - MAX_ARRAY_SIZE > 0)
			newCapacity = hugeCapacity(minCapacity);
		buf = Arrays.copyOf(buf, newCapacity);

		int pos = bb.position();
		bb = ByteBuffer.wrap(buf);
		bb.position(pos);
	}

	public int position() {
		return bb.position();
	}

	public void position(int pos) {
		bb.position(pos);
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
		return bb.get();
	}

	public ByteArray readByteArray() {
		int size = bb.getInt();
		byte[] ret = new byte[size];
		bb.get(ret);
		return new ByteArray(ret);
	}

	public ByteArrayMatrix readByteArrayMatrix() {
		int size = bb.getInt();
		ByteArrayMatrix ret = new ByteArrayMatrix(size);
		for (int i = 0; i < size; i++) {
			ret.add(readByteArray());
		}
		return ret;
	}

	public List<IntegerArrayMatrix> readByteArrayMatrixList() {
		int size = bb.getInt();
		List<IntegerArrayMatrix> ret = Generics.newArrayList(size);
		for (int i = 0; i < size; i++) {
			ret.add(readIntegerArrayMatrix());
		}
		return ret;
	}

	public double readDouble() {
		return bb.getDouble();
	}

	public DoubleArray readDoubleArray() {
		int size = bb.getInt();
		double[] ret = new double[size];
		for (int i = 0; i < size; i++) {
			ret[i] = bb.getDouble();
		}
		return new DoubleArray(ret);
	}

	public int readInteger() {
		return bb.getInt();
	}

	public IntegerArray readIntegerArray() {
		int size = bb.getInt();
		int[] ret = new int[size];
		for (int i = 0; i < size; i++) {
			ret[i] = bb.getInt();
		}
		return new IntegerArray(ret);
	}

	public IntegerArrayMatrix readIntegerArrayMatrix() {
		int size = bb.getInt();
		IntegerArrayMatrix ret = new IntegerArrayMatrix(size);
		for (int i = 0; i < size; i++) {
			ret.add(readIntegerArray());
		}
		return ret;
	}

	public long readLong() {
		return bb.getLong();
	}

	public LongArray readLongArray() {
		int size = bb.getInt();
		long[] ret = new long[size];
		for (int i = 0; i < size; i++) {
			ret[i] = bb.getLong();
		}
		return new LongArray(ret);
	}

	public ShortArray readShortArray() {
		int size = bb.getInt();
		short[] ret = new short[size];
		for (int i = 0; i < size; i++) {
			ret[i] = bb.getShort();
		}
		return new ShortArray(ret);
	}

	public String readString() {
		return new String(readByteArray().values());
	}

	public String toString() {
		return bb.toString();
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
		ensureCapacity(bb.position() + 1);
		bb.put(a);
	}

	public void write(ByteArray a) {
		ensureCapacity(bb.position() + a.size() + Integer.BYTES);

		bb.putInt(a.size());
		for (byte b : a) {
			bb.put(b);
		}
	}

	public void write(ByteArrayMatrix a) {
		int size = (int) ByteArrayUtils.sizeOfByteBuffer(a);
		ensureCapacity(bb.position() + size);
		bb.putInt(a.size());

		for (ByteArray b : a) {
			write(b);
		}
	}

	public void write(double a) {
		ensureCapacity(bb.position() + Double.BYTES);
		bb.putDouble(a);
	}

	public void write(DoubleArray a) {
		ensureCapacity(bb.position() + a.size() * Double.BYTES + Integer.BYTES);
		bb.putInt(a.size());
		for (double b : a) {
			bb.putDouble(b);
		}
	}

	public void write(int a) {
		ensureCapacity(bb.position() + Integer.BYTES);
		bb.putInt(a);
	}

	public void write(IntegerArray a) {
		ensureCapacity(bb.position() + a.size() * Integer.BYTES + Integer.BYTES);
		bb.putInt(a.size());
		for (int b : a) {
			bb.putInt(b);
		}
	}

	public void write(IntegerArrayMatrix a) {
		int size = ByteArrayUtils.sizeOfByteBuffer(a);
		ensureCapacity(bb.position() + size);
		bb.putInt(a.size());
		for (IntegerArray b : a) {
			write(b);
		}
	}

	public void write(List<IntegerArrayMatrix> a) {
		ensureCapacity(bb.position() + Integer.BYTES);
		bb.putInt(a.size());
		for (IntegerArrayMatrix b : a) {
			write(b);
		}
	}

	public void write(long a) {
		ensureCapacity(bb.position() + Long.BYTES);
		bb.putLong(a);
	}

	public void write(LongArray a) {
		ensureCapacity(bb.position() + a.size() * Long.BYTES + Integer.BYTES);
		bb.putInt(a.size());
		for (long b : a) {
			bb.putLong(b);
		}
	}

	public void write(String a) {
		ensureCapacity(bb.position() + a.length() + Integer.BYTES);
		bb.putInt(a.length());
		bb.put(a.getBytes());
	}

}
