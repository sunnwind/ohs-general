package ohs.io;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.List;

import ohs.types.number.DoubleArray;
import ohs.types.number.DoubleArrayMatrix;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.types.number.LongArray;
import ohs.types.number.ShortArray;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

public class ByteArrayUtils {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		// long v = Integer.MAX_VALUE * 4L;
		// double v2 = Long.MAX_VALUE / Integer.MAX_VALUE;
		// System.out.println(new ByteSize(Integer.MAX_VALUE * (long) Integer.BYTES));
		// System.out.println(new ByteSize(Integer.MAX_VALUE * (long) Byte.BYTES));
		// System.out.println(new ByteSize(Long.MAX_VALUE));

		IntegerArray a = new IntegerArray(new int[ByteArray.MAX_ARRAY_SIZE / 3]);
		ByteSize bs = new ByteSize(1L * a.size() * Integer.BYTES);

		System.out.println(a.size());
		System.out.println(bs.toString());

		ByteArrayMatrix b = toByteArrayMatrix(a);

		IntegerArray c = toIntegerArray(b);
		System.out.println(b.size());
		System.out.println(c.size());

		System.out.println("process ends.");
	}

	public static int sizeOfByteBuffer(byte[] a) {
		return sizeOfByteBuffer(new ByteArray(a));
	}

	public static int sizeOfByteBuffer(byte[][] a) {
		return sizeOfByteBuffer(new ByteArrayMatrix(a));
	}

	public static int sizeOfByteBuffer(ByteArray a) {
		return Integer.BYTES + Byte.BYTES * a.size();
	}

	public static int sizeOfByteBuffer(ByteArrayMatrix a) {
		int ret = Integer.BYTES;
		for (ByteArray b : a) {
			ret += sizeOfByteBuffer(b);
		}
		return ret;
	}

	public static int sizeOfByteBuffer(double[] a) {
		return sizeOfByteBuffer(new DoubleArray(a));
	}

	public static int sizeOfByteBuffer(double[][] a) {
		return sizeOfByteBuffer(new DoubleArrayMatrix(a));
	}

	public static int sizeOfByteBuffer(DoubleArray a) {
		return Integer.BYTES + Double.BYTES * a.size();
	}

	public static int sizeOfByteBuffer(DoubleArrayMatrix a) {
		int ret = Integer.BYTES;
		for (DoubleArray b : a) {
			ret += sizeOfByteBuffer(b);
		}
		return ret;
	}

	public static int sizeOfByteBuffer(int[] a) {
		return sizeOfByteBuffer(new IntegerArray(a));
	}

	public static int sizeOfByteBuffer(int[][] a) {
		return sizeOfByteBuffer(new IntegerArrayMatrix(a));
	}

	public static int sizeOfByteBuffer(IntegerArray a) {
		return Integer.BYTES * (1 + a.size());
	}

	public static int sizeOfByteBuffer(IntegerArrayMatrix a) {
		int ret = Integer.BYTES;
		for (IntegerArray b : a) {
			ret += sizeOfByteBuffer(b);
		}
		return ret;
	}

	public static int sizeOfByteBuffer(List<ByteArrayMatrix> a) {
		int ret = Integer.BYTES;
		for (ByteArrayMatrix b : a) {
			ret += sizeOfByteBuffer(b);
		}
		return ret;
	}

	public static int sizeOfByteBuffer(long[] a) {
		return sizeOfByteBuffer(new LongArray(a));
	}

	public static int sizeOfByteBuffer(LongArray a) {
		return Long.BYTES * (1 + a.size());
	}

	public static int sizeOfByteBuffer(short[] a) {
		return sizeOfByteBuffer(new ShortArray(a));
	}

	public static int sizeOfByteBuffer(ShortArray a) {
		return Integer.BYTES + Short.BYTES * a.size();
	}

	public static int sizeOfByteBuffer(String a) {
		return Integer.BYTES + Character.BYTES * a.length();
	}

	public static int sizeOfByteBuffer(String[] a) {
		int ret = Integer.BYTES;
		for (String b : a) {
			ret += sizeOfByteBuffer(b);
		}
		return ret;
	}

	public static boolean[] toBooleans(byte[] a) {
		boolean[] b = new boolean[a.length];
		for (int i = 0; i < a.length; i++) {
			b[i] = a[i] == 1 ? true : false;
		}
		return b;
	}

	public static ByteArray toByteArray(ByteArrayMatrix a) {
		ByteArray ret = new ByteArray((int) sizeOfByteBuffer(a));
		ret.addAll(toByteArray(a.size()));
		for (ByteArray b : a) {
			ret.addAll(toByteArray(b.size()));
			ret.addAll(b);
		}
		return ret;
	}

	public static ByteArray toByteArray(double a) {
		ByteBuffer buf = ByteBuffer.allocate(Double.BYTES);
		buf.putDouble(a);
		return new ByteArray(buf.array());
	}

	public static ByteArray toByteArray(DoubleArray a) {
		ByteBuffer c = ByteBuffer.allocate(Double.BYTES * a.size());
		for (double b : a) {
			c.putDouble(b);
		}
		return new ByteArray(c.array());
	}

	public static ByteArray toByteArray(int a) {
		ByteBuffer buf = ByteBuffer.allocate(Integer.BYTES);
		buf.putInt(a);
		return new ByteArray(buf.array());
	}

	public static ByteArray toByteArray(IntegerArray a) {
		ByteBuffer buf = ByteBuffer.allocate(a.size() * Integer.BYTES);
		for (int b : a) {
			buf.putInt(b);
		}
		return new ByteArray(buf.array());
	}

	public static ByteArray toByteArray(long a) throws Exception {
		ByteBuffer buf = ByteBuffer.allocate(Long.BYTES);
		buf.putLong(a);
		return new ByteArray(buf.array());
	}

	public static ByteArray toByteArray(LongArray a) {
		ByteBuffer c = ByteBuffer.allocate(Long.BYTES * a.size());
		for (long b : a) {
			c.putLong(b);
		}
		return new ByteArray(c.array());
	}

	public static ByteArrayMatrix toByteArrayMatrix(ByteArray a) {
		int i = 0;
		int size1 = toInteger(a.subArray(i, i += Integer.BYTES).values());
		ByteArrayMatrix b = new ByteArrayMatrix(size1);
		for (int j = 0; j < size1; j++) {
			int size2 = toInteger(a.subArray(i, i += Integer.BYTES).values());
			b.add(a.subArray(i, i += size2));
		}
		return b;
	}

	public static ByteArrayMatrix toByteArrayMatrix(IntegerArray a) {
		ByteBufferWrapper buf = new ByteBufferWrapper(FileUtils.DEFAULT_BUF_SIZE);
		return toByteArrayMatrix(a, buf);
	}

	public static ByteArrayMatrix toByteArrayMatrix(IntegerArray a, ByteBufferWrapper buf) {
		int chunk_size = (buf.capacity() - Integer.BYTES) / Integer.BYTES;

		List<ByteArray> ret = Generics.newLinkedList();

		buf.write(a.size());
		ret.add(buf.getByteArray());
		buf.clear();

		int s = 0;
		while (s < a.size()) {
			int e = Math.min(a.size(), s + chunk_size);
			int window_size = e - s;

			buf.write(window_size);
			for (int i = s; i < e; i++) {
				buf.write(a.value(i));
			}
			s = e;
			ret.add(buf.getByteArray());
			buf.clear();
		}
		return new ByteArrayMatrix(ret);
	}

	public static ByteArrayMatrix toByteArrayMatrix(LongArray a) {
		ByteBufferWrapper buf = new ByteBufferWrapper(FileUtils.DEFAULT_BUF_SIZE);
		return toByteArrayMatrix(a, buf);
	}

	public static ByteArrayMatrix toByteArrayMatrix(LongArray a, ByteBufferWrapper buf) {
		int chunk_size = (buf.capacity() - Integer.BYTES) / Long.BYTES;

		List<ByteArray> ret = Generics.newLinkedList();

		buf.write(a.size());
		ret.add(buf.getByteArray());
		buf.clear();

		int s = 0;
		while (s < a.size()) {
			int e = Math.min(a.size(), s + chunk_size);
			int window_size = e - s;

			buf.write(window_size);
			for (int i = s; i < e; i++) {
				buf.write(a.value(i));
			}
			s = e;
			ret.add(buf.getByteArray());
			buf.clear();
		}
		return new ByteArrayMatrix(ret);
	}

	public static List<ByteArrayMatrix> toByteArrayMatrixList(ByteArray a) {
		int i = 0;
		int size1 = toInteger(a.subArray(i, i += Integer.BYTES));
		List<ByteArrayMatrix> ret = Generics.newArrayList(size1);
		for (int j = 0; j < size1; j++) {
			int size2 = toInteger(a.subArray(i, i += Integer.BYTES));
			ByteArrayMatrix sub = new ByteArrayMatrix(size2);
			for (int k = 0; k < size2; k++) {
				int size3 = toInteger(a.subArray(i, i += Integer.BYTES));
				sub.add(a.subArray(i, i += size3));
			}
			ret.add(sub);
		}
		return ret;
	}

	public static byte[] toBytes(boolean[] a) {
		ByteBuffer c = ByteBuffer.allocate(a.length);
		for (boolean b : a) {
			c.put((byte) (b ? 1 : 0));
		}
		return c.array();
	}

	public static byte[] toBytes(double a) {
		ByteBuffer buf = ByteBuffer.allocate(Double.BYTES);
		buf.putDouble(a);
		return buf.array();
	}

	public static byte[] toBytes(long a) {
		ByteBuffer buf = ByteBuffer.allocate(Long.BYTES);
		buf.putLong(a);
		return buf.array();
	}

	public static byte[] toBytes(long[] a) {
		ByteBuffer buf = ByteBuffer.allocate(Long.BYTES * a.length);
		for (long b : a) {
			buf.putLong(b);
		}
		return buf.array();
	}

	public static double toDouble(ByteArray a) {
		return toDoubleArray(a).get(0);
	}

	public static DoubleArray toDoubleArray(ByteArray a) {
		DoubleBuffer b = ByteBuffer.wrap(a.values()).asDoubleBuffer();
		double[] c = new double[b.remaining()];
		b.get(c);
		return new DoubleArray(c);
	}

	public static int toInteger(byte[] a) {
		return toInteger(new ByteArray(a));
	}

	public static int toInteger(ByteArray a) {
		return toIntegerArray(a).get(0);
	}

	public static IntegerArray toIntegerArray(byte[] a) {
		return new IntegerArray(toIntegers(a));
	}

	public static IntegerArray toIntegerArray(ByteArray a) {
		IntBuffer b = ByteBuffer.wrap(a.values()).asIntBuffer();
		int[] c = new int[b.remaining()];
		b.get(c);
		return new IntegerArray(c);
	}

	public static IntegerArray toIntegerArray(ByteArrayMatrix a) {
		int i = 0;
		int size = new ByteBufferWrapper(a.get(i++)).readInteger();
		IntegerArray ret = new IntegerArray(size);
		while (i < a.size()) {
			ByteArray b = a.get(i++);
			ByteBufferWrapper buf = new ByteBufferWrapper(b);
			ret.addAll(buf.readIntegerArray());
		}
		return ret;
	}

	public static int[] toIntegers(byte[] a) {
		return toIntegerArray(new ByteArray(a)).values();
	}

	public static long toLong(byte[] a) {
		return toLong(new ByteArray(a));
	}

	public static long toLong(ByteArray a) {
		return toLongArray(a).get(0);
	}

	public static LongArray toLongArray(ByteArray a) {
		LongBuffer b = ByteBuffer.wrap(a.values()).asLongBuffer();
		long[] c = new long[b.remaining()];
		b.get(c);
		return new LongArray(c);
	}

	public static LongArray toLongArray(ByteArrayMatrix a) {
		int i = 0;
		int size = new ByteBufferWrapper(a.get(i++)).readInteger();
		LongArray ret = new LongArray(size);
		while (i < a.size()) {
			ByteArray b = a.get(i++);
			ByteBufferWrapper buf = new ByteBufferWrapper(b);
			ret.addAll(buf.readLongArray());
		}
		return ret;
	}

	public static long[] toLongs(byte[] a) throws Exception {
		return toLongArray(new ByteArray(a)).values();
	}

	public static short[] toShortArray(byte[] a) throws Exception {
		return toShorts(new ByteArray(a)).values();
	}

	public static ShortArray toShorts(ByteArray a) throws Exception {
		ShortBuffer b = ByteBuffer.wrap(a.values()).asShortBuffer();
		short[] c = new short[b.remaining()];
		b.get(c);
		return new ShortArray(c);
	}

}
