package ohs.corpus.type;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Collection;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import ohs.io.ByteArray;
import ohs.math.ArrayUtils;
import ohs.types.number.IntegerArray;
import ohs.types.number.LongArray;
import ohs.types.number.ShortArray;
import ohs.utils.ByteSize;

public class DataCompression {

	private static final int BUF_SIZE = 1024 * 5;

	public static void decodeGaps(int[] a) {
		decodeGaps(a, 1, a.length);
	}

	public static void decodeGaps(int[] a, int start, int end) {
		decodeGaps(new IntegerArray(a), start, end);
	}

	public static void decodeGaps(IntegerArray a) {
		decodeGaps(a, 1, a.size());
	}

	public static void decodeGaps(IntegerArray a, int start, int end) {
		for (int i = start; i < end; i++) {
			a.set(i, a.get(i) + a.get(i - 1));
		}
	}

	public static void decodeGaps(long[] a) {
		decodeGaps(new LongArray(a), 1, a.length);
	}

	public static void decodeGaps(LongArray a) {
		decodeGaps(a, 1, a.size());
	}

	public static void decodeGaps(LongArray a, int start, int end) {
		for (int i = start; i < end; i++) {
			a.set(i, a.get(i) + a.get(i - 1));
		}
	}

	public static void decodeGaps(short[] a) {
		for (int i = 1; i < a.length; i++) {
			a[i] = (short) (a[i] + a[i - 1]);
		}
	}

	public static IntegerArray decodeToIntegerArray(ByteArray a) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ByteArrayInputStream bais = new ByteArrayInputStream(a.values());
		GZIPInputStream zis = new GZIPInputStream(bais);

		int byte_cnt = 0;
		byte[] buf = new byte[BUF_SIZE];

		while ((byte_cnt = zis.read(buf, 0, buf.length)) != -1) {
			baos.write(buf, 0, byte_cnt);
		}

		// IntBuffer ibuf =
		// ByteBuffer.wrap(baos.toByteArray()).order(ByteOrder.BIG_ENDIAN).asIntBuffer();
		IntBuffer ibuf = ByteBuffer.wrap(baos.toByteArray()).asIntBuffer();
		int[] ret = new int[ibuf.remaining()];
		ibuf.get(ret);
		return new IntegerArray(ret);
	}

	/**
	 * http://stackoverflow.com/questions/11437203/byte-array-to-int-array
	 * 
	 * @param a
	 * @return
	 * @throws Exception
	 */
	public static int[] decodeToIntegers(byte[] a) throws Exception {
		return decodeToIntegerArray(new ByteArray(a)).values();
	}

	public static LongArray decodeToLongArray(ByteArray a) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ByteArrayInputStream bais = new ByteArrayInputStream(a.values());
		GZIPInputStream zis = new GZIPInputStream(bais);

		int byte_cnt = 0;
		byte[] buf = new byte[BUF_SIZE];

		while ((byte_cnt = zis.read(buf, 0, buf.length)) != -1) {
			baos.write(buf, 0, byte_cnt);
		}
		LongBuffer ibuf = ByteBuffer.wrap(baos.toByteArray()).asLongBuffer();
		long[] ret = new long[ibuf.remaining()];
		ibuf.get(ret);
		return new LongArray(ret);
	}

	public static long[] decodeToLongs(ByteArray a) throws Exception {
		return decodeToLongArray(a).values();
	}

	public ByteArray encode(ByteArray a) throws IOException {
		ByteBuffer buf = ByteBuffer.allocate(a.size());
		for (byte b : a) {
			buf.put(b);
		}

		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		GZIPOutputStream zos = new GZIPOutputStream(baos);

		zos.write(buf.array());
		zos.close();
		baos.close();
		byte[] ret = baos.toByteArray();
		return new ByteArray(ret);
	}

	public static ByteArray decode(ByteArray a) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ByteArrayInputStream bais = new ByteArrayInputStream(a.values());
		GZIPInputStream zis = new GZIPInputStream(bais);

		int byte_cnt = 0;
		byte[] buf = new byte[BUF_SIZE];

		while ((byte_cnt = zis.read(buf, 0, buf.length)) != -1) {
			baos.write(buf, 0, byte_cnt);
		}

		return new ByteArray(baos.toByteArray());
	}

	public static ShortArray decodeToShortArray(ByteArray a) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ByteArrayInputStream bais = new ByteArrayInputStream(a.values());
		GZIPInputStream zis = new GZIPInputStream(bais);

		int byte_cnt = 0;
		byte[] buf = new byte[BUF_SIZE];

		while ((byte_cnt = zis.read(buf, 0, buf.length)) != -1) {
			baos.write(buf, 0, byte_cnt);
		}

		ShortBuffer sbuf = ByteBuffer.wrap(baos.toByteArray()).asShortBuffer();
		short[] ret = new short[sbuf.remaining()];
		sbuf.get(ret);
		return new ShortArray(ret);
	}

	public static short[] decodeToShorts(byte[] a) throws Exception {
		return decodeToShortArray(new ByteArray(a)).values();
	}

	/**
	 * http://stackoverflow.com/questions/10974941/gzip-decompress-string-and-byte-conversion
	 * 
	 * @param a
	 * @return
	 * @throws Exception
	 */
	public static String decodeToString(ByteArray a) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ByteArrayInputStream bais = new ByteArrayInputStream(a.values());
		GZIPInputStream zis = new GZIPInputStream(bais);

		int byte_cnt = 0;
		byte[] buf = new byte[BUF_SIZE];
		while ((byte_cnt = zis.read(buf, 0, buf.length)) != -1) {
			baos.write(buf, 0, byte_cnt);
		}

		String s1 = new String(baos.toByteArray());
		// s1 = new String(buffer.toByteArray());
		return s1;
	}

	public static ByteArray encode(Collection<String> s) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		GZIPOutputStream zos = new GZIPOutputStream(baos);

		StringBuffer sb = new StringBuffer();
		for (String a : s) {
			sb.append(a);
			sb.append("\t");
		}

		zos.write(sb.toString().trim().getBytes());
		zos.finish();
		zos.close();
		return new ByteArray(baos.toByteArray());
	}

	public static byte[] encode(int[] a) throws Exception {
		return encode(new IntegerArray(a)).values();
	}

	public static byte[][] encode(int[][] a) throws Exception {
		byte[][] ret = new byte[a.length][];
		for (int i = 0; i < a.length; i++) {
			ret[i] = encode(a[i]);
		}
		return ret;
	}

	public static ByteArray encode(IntegerArray a) throws Exception {
		ByteBuffer buf = ByteBuffer.allocate(a.size() * Integer.BYTES);
		for (int v : a) {
			buf.putInt(v);
		}

		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		GZIPOutputStream zos = new GZIPOutputStream(baos);

		zos.write(buf.array());
		zos.close();
		baos.close();
		byte[] ret = baos.toByteArray();
		return new ByteArray(ret);
	}

	public static byte[] encode(long[] a) throws Exception {
		return encode(new LongArray(a)).values();
	}

	public static byte[][] encode(long[][] a) throws Exception {
		byte[][] ret = new byte[a.length][];
		for (int i = 0; i < a.length; i++) {
			ret[i] = encode(a[i]);
		}
		return ret;
	}

	public static ByteArray encode(LongArray a) throws Exception {
		ByteBuffer buf = ByteBuffer.allocate(a.size() * Long.BYTES);
		for (long v : a) {
			buf.putLong(v);
		}

		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		GZIPOutputStream zos = new GZIPOutputStream(baos);
		zos.write(buf.array());
		zos.close();
		return new ByteArray(baos.toByteArray());
	}

	public static byte[] encode(short[] a) throws Exception {
		return encode(new ShortArray(a)).values();
	}

	public static byte[][] encode(short[][] a) throws Exception {
		byte[][] ret = new byte[a.length][];
		for (int i = 0; i < a.length; i++) {
			ret[i] = encode(a[i]);
		}
		return ret;
	}

	public static ByteArray encode(ShortArray a) throws Exception {
		ByteBuffer buf = ByteBuffer.allocate(a.size() * Short.BYTES);
		for (short v : a) {
			buf.putShort(v);
		}

		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		GZIPOutputStream zos = new GZIPOutputStream(baos);

		zos.write(buf.array());
		zos.close();
		baos.close();
		byte[] ret = baos.toByteArray();
		return new ByteArray(ret);
	}

	public static ByteArray encode(String s) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		GZIPOutputStream zos = new GZIPOutputStream(baos);

		zos.write(s.getBytes());
		zos.close();
		baos.close();
		byte[] ret = baos.toByteArray();
		return new ByteArray(ret);
	}

	public static byte[] encode(String[] s) throws Exception {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		GZIPOutputStream zos = new GZIPOutputStream(baos);

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < s.length; i++) {
			sb.append(s[i]);
			if (i != s.length - 1) {
				sb.append("\t");
			}
		}

		zos.write(sb.toString().getBytes());
		zos.finish();
		zos.close();
		byte[] ret = baos.toByteArray();
		return ret;
	}

	public static void encodeGaps(int[] a) {
		encodeGaps(a, 1, a.length);
	}

	public static void encodeGaps(int[] a, int start, int end) {
		encodeGaps(new IntegerArray(a), start, end);
	}

	public static void encodeGaps(IntegerArray a) {
		encodeGaps(a, 1, a.size());
	}

	public static void encodeGaps(IntegerArray a, int start, int end) {
		int prev = a.get(start - 1);
		for (int i = start; i < end; i++) {
			int curr = a.get(i);
			int gap = curr - prev;
			a.set(i, gap);
			prev = curr;
		}
	}

	public static void encodeGaps(long[] a) {
		encodeGaps(new LongArray(a), 1, a.length);
	}

	public static void encodeGaps(LongArray a) {
		encodeGaps(a, 1, a.size());
	}

	public static void encodeGaps(LongArray a, int start, int end) {
		long prev = a.get(start - 1);
		for (int i = start; i < end; i++) {
			long curr = a.get(i);
			long gap = curr - prev;
			a.set(i, gap);
			prev = curr;
		}
	}

	// public static int[] decodeToIntegers(byte[] in) throws Exception {
	// ByteArrayInputStream bais = new ByteArrayInputStream(in);
	// GZIPInputStream zis = new GZIPInputStream(bais);
	//
	// int size = zis.read();
	// byte[] out = new byte[size];
	// zis.read(out);
	//
	// ByteBuffer buf = ByteBuffer.wrap(out);
	// }

	public static void encodeGaps(short[] a) {
		short prev = a[0];
		for (int i = 1; i < a.length; i++) {
			short curr = a[i];
			short gap = (short) (curr - prev);
			a[i] = gap;
			prev = curr;
		}
	}

	public static void main(String[] args) throws Exception {

		IntegerArray a = new IntegerArray(ArrayUtils.range(100000000));

		encodeGaps(a);
		ByteArray b = DataCompression.encode(a);

		System.out.println(new ByteSize(a.size() * Integer.BYTES));
		System.out.println(new ByteSize(b.size()));
	}

}
