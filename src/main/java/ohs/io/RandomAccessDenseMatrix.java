package ohs.io;

import java.io.File;
import java.io.ObjectInputStream;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import ohs.ir.medical.general.MIRPath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.LongArray;
import ohs.utils.Generics;

public class RandomAccessDenseMatrix {

	public static void build(DenseMatrix a, FileChannel fc) throws Exception {
		long old_start = fc.position();
		int row_size = a.rowSize();
		int col_size = a.colSize();

		LongArray starts = new LongArray(new long[a.size()]);
		IntegerArray lens = new IntegerArray(new int[a.size()]);

		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(starts), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(lens), fc);
		FileUtils.write(ByteArrayUtils.toByteArray(new IntegerArray(new int[] { row_size, col_size })), fc);

		for (int i = 0; i < a.size(); i++) {
			DenseVector row = a.row(i);
			ByteArray data = ByteArrayUtils.toByteArray(new DoubleArray(row.values()));
			long[] info = FileUtils.write(data, fc);
			starts.set(i, info[0]);
			lens.set(i, (int) info[1]);
		}

		fc.position(old_start);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(starts), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(lens), fc);

		fc.position(fc.size());
	}

	public static void build(DenseMatrix a, String outFileName) throws Exception {
		FileChannel fc = FileUtils.openFileChannel(outFileName, "rw");
		build(a, fc);
		fc.close();
	}

	public static void buildFromTempFile(String inFileName, String outFileName) throws Exception {
		System.out.printf("write at [%s] from [%s]\n", outFileName, inFileName);

		ObjectInputStream ois = FileUtils.openObjectInputStream(inFileName);
		int row_size = ois.readInt();
		int col_size = ois.readInt();

		FileChannel fc = FileUtils.openFileChannel(outFileName, "rw");

		LongArray starts = new LongArray(new long[row_size]);
		IntegerArray lens = new IntegerArray(new int[row_size]);

		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(starts), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(lens), fc);
		FileUtils.write(ByteArrayUtils.toByteArray(new IntegerArray(new int[] { row_size, col_size })), fc);

		for (int i = 0; i < row_size; i++) {
			DenseVector row = new DenseVector(ois);
			ByteArray data = ByteArrayUtils.toByteArray(new DoubleArray(row.values()));
			long[] info = FileUtils.write(data, fc);
			starts.set(i, info[0]);
			lens.set(i, (int) info[1]);
		}

		fc.position(0);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(starts), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(lens), fc);

		ois.close();
		fc.close();
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		String[] dirs = { MIRPath.OHSUMED_DIR, MIRPath.WIKI_DIR, MIRPath.BIOASQ_DIR, MIRPath.TREC_CDS_2016_DIR };
		for (int i = 0; i < dirs.length; i++) {
			File dir = new File(dirs[i], "emb");
			File inFile = new File(dir, "glove.ser.gz");
			File outFile = new File(dir, "glove_ra.ser");

			if (!inFile.exists()) {
				continue;
			}

			outFile.delete();

			FileChannel fc = FileUtils.openFileChannel(outFile, "rw");
			DenseMatrix E = new DenseMatrix(inFile.getPath());

			build(E, fc);

			fc.close();
		}

		// else {
		//
		// Vocab vocab = new Vocab();
		// vocab.readObject(MIRPath.TREC_CDS_2014_COL_DC_DIR + "/vocab.ser.gz");
		//
		// RandomAccessDenseMatrix ram = new RandomAccessDenseMatrix(fc);
		//
		// int w = vocab.indexOf("pain");
		//
		// DenseVector v1 = ram.row(w);
		//
		// Counter<String> c = Generics.newCounter();
		//
		// for (int t = 0; t < ram.rowSize(); t++) {
		// if (t == w) {
		// continue;
		// }
		// DenseVector v2 = ram.row(t);
		//
		// double cosine = VectorMath.cosine(v1, v2);
		//
		// String word = vocab.getObject(t);
		//
		// c.setCount(word, cosine);
		// }
		//
		// System.out.println(c.toString());
		// }

		System.out.println("process ends.");
	}

	public static void read(FileChannel fc) throws Exception {

		System.out.println();
	}

	private FileChannel fc;

	private IntegerArray dims;

	private LongArray starts;

	private IntegerArray lens;

	private Map<Integer, DenseVector> softCache;

	private List<DenseVector> hardCache;

	private ReentrantLock lock = new ReentrantLock();

	public RandomAccessDenseMatrix(FileChannel fc, boolean cache_all) throws Exception {
		this.fc = fc;
		starts = ByteArrayUtils.toLongArray(FileUtils.readByteArrayMatrix(fc));
		lens = ByteArrayUtils.toIntegerArray(FileUtils.readByteArrayMatrix(fc));
		dims = ByteArrayUtils.toIntegerArray(FileUtils.readByteArray(fc));

		softCache = Generics.newWeakHashMap(rowSize());

		if (cache_all) {
			hardCache = Generics.newArrayList(rowSize());
			for (int i = 0; i < rowSize(); i++) {
				hardCache.add(null);
			}
		}
	}

	public RandomAccessDenseMatrix(String fileName) throws Exception {
		this(fileName, false);
	}

	public RandomAccessDenseMatrix(String fileName, boolean cache_all) throws Exception {
		this(FileUtils.openFileChannel(fileName, "r"), cache_all);
	}

	public void close() throws Exception {
		fc.close();
	}

	public int colSize() {
		return dims.get(1);
	}

	public Map<Integer, DenseVector> getRowCache() {
		return softCache;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[Random Access Dense Matrix]");
		sb.append(String.format("\nrows:\t%d", dims.get(0)));
		sb.append(String.format("\ncols:\t%d", dims.get(1)));
		return sb.toString();
	}

	public DenseVector row(int i) throws Exception {
		DenseVector ret = null;
		if (i >= 0 && i < dims.get(0)) {

			if (hardCache != null) {
				try {
					lock.lock();
					ret = hardCache.get(i);
					if (ret == null) {
						fc.position(starts.get(i));
						int len = lens.get(i);
						ByteArray data = FileUtils.readByteArray(fc, len);

						ByteBufferWrapper buf = new ByteBufferWrapper(data);
						buf.readInteger();

						ret = new DenseVector(colSize());
						for (int j = 0; j < ret.size(); j++) {
							ret.add(j, buf.readDouble());
						}

						hardCache.set(i, ret);
					}
				} finally {
					lock.unlock();
				}
			} else {
				try {
					lock.lock();
					ret = softCache.get(i);
					if (ret == null) {
						fc.position(starts.get(i));
						ByteArray data = FileUtils.readByteArray(fc, lens.get(i));

						ByteBufferWrapper buf = new ByteBufferWrapper(data);
						buf.readInteger();

						ret = new DenseVector(colSize());
						for (int j = 0; j < ret.size(); j++) {
							ret.add(j, buf.readDouble());
						}

						softCache.put(i, ret);
					}
				} finally {
					lock.unlock();
				}
			}
		}

		return ret;
	}

	public DenseMatrix rowsAsMatrix() throws Exception {
		return rowsAsMatrix(0, rowSize());
	}

	public DenseMatrix rowsAsMatrix(int start, int size) throws Exception {
		List<DenseVector> rows = Generics.newArrayList(size);
		for (int i = start; i < start + size; i++) {
			rows.add(row(i));
		}
		return new DenseMatrix(rows);
	}

	public DenseMatrix rowsAsMatrix(int[] is) throws Exception {
		List<DenseVector> rows = Generics.newArrayList(is.length);
		for (int i : is) {
			rows.add(row(i));
		}
		return new DenseMatrix(rows);
	}

	public int rowSize() {
		return dims.get(0);
	}

	public String toString() {
		return info();
	}

}
