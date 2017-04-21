package ohs.io;

import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import ohs.math.VectorUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.LongArray;
import ohs.utils.Generics;

public class RandomAccessSparseMatrix {

	public static void build(SparseMatrix a, FileChannel fc) throws Exception {
		long old_start = fc.position();

		LongArray starts = new LongArray(new long[a.size()]);
		IntegerArray lens = new IntegerArray(new int[a.size()]);
		IntegerArray rowIdxs = new IntegerArray(a.rowIndexes());

		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(starts), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(lens), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(rowIdxs), fc);

		int row_size = a.rowSize();
		int col_size = a.colSize();

		FileUtils.write(ByteArrayUtils.toByteArray(new IntegerArray(new int[] { row_size, col_size })), fc);

		for (int i = 0; i < a.size(); i++) {
			int row_idx = a.indexAt(i);
			SparseVector row = a.row(i);

			ByteArrayMatrix data = new ByteArrayMatrix(2);
			data.add(ByteArrayUtils.toByteArray(new IntegerArray(row.indexes())));
			data.add(ByteArrayUtils.toByteArray(new DoubleArray(row.values())));
			long[] info = FileUtils.write(data, fc);
			starts.add(info[0]);
			lens.add((int) info[1]);
			rowIdxs.add(row_idx);
		}

		fc.position(old_start);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(starts), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(lens), fc);
		FileUtils.write(ByteArrayUtils.toByteArrayMatrix(rowIdxs), fc);
	}

	public static void build(SparseMatrix a, String fileName) throws Exception {
		FileChannel fc = FileUtils.openFileChannel(fileName, "w");
		build(a, fc);
		fc.close();
	}

	public static void build(String inFileName, String outFileName) throws Exception {
		build(new SparseMatrix(inFileName), FileUtils.openFileChannel(outFileName, "w"));
	}

	private FileChannel fc;

	private IntegerArray dims;

	private LongArray starts;

	private IntegerArray lens;

	private IntegerArray rowIdxs;

	private Map<Integer, SparseVector> rowCache;

	private Map<Integer, SparseVector> colCache;

	private ReentrantLock lock = new ReentrantLock();

	public RandomAccessSparseMatrix(FileChannel fc, boolean cache_all) throws Exception {
		this.fc = fc;
		dims = ByteArrayUtils.toIntegerArray(FileUtils.readByteArray(fc));
		starts = ByteArrayUtils.toLongArray(FileUtils.readByteArray(fc));
		lens = ByteArrayUtils.toIntegerArray(FileUtils.readByteArray(fc));
		rowIdxs = ByteArrayUtils.toIntegerArray(FileUtils.readByteArray(fc));

		rowCache = Generics.newWeakHashMap();
		colCache = Generics.newWeakHashMap();

		if (cache_all) {
			rowCache = Generics.newHashMap();
			colCache = Generics.newHashMap();
		}
	}

	public RandomAccessSparseMatrix(String fileName, boolean cache_all) throws Exception {
		this(FileUtils.openFileChannel(fileName, "r"), cache_all);
	}

	public void close() throws Exception {
		fc.close();
	}

	public int colSize() {
		return dims.get(1);
	}

	public SparseVector column(int j) throws Exception {
		SparseVector ret = null;
		if (j >= 0 && j < colSize()) {
			try {
				lock.lock();

				if (ret == null) {
					Counter<Integer> c = Generics.newCounter();
					for (int loc = 0; loc < rowSize(); loc++) {
						SparseVector row = row(loc);
						double v = row.value(j);
						if (v != 0) {
							c.setCount(loc, v);
						}
					}

					if (c.size() > 0) {
						ret = VectorUtils.toSparseVector(c);
						colCache.put(j, ret);
					}
				}

			} finally {
				lock.unlock();
			}
			ret = colCache.get(j);
		}
		return ret;
	}

	public SparseVector row(int i) throws Exception {
		SparseVector ret = null;
		if (i >= 0) {
			int loc = Arrays.binarySearch(rowIdxs.values(), i);
			if (loc > -1) {
				ret = rowAt(loc);
			}
		}
		return ret;
	}

	public SparseVector rowAt(int loc) throws Exception {
		SparseVector ret = null;
		try {
			lock.lock();
			ret = rowCache.get(loc);
			if (ret == null) {
				fc.position(starts.get(loc));
				ByteBufferWrapper buf = new ByteBufferWrapper(FileUtils.readByteArray(fc));
				IntegerArray idxs = buf.readIntegerArray();
				DoubleArray vals = buf.readDoubleArray();
				ret = new SparseVector(idxs.values(), vals.values());
				rowCache.put(loc, ret);
			}
		} finally {
			lock.unlock();
		}
		return ret;
	}

	public IntegerArray rowIndexes() {
		return rowIdxs;
	}

	public SparseMatrix rowsAsMatrix() throws Exception {
		return rowsAsMatrix(0, rowSize());
	}

	public SparseMatrix rowsAsMatrix(int start, int size) throws Exception {
		Map<Integer, SparseVector> m = Generics.newHashMap(size);
		for (int loc = start; loc < start + size; loc++) {
			m.put(rowIdxs.get(loc), rowAt(loc));
		}
		return new SparseMatrix(m);
	}

	public int rowSize() {
		return dims.get(1);
	}

}
