package ohs.math;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.matrix.Matrix;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.matrix.Vector;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public class VectorUtils {
	public static double copy(Counter<Integer> a, double[] b) {
		double sum = 0;
		for (Entry<Integer, Double> e : a.entrySet()) {
			int i = e.getKey();
			b[e.getKey()] = e.getValue();
			sum += b[e.getKey()];
		}
		return sum;
	}

	public static double copy(CounterMap<Integer, Integer> a, double[][] b) {
		double sum = 0;
		for (int i : a.keySet()) {
			sum += copy(a.getCounter(i), b[i]);
		}
		return sum;
	}

	public static double copy(DenseTensor a, DenseTensor b) {
		double sum = 0;
		for (int i = 0; i < a.size(); i++) {
			sum = copy(a.get(i), b.get(i));
		}
		return sum;
	}

	/**
	 * @param a
	 * @param m
	 *            start of a
	 * @param b
	 * @param n
	 *            start of b
	 * @param size
	 * @return
	 */
	public static double copy(DenseVector a, int m, DenseVector b, int n, int size) {
		double sum = ArrayUtils.copy(a.values(), m, b.values(), n, size);
		b.setSum(sum);
		return sum;
	}

	public static double copy(Matrix a, Matrix b) {
		double sum = 0;
		if (VectorChecker.isSparse(a) && VectorChecker.isSparse(b)) {
			for (int i = 0; i < a.rowSize(); i++) {
				sum += copy(a.row(i), b.row(i));
			}
		} else if (!VectorChecker.isSparse(a) && !VectorChecker.isSparse(b)) {
			for (int i = 0; i < a.rowSize(); i++) {
				sum += copy(a.row(i), b.row(i));
			}
		} else {
			throw new IllegalArgumentException();
		}

		return sum;
	}

	public static double copy(Vector a, Vector b) {
		if (VectorChecker.isSparse(a) && VectorChecker.isSparse(b)) {
			ArrayUtils.copy(a.indexes(), b.indexes());
			ArrayUtils.copy(a.values(), b.values());
			b.setSum(a.sum());
		} else if (!VectorChecker.isSparse(a) && !VectorChecker.isSparse(b)) {
			ArrayUtils.copy(a.values(), b.values());
			b.setSum(a.sum());
		} else {
			throw new IllegalArgumentException();
		}
		return b.sum();
	}

	public static double copyColumn(DenseMatrix a, int j, DenseVector b) {
		double sum = ArrayUtils.copyColumn(a.values(), j, b.values());
		b.setSum(sum);
		return sum;
	}

	public static double copyRows(DenseMatrix a, DenseVector b) {
		return copyRows(a, 0, a.rowSize(), b);
	}

	public static double copyRows(DenseMatrix a, int start, int end, DenseVector b) {
		double sum = ArrayUtils.copyRows(a.values(), start, end, b.values());
		b.setSum(sum);
		return sum;
	}

	public static double copyRows(DenseVector[] a, DenseVector b) {
		return copyRows(new DenseMatrix(a), b);
	}

	public static boolean enlarge(DenseMatrix a, int new_row_size, int new_col_size) {
		boolean ret = false;
		if (new_row_size > a.rowSize() || new_col_size > a.colSize()) {
			double[][] v = new double[new_row_size][new_col_size];
			a.setValues(v);
			ret = true;
		}
		return ret;
	}

	public static SparseVector freqOfFreq(DenseVector x) {
		Counter<Integer> counter = new Counter<Integer>();
		for (int i = 0; i < x.size(); i++) {
			int freq = (int) x.value(i);
			counter.incrementCount(freq, 1);
		}
		SparseVector ret = toSparseVector(counter);
		return ret;
	}

	public static DenseMatrix joinColumns(DenseMatrix a, DenseMatrix b) {
		List<DenseVector> rows = Generics.newArrayList(a.rowSize() + b.rowSize());

		for (Vector row : a) {
			rows.add((DenseVector) row);
		}

		for (Vector row : b) {
			rows.add((DenseVector) row);
		}

		return new DenseMatrix(rows);
	}

	public static void subVector(SparseVector x, int[] indexSet) {
		List<Integer> indexList = new ArrayList<Integer>();
		List<Double> valueList = new ArrayList<Double>();
		double sum = 0;

		for (int index : indexSet) {
			int loc = x.location(index);
			if (loc < 0) {
				continue;
			}

			double value = x.valueAt(loc);
			indexList.add(index);
			valueList.add(value);
			sum += value;
		}

		int[] indexes = new int[indexList.size()];
		double[] values = new double[valueList.size()];

		ArrayUtils.copyIntegers(indexList, indexes);
		ArrayUtils.copyDoubles(valueList, values);

		x.setIndexes(indexes);
		x.setValues(values);
		x.setSum(sum);
	}

	public static SparseVector subVectorTo(SparseVector x, int[] subset) {
		SparseVector ret = x.copy();
		subVector(x, subset);
		return ret;
	}

	public static SparseMatrix symmetric(SparseMatrix a) {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap(a.rowSize());

		for (int m = 0; m < a.rowSize(); m++) {
			int i = a.indexAt(m);
			SparseVector row = a.rowAt(m);
			for (int n = 0; n < row.size(); n++) {
				int j = row.indexAt(n);
				double v = row.valueAt(n);
				cm.incrementCount(i, j, v);
				if (i != j) {
					cm.incrementCount(j, i, v);
				}
			}
		}
		return new SparseMatrix(cm);
	}

	public static Counter<String> toCounter(Counter<Integer> x, Indexer<String> indexer) {
		Counter<String> ret = new Counter<String>();
		for (int index : x.keySet()) {
			double value = x.getCount(index);
			String obj = indexer.getObject(index);
			if (obj == null) {
				continue;
			}
			ret.setCount(obj, value);
		}
		return ret;
	}

	public static Counter<String> toCounter(DenseVector x, Indexer<String> indexer) {
		Counter<String> ret = new Counter<String>();
		for (int i = 0; i < x.size(); i++) {
			double value = x.value(i);
			if (value == 0) {
				continue;
			}
			String obj = indexer.getObject(i);
			ret.setCount(obj, value);
		}
		return ret;
	}

	public static Counter<Integer> toCounter(int[] x) {
		Counter<Integer> ret = Generics.newCounter();
		for (int idx : x) {
			ret.incrementCount(idx, 1);
		}
		return ret;
	}

	public static Counter<Integer> toCounter(IntegerArray x) {
		Counter<Integer> ret = Generics.newCounter();
		for (int i = 0; i < x.size(); i++) {
			ret.incrementCount(x.get(i), 1);
		}
		return ret;
	}

	public static Counter<Integer> toCounter(IntegerMatrix X) {
		Counter<Integer> ret = Generics.newCounter();
		for (int i = 0; i < X.size(); i++) {
			IntegerArray x = X.get(i);
			for (int j = 0; j < x.size(); j++) {
				ret.incrementCount(x.get(j), 1);
			}
		}

		double cnt = ret.totalCount();
		int size = ret.size();

		return ret;
	}

	public static Counter<Integer> toCounter(SparseMatrix X) {
		Counter<Integer> ret = new Counter<Integer>();
		for (SparseVector x : X) {
			for (int i = 0; i < x.size(); i++) {
				int idx = x.indexAt(i);
				if (idx < 0) {
					continue;
				}
				double value = x.valueAt(i);
				ret.incrementCount(idx, value);
			}
		}
		return ret;
	}

	public static Counter<Integer> toCounter(SparseVector x) {
		Counter<Integer> ret = new Counter<Integer>();
		for (int i = 0; i < x.size(); i++) {
			int index = x.indexAt(i);
			if (index < 0) {
				continue;
			}
			double value = x.valueAt(i);
			ret.incrementCount(index, value);
		}
		return ret;
	}

	public static Counter<String> toCounter(SparseVector x, Indexer<String> indexer) {
		Counter<String> ret = new Counter<String>();
		for (int i = 0; i < x.size(); i++) {
			int index = x.indexAt(i);

			if (index < 0 || index >= indexer.size()) {
				continue;
			}

			double value = x.valueAt(i);
			String obj = indexer.getObject(index);

			if (obj == null) {
				continue;
			}
			ret.incrementCount(obj, value);
		}

		return ret;
	}

	public static Counter<String> toCounter(String s) {
		Counter<String> ret = new Counter<String>();
		for (String tok : s.substring(1, s.length() - 1).split(" ")) {
			String[] two = tok.split(":");
			ret.setCount(two[0], Double.parseDouble(two[1]));
		}
		return ret;
	}

	public static CounterMap<String, String> toCounterMap(CounterMap<Integer, Integer> cm, Indexer<String> rowIndexer,
			Indexer<String> columnIndexer) {
		CounterMap<String, String> ret = new CounterMap<String, String>();
		for (int rowId : cm.keySet()) {
			String rowStr = null;
			if (rowIndexer == null) {
				rowStr = rowId + "";
			} else {
				rowStr = rowIndexer.getObject(rowId);
			}

			for (Entry<Integer, Double> entry : cm.getCounter(rowId).entrySet()) {
				int colId = entry.getKey();
				String colStr = null;

				if (columnIndexer == null) {
					colStr = colId + "";
				} else {
					colStr = columnIndexer.getObject(colId);
				}

				double value = entry.getValue();
				ret.incrementCount(rowStr, colStr, value);
			}
		}
		return ret;
	}

	public static CounterMap<String, String> toCounterMap(DenseMatrix m, Indexer<String> rowIndexer,
			Indexer<String> columnIndexer) {
		CounterMap<String, String> ret = new CounterMap<String, String>();

		for (int i = 0; i < rowIndexer.size(); i++) {
			String key1 = null;
			if (rowIndexer == null) {
				key1 = i + "";
			} else {
				key1 = rowIndexer.getObject(i);
			}

			for (int j = 0; j < columnIndexer.size(); j++) {
				String key2 = null;

				if (columnIndexer == null) {
					key2 = j + "";
				} else {
					key2 = columnIndexer.getObject(j);
				}

				double val = m.value(i, j);
				if (val != 0) {
					ret.incrementCount(key1, key2, val);
				}
			}
		}
		return ret;
	}

	public static CounterMap<Integer, Integer> toCounterMap(SparseMatrix sm) {
		CounterMap<Integer, Integer> ret = Generics.newCounterMap();
		for (int i = 0; i < sm.rowSize(); i++) {
			ret.setCounter(sm.indexAt(i), toCounter(sm.rowAt(i)));
		}
		return ret;
	}

	public static CounterMap<String, String> toCounterMap(SparseMatrix sm, Indexer<String> rowIndexer,
			Indexer<String> colIndexer) {
		CounterMap<String, String> ret = new CounterMap<String, String>();
		for (int i = 0; i < sm.rowSize(); i++) {
			int row_id = sm.indexAt(i);
			String rs = null;

			if (rowIndexer == null) {
				rs = row_id + "";
			} else {
				rs = rowIndexer.getObject(row_id);
			}

			SparseVector row = sm.rowAt(i);

			for (int j = 0; j < row.size(); j++) {
				int colId = row.indexAt(j);
				double value = row.valueAt(j);
				String cs = null;

				if (colIndexer == null) {
					cs = colId + "";
				} else {
					cs = colIndexer.getObject(colId);
				}
				ret.incrementCount(rs, cs, value);
			}
		}
		return ret;
	}

	public static DenseMatrix toDenseMatrix(CounterMap<String, String> cm, Indexer<String> rowIndexer,
			Indexer<String> colIndexer, boolean add_if_unseen) {

		DenseMatrix ret = new DenseMatrix(rowIndexer.size(), colIndexer.size());

		for (String rowKey : cm.keySet()) {
			int i = -1;

			if (add_if_unseen) {
				i = rowIndexer.getIndex(rowKey);
			} else {
				i = rowIndexer.indexOf(rowKey);
			}

			SparseVector row = toSparseVector(cm.getCounter(rowKey), colIndexer, add_if_unseen);

			if (i < 0 || row.size() == 0) {
				continue;
			}

			for (int k = 0; k < row.size(); k++) {
				int j = row.indexAt(k);
				double v = row.valueAt(k);
				ret.add(i, j, v);
			}
		}
		return ret;
	}

	public static DenseVector toDenseVector(Counter<String> x, Indexer<String> indexer) {
		DenseVector ret = new DenseVector(indexer.size());

		for (Entry<String, Double> e : x.entrySet()) {
			String key = e.getKey();
			double v = e.getValue();
			int i = indexer.indexOf(key);
			if (i < 0) {
				continue;
			}
			ret.add(i, v);
		}
		return ret;
	}

	public static String toRankedString(SparseVector x, Indexer<String> indexer) {
		StringBuffer sb = new StringBuffer();

		x.sortValues();
		x.reverse();

		for (int i = 0; i < x.size(); i++) {
			int index = x.indexAt(i);
			int rank = (int) x.valueAt(i);
			sb.append(String.format(" %s:%d", indexer.getObject(index), rank));
		}
		return sb.toString().trim();
	}

	public static SparseMatrix toSparseMatrix(CounterMap<Integer, Integer> cm) {
		return new SparseMatrix(cm);
	}

	public static SparseMatrix toSparseMatrix(CounterMap<String, String> cm, Indexer<String> rowIndexer,
			Indexer<String> colIndexer, boolean add_if_unseen) {

		List<Integer> rowIdxs = Generics.newArrayList(cm.size());
		List<SparseVector> rows = Generics.newArrayList(cm.size());

		for (String rowKey : cm.keySet()) {
			int rowid = -1;
			if (add_if_unseen) {
				rowid = rowIndexer.getIndex(rowKey);
			} else {
				rowid = rowIndexer.indexOf(rowKey);
			}

			SparseVector row = toSparseVector(cm.getCounter(rowKey), colIndexer, add_if_unseen);

			if (rowid < 0 || row.size() == 0) {
				continue;
			}

			rowIdxs.add(rowid);
			rows.add(row);
		}

		SparseMatrix ret = new SparseMatrix(rowIdxs, rows);
		ret.sortRowIndexes();
		return ret;
	}

	public static SparseVector toSparseVector(Counter<Integer> a) {
		return new SparseVector(a);
	}

	public static SparseVector toSparseVector(Counter<String> x, Indexer<String> indexer) {
		return toSparseVector(x, indexer, false);
	}

	public static SparseVector toSparseVector(Counter<String> x, Indexer<String> indexer, boolean add_if_unseen) {
		List<Integer> idxs = Generics.newArrayList(x.size());
		List<Double> vals = Generics.newArrayList(x.size());

		for (Entry<String, Double> e : x.entrySet()) {
			String key = e.getKey();
			double value = e.getValue();
			int idx = indexer.indexOf(key);
			if (idx < 0) {
				if (add_if_unseen) {
					idx = indexer.getIndex(key);
				} else {
					continue;
				}
			}
			idxs.add(idx);
			vals.add(value);
		}
		return new SparseVector(idxs, vals);
	}

	public static SparseVector toSparseVector(IntegerArray a) {
		Counter<Integer> c = Generics.newCounter();
		for (int i : a) {
			c.incrementCount(i, 1);
		}
		return new SparseVector(c);
	}

	public static SparseVector toSparseVector(List<String> x, Indexer<String> indexer, boolean add_if_unseen) {
		Counter<Integer> ret = new Counter<Integer>();
		for (String item : x) {
			int index = indexer.indexOf(item);
			if (index < 0) {
				if (add_if_unseen) {
					index = indexer.getIndex(item);
				} else {
					continue;
				}
			}
			ret.incrementCount(index, 1);
		}
		return toSparseVector(ret);
	}

	public static SparseVector toSparseVector(SparseVector Q) {
		return toSparseVector(toCounter(Q));
	}

	public static SparseVector toSparseVector(SparseVector x, Indexer<String> oldIndexer, Indexer<String> newIndexer) {
		IntegerArray idxs = new IntegerArray();
		DoubleArray vals = new DoubleArray();

		for (int i = 0; i < x.size(); i++) {
			int idx1 = x.indexAt(i);
			double val = x.valueAt(i);
			String key = oldIndexer.getObject(idx1);
			int idx2 = newIndexer.indexOf(key);

			if (idx2 == -1) {
				continue;
			}

			idxs.add(idx2);
			vals.add(val);
		}
		idxs.trimToSize();
		vals.trimToSize();
		SparseVector ret = new SparseVector(idxs, vals);
		return ret;
	}

	public static SparseVector toSparseVector(String s, Indexer<String> indexer) {
		return toSparseVector(toCounter(s), indexer);
	}

	public static SparseMatrix toSpasreMatrix(CounterMap<Integer, Integer> cm) {
		int[] rowIds = new int[cm.keySet().size()];
		SparseVector[] rows = new SparseVector[rowIds.length];
		int loc = 0;
		for (int index : cm.keySet()) {
			rowIds[loc] = index;
			rows[loc] = toSparseVector(cm.getCounter(index));
			loc++;
		}
		SparseMatrix ret = new SparseMatrix(rowIds, rows);
		ret.sortRowIndexes();
		return ret;
	}

	public static String toSVMFormat(SparseVector x, NumberFormat nf) {
		if (nf == null) {
			nf = NumberFormat.getInstance();
			nf.setMinimumFractionDigits(8);
			nf.setGroupingUsed(false);
		}

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < x.size(); i++) {
			sb.append(String.format(" %d:%s", x.indexAt(i), nf.format(x.valueAt(i))));
		}
		return sb.toString();
	}
}
