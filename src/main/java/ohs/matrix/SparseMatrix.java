package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.utils.ByteSize;
import ohs.utils.ByteSize.Type;
import ohs.utils.Generics;

/**
 * @author Heung-Seon Oh
 * 
 */
public class SparseMatrix extends ArrayList<SparseVector> implements Matrix {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3542638642565119292L;

	private int[][] idxs;

	private int[] rowIdxs;

	private double[][] vals;

	public SparseMatrix() {

	}

	public SparseMatrix(CounterMap<Integer, Integer> cm) {
		int size = cm.size();
		rowIdxs = new int[size];

		int loc = 0;
		for (int key : cm.keySet()) {
			rowIdxs[loc] = key;
			add(new SparseVector(cm.getCounter(key)));
			loc++;
		}
		sortRowIndexes();
		unwrapValues();
	}

	public SparseMatrix(int[] rowIdxs, SparseVector[] rows) {
		this.rowIdxs = rowIdxs;
		ensureCapacity(rows.length);
		for (SparseVector row : rows) {
			add(row);
		}
		unwrapValues();
	}

	public SparseMatrix(List<Integer> rowIdxs, List<SparseVector> rows) {
		this.rowIdxs = new int[rowIdxs.size()];
		ensureCapacity(rows.size());
		for (int i = 0; i < rows.size(); i++) {
			this.rowIdxs[i] = rowIdxs.get(i);
			add(rows.get(i));
		}
		unwrapValues();
	}

	public SparseMatrix(List<SparseVector> rows) {
		this.rowIdxs = ArrayUtils.range(rows.size());
		ensureCapacity(rows.size());
		for (int i = 0; i < rows.size(); i++) {
			add(rows.get(i));
		}
		unwrapValues();
	}

	public SparseMatrix(Map<Integer, SparseVector> m) {
		rowIdxs = new int[m.size()];
		ensureCapacity(m.size());
		int loc = 0;

		for (int rowIdx : m.keySet()) {
			rowIdxs[loc] = rowIdx;
			add(m.get(rowIdx));
			loc++;
		}
		sortRowIndexes();
		unwrapValues();
	}

	public SparseMatrix(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public SparseMatrix(SparseVector[] rows) {
		this.rowIdxs = ArrayUtils.range(rows.length);
		ensureCapacity(rows.length);
		for (SparseVector row : rows) {
			add(row);
		}
		unwrapValues();
	}

	public SparseMatrix(String fileName) throws Exception {
		readObject(fileName);
	}

	@Override
	public void add(double value) {
		for (SparseVector row : this) {
			row.add(value);
		}
	}

	@Override
	public int[] argMax() {
		int[] loc = argMaxLoc();
		return indexAt(loc[0], loc[1]);
	}

	public int[] argMaxLoc() {
		double max = -Double.MAX_VALUE;
		int x = -1;
		int y = -1;

		for (int i = 0; i < rowSize(); i++) {
			SparseVector row = rowAt(i);
			int j = row.argMinLoc();
			double v = row.valueAt(j);

			if (max < v) {
				max = v;
				x = i;
				y = j;
			}
		}
		return new int[] { x, y };
	}

	@Override
	public int[] argMin() {
		int[] loc = argMinLoc();
		return indexAt(loc[0], loc[1]);
	}

	public int[] argMinLoc() {
		double min = Double.MAX_VALUE;
		int x = -1;
		int y = -1;

		for (int i = 0; i < rowSize(); i++) {
			SparseVector row = rowAt(i);
			int j = row.argMinLoc();
			double v = row.valueAt(j);

			if (min > v) {
				min = v;
				x = i;
				y = j;
			}
		}
		return new int[] { x, y };
	}

	@Override
	public ByteSize byteSize() {
		long bytes = 0;
		for (SparseVector row : this) {
			bytes += row.byteSize().getBytes();
		}
		bytes += Integer.BYTES * rowIdxs.length;
		return new ByteSize(bytes);
	}

	@Override
	public int colSize() {
		int col_size = 0;
		int size = 0;
		for (SparseVector row : this) {
			col_size = Math.max(col_size, ArrayMath.max(row.indexes()));
			size += row.size();
		}

		if (size > 0) {
			col_size += 1;
		}
		return col_size;
	}

	@Override
	public SparseVector column(int j) {
		List<Integer> idxs = Generics.newArrayList(size());
		List<Double> vals = Generics.newArrayList(size());

		for (int m = 0; m < size(); m++) {
			int i = rowIdxs[m];
			SparseVector row = get(m);
			int loc = row.location(j);
			if (loc < 0) {
				continue;
			}
			idxs.add(i);
			vals.add(row.valueAt(loc));
		}
		return new SparseVector(idxs, vals);
	}

	public SparseMatrix copy() {
		return copy(false);
	}

	public SparseMatrix copy(boolean copy_template) {
		SparseVector[] rows = new SparseVector[rowSize()];
		for (int i = 0; i < rowSize(); i++) {
			rows[i] = row(i).copy(copy_template);
		}
		return new SparseMatrix(rows);
	}

	public Map<Integer, SparseVector> entries() {
		Map<Integer, SparseVector> ret = new HashMap<Integer, SparseVector>();
		for (int i = 0; i < rowIdxs.length; i++) {
			ret.put(rowIdxs[i], get(i));
		}
		return ret;
	}

	@Override
	public int indexAt(int loc) {
		return rowIdxs[loc];
	}

	public int[] indexAt(int i, int j) {
		return new int[] { rowIdxs[i], rowAt(i).indexAt(j) };
	}

	public int[][] indexes() {
		int[][] ret = new int[size()][];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = get(i).indexes();
		}
		return ret;
	}

	@Override
	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ SparseMatrix Info ]\n");
		sb.append(String.format("rows:\t[%d]\n", rowSize()));
		sb.append(String.format("cols:\t[%d]\n", colSize()));
		sb.append(String.format("size:\t[%d]\n", sizeOfEntries()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public int locationAtRow(int i) {
		return Arrays.binarySearch(rowIdxs, i);
	}

	@Override
	public double max() {
		return ArrayMath.max(vals);
	}

	@Override
	public double min() {
		return ArrayMath.min(vals);
	}

	@Override
	public void multiply(double value) {
		for (SparseVector row : this) {
			row.multiply(value);
		}
	}

	@Override
	public void normalizeColumns() {
		SparseVector columnSums = sumColumns();
		for (int i = 0; i < size(); i++) {
			SparseVector row = get(i);
			for (int j = 0; j < row.size(); j++) {
				int colid = row.indexAt(j);
				double value = row.valueAt(j);
				double sum = columnSums.value(colid);
				if (sum != 0) {
					row.setAt(j, colid, value / sum);
				}
			}
			row.summation();
		}
	}

	@Override
	public void normalizeRows() {
		for (Vector row : this) {
			row.normalizeAfterSummation();
		}
	}

	@Override
	public double prob(int i, int j) {
		return row(i).prob(j);
	}

	public double probAt(int i, int j) {
		return rowAt(i).probAt(j);
	}

	private int qPartition(int low, int high) {
		// First element
		// int pivot = a[low];

		// Middle element
		// int middle = (low + high) / 2;

		int i = low - 1;
		int j = high + 1;

		// ascending order
		int randomIndex = (int) (Math.random() * (high - low)) + low;
		int pivotValue = rowIdxs[randomIndex];

		while (i < j) {
			i++;
			while (rowIdxs[i] < pivotValue) {
				i++;
			}

			j--;
			while (rowIdxs[j] > pivotValue) {
				j--;
			}

			if (i < j) {
				swapRows(i, j);
			}
		}
		return j;
	}

	private void qSort(int low, int high) {
		if (low >= high)
			return;
		int p = qPartition(low, high);
		qSort(low, p);
		qSort(p + 1, high);
	}

	private void quicksort() {
		qSort(0, rowIdxs.length - 1);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		ensureCapacity(size);
		rowIdxs = new int[size];

		for (int i = 0; i < size; i++) {
			rowIdxs[i] = ois.readInt();
			add(new SparseVector(ois));
		}
	}

	@Override
	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	@Override
	public SparseVector row(int i) {
		SparseVector ret = new SparseVector(0);
		int loc = locationAtRow(i);
		if (loc > -1) {
			ret = get(loc);
		}
		return ret;
	}

	@Override
	public SparseVector rowAt(int loc) {
		return get(loc);
	}

	@Override
	public int[] rowIndexes() {
		return rowIdxs;
	}

	@Override
	public SparseMatrix rows(int size) {
		return rows(0, size);
	}

	@Override
	public SparseMatrix rows(int start, int size) {
		int[] idxs = new int[size];
		SparseVector[] rows = new SparseVector[size];
		int loc = 0;
		for (int i = start; i < start + size; i++) {
			idxs[loc] = rowIdxs[i];
			rows[loc] = rowAt(i);
			loc++;
		}
		return new SparseMatrix(idxs, rows);
	}

	@Override
	public SparseMatrix rows(int[] is) {
		int[] idxs = new int[is.length];
		SparseVector[] rows = new SparseVector[is.length];
		int loc = 0;
		for (int i : is) {
			idxs[loc] = rowIdxs[i];
			rows[loc] = rows[i];
			loc++;
		}
		return new SparseMatrix(idxs, rows);
	}

	@Override
	public int rowSize() {
		return rowIdxs.length;
	}

	@Override
	public void set(int i, int j, double value) {
		int rl = locationAtRow(i);
		if (rl > -1) {
			SparseVector row = rowAt(rl);
			int cl = row.location(j);
			if (cl > -1) {
				row.setAt(cl, value);
			}
		}
	}

	@Override
	public void setAll(double value) {
		for (SparseVector row : this) {
			row.setAll(value);
		}
	}

	public void setRow(int loc, int i, SparseVector x) {
		rowIdxs[loc] = i;
		set(loc, x);
	}

	public void setRow(int i, SparseVector x) {
		int loc = locationAtRow(i);
		if (loc > -1) {
			setRowAt(loc, x);
		}
	}

	@Override
	public void setRow(int i, Vector x) {
		int loc = locationAtRow(i);
		if (loc > -1) {
			setRowAt(loc, x);
		}
	}

	@Override
	public void setRowAt(int loc, Vector x) {
		set(loc, (SparseVector) x);
		vals[loc] = x.values();
	}

	@Override
	public void setRows(Vector[] rows) {
		this.clear();
		this.ensureCapacity(rows.length);
		;

		for (Vector row : rows) {
			add((SparseVector) row);
		}
	}

	@Override
	public int sizeOfEntries() {
		int ret = 0;
		for (SparseVector row : this) {
			ret += row.size();
		}
		return ret;
	}

	public void sortRowIndexes() {
		quicksort();
	}

	@Override
	public double sum() {
		return sumRows().sum();
	}

	@Override
	public SparseVector sumColumns() {
		Counter<Integer> c = Generics.newCounter();
		for (int m = 0; m < size(); m++) {
			SparseVector row = get(m);
			for (int n = 0; n < row.size(); n++) {
				c.incrementCount(row.indexAt(n), row.valueAt(n));
			}
		}
		return new SparseVector(c);
	}

	@Override
	public SparseVector sumRows() {
		SparseVector ret = new SparseVector(size());
		for (int i = 0; i < rowIdxs.length; i++) {
			SparseVector row = rowAt(i);
			row.summation();
			ret.addAt(i, rowIdxs[i], row.sum());
		}
		return ret;
	}

	public void swapRows(int i, int j) {
		int temp1 = rowIdxs[i];
		int temp2 = rowIdxs[j];
		rowIdxs[i] = temp2;
		rowIdxs[j] = temp1;

		SparseVector temp3 = get(i);
		SparseVector temp4 = get(j);

		set(i, temp4);
		set(j, temp3);
	}

	public DenseMatrix toDenseMatrix() {
		int row_size = ArrayMath.max(rowIdxs) + 1;
		int col_size = ArrayMath.max(idxs) + 1;
		return toDenseMatrix(row_size, col_size);
	}

	public DenseMatrix toDenseMatrix(int row_size, int col_size) {
		DenseMatrix ret = new DenseMatrix(row_size, col_size);

		for (int i = 0; i < rowIdxs.length; i++) {
			int rowIdx = rowIdxs[i];
			SparseVector sv = get(i);
			DenseVector dv = ret.row(rowIdx);

			for (int j = 0; j < sv.size(); j++) {
				dv.add(sv.indexAt(j), sv.valueAt(j));
			}
		}
		return ret;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("[rows:\t%d]\n", rowSize()));
		sb.append(String.format("[cols:\t%d]\n", colSize()));
		sb.append(String.format("[size:\t%d]\n", sizeOfEntries()));
		sb.append(String.format("[mem:\t%f MBs]\n", byteSize().getSize(Type.MEGA)));
		for (int i = 0; i < rowIdxs.length && i < 15; i++) {
			sb.append(String.format("%dth: %d, %s\n", i + 1, rowIdxs[i], get(i)));
		}
		return sb.toString().trim();
	}

	@Override
	public SparseMatrix transpose() {
		return new SparseMatrix(VectorUtils.toCounterMap(this).invert());
	}

	@Override
	public void unwrapValues() {
		idxs = new int[size()][];
		vals = new double[size()][];
		for (int i = 0; i < vals.length; i++) {
			idxs[i] = get(i).indexes();
			vals[i] = get(i).values();
		}
	}

	@Override
	public double value(int i, int j) {
		return row(i).value(j);
	}

	public double valueAt(int i, int j) {
		return rowAt(i).valueAt(j);
	}

	@Override
	public double[][] values() {
		return vals;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(size());
		for (int i = 0; i < size(); i++) {
			oos.writeInt(rowIdxs[i]);
			get(i).writeObject(oos);
		}
	}

	@Override
	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
