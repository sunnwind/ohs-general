package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

/**
 * @author Heung-Seon Oh
 * 
 */
public class SparseVector implements Vector {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6671749703272005320L;

	private int[] idxs;

	private double[] vals;

	private double sum;

	public SparseVector() {

	}

	public SparseVector(Collection<Integer> idxs) {
		this(ArrayUtils.copyIntegers(idxs));
	}

	public SparseVector(Counter<Integer> c) {
		ensureSize(c.size());

		sum = 0;
		int loc = 0;
		for (Entry<Integer, Double> e : c.entrySet()) {
			idxs[loc] = e.getKey();
			vals[loc] = e.getValue();
			sum += vals[loc];
			loc++;
		}
		sortIndexes();
	}

	public SparseVector(int size) {
		this(new int[size], new double[size]);
	}

	public SparseVector(int[] idxs) {
		this(idxs, new double[idxs.length]);
	}

	public SparseVector(int[] idxs, double[] vals) {
		this.idxs = idxs;
		this.vals = vals;
		this.sum = ArrayMath.sum(vals);
	}

	public SparseVector(IntegerArray idxs, DoubleArray vals) {
		this(idxs.values(), vals.values());
	}

	public SparseVector(List<Integer> idxs, List<Double> vals) {
		this.idxs = new int[idxs.size()];
		this.vals = new double[vals.size()];
		sum = 0;
		for (int i = 0; i < idxs.size(); i++) {
			this.idxs[i] = idxs.get(i);
			this.vals[i] = vals.get(i);
			sum += vals.get(i);
		}
	}

	public SparseVector(Map<Integer, Double> c) {
		ensureSize(c.size());

		sum = 0;
		int loc = 0;

		for (Entry<Integer, Double> e : c.entrySet()) {
			idxs[loc] = e.getKey();
			vals[loc] = e.getValue();
			sum += vals[loc];
			loc++;
		}
		sortIndexes();
	}

	public SparseVector(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public SparseVector(SparseVector[] vs) {
		int size = 0;
		for (int i = 0; i < vs.length; i++) {
			size += vs[i].size();
		}
		ensureSize(size);

		sum = 0;
		for (int i = 0, k = 0; i < vs.length; i++) {
			SparseVector v = vs[i];
			for (int j = 0; j < v.size(); j++) {
				idxs[k] = v.indexAt(j);
				vals[k] = v.valueAt(j);
				sum += vals[k];
			}
		}
	}

	public SparseVector(String fileName) throws Exception {
		readObject(fileName);
	}

	@Override
	public void add(double value) {
		sum = ArrayMath.add(vals, value, vals);
	}

	@Override
	public void add(int i, double value) {
		int loc = location(i);
		if (loc > -1) {
			vals[loc] += value;
			sum += value;
		}
	}

	@Override
	public void addAt(int loc, double value) {
		vals[loc] += value;
		sum += value;
	}

	@Override
	public void addAt(int loc, int i, double value) {
		idxs[loc] = i;
		vals[loc] += value;
		sum += value;
	}

	@Override
	public int argMax() {
		return indexAt(argMaxLoc());
	}

	public int argMaxLoc() {
		return ArrayMath.argMax(vals);
	}

	@Override
	public int argMin() {
		return indexAt(argMinLoc());
	}

	public int argMinLoc() {
		return ArrayMath.argMin(vals);
	}

	@Override
	public ByteSize byteSize() {
		return new ByteSize(Double.BYTES * vals.length + Integer.BYTES * idxs.length);
	}

	@Override
	public SparseVector copy() {
		return copy(false);
	}

	public SparseVector copy(boolean copy_template) {
		SparseVector ret = null;
		if (copy_template) {
			ret = new SparseVector(vals.length);
		} else {
			ret = new SparseVector(ArrayUtils.copy(idxs), ArrayUtils.copy(vals));
			ret.setSum(sum);
		}
		return ret;
	}

	public int[] copyIndexes() {
		return ArrayUtils.copy(idxs);
	}

	public double[] copyValues() {
		return ArrayUtils.copy(vals);
	}

	private void ensureSize(int size) {
		idxs = new int[size];
		vals = new double[size];
	}

	@Override
	public int indexAt(int loc) {
		return idxs[loc];
	}

	@Override
	public int[] indexes() {
		return idxs;
	}

	@Override
	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ SparseVector Info ]\n");
		sb.append(String.format("size:\t%d\n", size()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	@Override
	public void keepAbove(double cutoff) {
		List<Integer> tis = Generics.newArrayList(idxs.length);
		List<Double> tvs = Generics.newArrayList(vals.length);
		for (int i = 0; i < vals.length; i++) {
			if (vals[i] < cutoff) {
				continue;
			}
			tis.add(idxs[i]);
			tvs.add(vals[i]);
		}

		idxs = new int[tis.size()];
		vals = new double[tis.size()];
		sum = 0;
		for (int i = 0; i < tis.size(); i++) {
			idxs[i] = tis.get(i);
			vals[i] = tvs.get(i);
			sum += vals[i];
		}
	}

	@Override
	public void keepTopN(int top_n) {
		if (size() > top_n) {
			sortValues();
			SparseVector sv = subVector(top_n);
			idxs = sv.indexes();
			vals = sv.values();
			sum = sv.sum();
			sortIndexes();
		}
	}

	/**
	 * 
	 * it should be called after calling sortByIndex
	 * 
	 * @param i
	 * @return
	 */
	@Override
	public int location(int i) {
		return Arrays.binarySearch(idxs, i);
	}

	@Override
	public double max() {
		return valueAt(argMaxLoc());
	}

	@Override
	public double min() {
		return valueAt(argMinLoc());
	}

	@Override
	public double multiply(double factor) {
		return sum = ArrayMath.multiply(vals, factor, vals);
	}

	@Override
	public double multiply(int i, double factor) {
		double ret = 0;
		int loc = location(i);
		if (loc > -1) {
			ret = multiplyAt(loc, factor);
		}
		return ret;
	}

	@Override
	public double multiplyAt(int loc, double factor) {
		vals[loc] *= factor;
		return vals[loc];
	}

	@Override
	public double multiplyAt(int loc, int i, double factor) {
		idxs[loc] = i;
		vals[loc] *= factor;
		return vals[loc];

	}

	@Override
	public double normalize() {
		return sum = ArrayMath.multiply(vals, 1f / sum, vals);
	}

	@Override
	public double normalizeAfterSummation() {
		return sum = ArrayMath.normalize(vals);
	}

	@Override
	public double prob(int i) {
		double ret = 0;
		int loc = location(i);
		if (loc > -1) {
			ret = probAt(loc);
		}
		return ret;
	}

	@Override
	public double probAt(int loc) {
		return vals[loc] / sum;
	}

	@Override
	public void prune(final Set<Integer> toRemove) {
		List<Integer> tis = Generics.newArrayList(idxs.length);
		List<Double> tvs = Generics.newArrayList(idxs.length);
		sum = 0;

		for (int i = 0; i < size(); i++) {
			int index = indexAt(i);
			double value = valueAt(i);
			if (toRemove.contains(index)) {
				continue;
			}
			tis.add(index);
			tvs.add(value);
			sum += value;
		}

		idxs = new int[tis.size()];
		vals = new double[tvs.size()];
		ArrayUtils.copyIntegers(tis, idxs);
		ArrayUtils.copyDoubles(tvs, vals);
	}

	@Override
	public void pruneExcept(final Set<Integer> toKeep) {
		IntegerArray is = new IntegerArray(size());
		DoubleArray vs = new DoubleArray(size());
		sum = 0;

		for (int i = 0; i < size(); i++) {
			int idx = indexAt(i);
			double val = valueAt(i);
			if (!toKeep.contains(idx)) {
				continue;
			}
			is.add(idx);
			vs.add(val);
			sum += val;
		}

		is.trimToSize();
		vs.trimToSize();

		idxs = is.values();
		vals = vs.values();
	}

	public int[] rankedIndexes() {
		sortValues();
		int[] ret = new int[size()];
		for (int i = 0; i < size(); i++) {
			ret[i] = indexAt(i);
		}
		sortIndexes();
		return ret;
	}

	@Override
	public SparseVector ranking() {
		return ranking(false);
	}

	public SparseVector ranking(boolean ascending) {
		SparseVector ret = copy();
		ret.sortValues();

		for (int i = 0; i < ret.size(); i++) {
			int index = ret.indexAt(i);
			double rank = i + 1;
			if (ascending) {
				rank = ret.size() - i;
			}
			ret.setAt(i, index, rank);
		}
		ret.sortIndexes();
		return ret;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		idxs = new int[size];
		vals = new double[size];
		sum = 0;
		for (int i = 0; i < size; i++) {
			idxs[i] = ois.readInt();
			vals[i] = ois.readDouble();
			sum += vals[i];
		}
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void removeZeros() {
		sum = 0;
		List<Integer> tis = Generics.newArrayList();
		List<Double> tvs = Generics.newArrayList();

		for (int i = 0; i < idxs.length; i++) {
			int index = idxs[i];
			double value = vals[i];
			if (value == 0) {
				continue;
			}
			sum += vals[i];
			tis.add(index);
			tvs.add(value);
		}

		idxs = new int[tis.size()];
		vals = new double[tvs.size()];

		ArrayUtils.copyIntegers(tis, idxs);
		ArrayUtils.copyDoubles(tvs, vals);

	}

	public void reverse() {
		int middle = idxs.length / 2;
		for (int i = 0; i < middle; i++) {
			int left = i;
			int right = idxs.length - i - 1;
			swap(left, right);
		}
	}

	@Override
	public double set(int i, double value) {
		int loc = location(i);
		double ret = 0;
		if (loc > -1) {
			ret = setAt(loc, value);
		}
		return ret;
	}

	@Override
	public void setAll(double value) {
		ArrayUtils.setAll(vals, value);
		sum = value * vals.length;
	}

	@Override
	public double setAt(int loc, double value) {
		double ret = vals[loc];
		vals[loc] = value;
		return ret;
	}

	@Override
	public double setAt(int loc, int i, double value) {
		double ret = vals[loc];
		idxs[loc] = i;
		vals[loc] = value;
		return ret;
	}

	@Override
	public void setIndexes(int[] indexes) {
		this.idxs = indexes;
	}

	@Override
	public double setSum(double sum) {
		double ret = sum;
		this.sum = sum;
		return ret;
	}

	@Override
	public void setValues(double[] values) {
		this.vals = values;
	}

	@Override
	public int size() {
		return idxs.length;
	}

	@Override
	public int sizeOfNonzero() {
		return ArrayUtils.sizeOfNonzero(vals);
	}

	public void sortIndexes() {
		ArrayUtils.quickSort(idxs, vals, true);
	}

	public void sortValues() {
		ArrayUtils.quickSort(idxs, vals, false);
	}

	public void sortValues(boolean descending) {
		ArrayUtils.quickSort(idxs, vals, false);

		if (!descending) {
			reverse();
		}
	}

	@Override
	public SparseVector subVector(int size) {
		return subVector(0, size);
	}

	@Override
	public SparseVector subVector(int start, int size) {
		SparseVector ret = new SparseVector(size);
		for (int i = start, j = 0; i < start + size; i++, j++) {
			ret.addAt(j, indexAt(i), valueAt(i));
		}
		return ret;
	}

	@Override
	public SparseVector subVector(int[] is) {
		SparseVector ret = new SparseVector(is.length);
		for (int j = 0, k = 0; j < is.length; j++, k++) {
			ret.addAt(k, indexAt(is[j]), valueAt(is[j]));
		}
		return ret;
	}

	@Override
	public double sum() {
		return sum;
	}

	@Override
	public double summation() {
		sum = ArrayMath.sum(vals);
		return sum;
	}

	private void swap(int i, int j) {
		ArrayUtils.swap(idxs, i, j);
		ArrayUtils.swap(vals, i, j);
	}

	public DenseVector toDenseVector() {
		return toDenseVector(ArrayMath.max(idxs) + 1);
	}

	public DenseVector toDenseVector(int size) {
		DenseVector ret = new DenseVector(size);
		for (int i = 0; i < idxs.length; i++) {
			ret.add(idxs[i], vals[i]);
		}
		return ret;
	}

	@Override
	public String toString() {
		return toString(false, 20, null);
	}

	public String toString(boolean vertical, int size, Indexer<String> indexer) {
		int tmp = (int) sum;
		StringBuffer sb = new StringBuffer();

		if (sum - tmp == 0) {
			sb.append(String.format("(%d, %d) ->", size(), tmp));
		} else {
			sb.append(String.format("(%d, %f) ->", size(), sum));
		}

		// sortByValue();

		if (vertical) {
			sb.append("\n");
		}

		for (int i = 0, j = 0; i < idxs.length; i++) {
			int idx = idxs[i];
			double val = vals[i];
			tmp = (int) val;
			if (j == size) {
				break;
			}

			if (indexer == null) {
				if (val != 0) {
					if (val - tmp == 0) {
						sb.append(String.format(" %d:%d", idx, tmp));
					} else {
						sb.append(String.format(" %d:%f", idx, val));
					}
					j++;
				}
			} else {
				if (val != 0) {
					if (val - tmp == 0) {
						sb.append(String.format(" %s:%d", indexer.getObject(idx), tmp));
					} else {
						sb.append(String.format(" %s:%f", indexer.getObject(idx), val));
					}
					j++;
				}
			}

			if (vertical) {
				sb.append("\n");
			}
		}

		// sortByIndex();

		return sb.toString().trim();
	}

	public String toSvmString() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size(); i++) {
			sb.append(String.format(" %d:%s", idxs[i], vals[i] + ""));
			if (i != size() - 1) {
				sb.append(" ");
			}
		}
		return sb.toString();
	}

	@Override
	public double value(int index) {
		double ret = 0;
		int loc = location(index);
		if (loc > -1) {
			ret = valueAt(loc);
		}
		return ret;
	}

	@Override
	public double valueAt(int loc) {
		return vals[loc];
	}

	@Override
	public double[] values() {
		return vals;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(vals.length);
		for (int i = 0; i < vals.length; i++) {
			oos.writeInt(idxs[i]);
			oos.writeDouble(vals[i]);
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
