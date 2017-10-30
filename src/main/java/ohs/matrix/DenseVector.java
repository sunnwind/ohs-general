package ohs.matrix;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.NumberFormat;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.types.generic.Pair;
import ohs.types.number.DoubleArray;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

public class DenseVector implements Vector {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1185683442330052104L;

	private double sum;

	private double[] vals;

	public DenseVector() {

	}

	public DenseVector(Collection<DenseVector> vs) {
		this(vs.toArray(new DenseVector[vs.size()]));
	}

	public DenseVector(DenseVector[] dvs) {
		int size = 0;
		for (int i = 0; i < dvs.length; i++) {
			size += dvs[i].size();
		}
		vals = new double[size];
		sum = 0;

		for (int i = 0, k = 0; i < dvs.length; i++) {
			DenseVector dv = dvs[i];
			for (int j = 0; j < dv.size(); j++) {
				vals[k] = dv.value(j);
				sum += vals[k];
				k++;
			}
		}
	}

	public DenseVector(double[] a) {
		this.vals = a;
		summation();
	}

	public DenseVector(DoubleArray a) {
		this(a.values());
	}

	public DenseVector(int size) {
		vals = new double[size];
	}

	public DenseVector(int size, double v) {
		this.vals = new double[size];
		setAll(v);
	}

	public DenseVector(int[] vals) {
		this.vals = new double[vals.length];

		for (int i = 0; i < vals.length; i++) {
			this.vals[i] = vals[i];
			sum += vals[i];
		}
	}

	public DenseVector(List<Double> vals) {
		this(vals.size());

		for (int i = 0; i < vals.size(); i++) {
			this.vals[i] = ((Double) vals.get(i)).doubleValue();
			sum += this.vals[i];
		}
	}

	public DenseVector(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public DenseVector(String fileName) throws Exception {
		readObject(fileName);
	}

	@Override
	public void add(double v) {
		sum = ArrayMath.add(vals, v, vals);
	}

	@Override
	public void add(int i, double v) {
		vals[i] += v;
		sum += v;
	}

	@Override
	public void addAt(int loc, double v) {
		add(loc, v);
	}

	@Override
	public void addAt(int loc, int i, double v) {
		new UnsupportedOperationException("unsupported");
	}

	@Override
	public int argMax() {
		return ArrayMath.argMax(vals);
	}

	@Override
	public int argMin() {
		return ArrayMath.argMin(vals);
	}

	@Override
	public ByteSize byteSize() {
		return new ByteSize(Double.BYTES * vals.length);
	}

	public void clear() {
		setAll(0);
		sum = 0;
	}

	@Override
	public DenseVector copy() {
		return copy(false);
	}

	public DenseVector copy(boolean shallow_copy) {
		DenseVector ret = null;
		if (shallow_copy) {
			ret = new DenseVector(vals.length);
		} else {
			ret = new DenseVector(ArrayUtils.copy(vals));
			ret.setSum(sum);
		}
		return ret;
	}

	public double[] copyValues() {
		return ArrayUtils.copy(vals);
	}

	@Override
	public int indexAt(int loc) {
		return loc;
	}

	@Override
	public int[] indexes() {
		new UnsupportedOperationException("unsupported");
		return null;
	}

	@Override
	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ DenseVector Info ]\n");
		sb.append(String.format("size:\t%d\n", size()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	@Override
	public void keepAbove(double cutoff) {
		for (int i = 0; i < vals.length; i++) {
			double value = vals[i];
			if (vals[i] < cutoff) {
				vals[i] -= value;
				sum -= value;
			}
		}
	}

	@Override
	public void keepTopN(int topN) {
		SparseVector x1 = toSparseVector();
		x1.keepTopN(topN);
		DenseVector x2 = x1.toDenseVector();
		this.vals = x2.values();
		this.sum = x2.sum();
	}

	@Override
	public int location(int index) {
		return index;
	}

	@Override
	public double max() {
		int index = argMax();
		double max = Double.NEGATIVE_INFINITY;
		if (index > -1) {
			max = vals[index];
		}
		return max;
	}

	@Override
	public double min() {
		int index = argMin();
		double max = Double.POSITIVE_INFINITY;
		if (index > -1) {
			max = vals[index];
		}
		return max;
	}

	@Override
	public double multiply(double factor) {
		return sum = ArrayMath.multiply(vals, factor, vals);
	}

	@Override
	public double multiply(int i, double factor) {
		vals[i] *= factor;
		return vals[i];
	}

	@Override
	public double multiplyAt(int loc, double factor) {
		return multiply(loc, factor);
	}

	@Override
	public double multiplyAt(int loc, int i, double factor) {
		new UnsupportedOperationException("unsupported");
		return 0;
	}

	@Override
	public double normalize() {
		return sum = ArrayMath.multiply(vals, 1f / sum, vals);
	}

	@Override
	public double normalizeAfterSummation() {
		return sum = ArrayMath.normalize(vals);
	}

	public List<Pair<Integer, Double>> pairs() {
		List<Pair<Integer, Double>> ret = Generics.newArrayList(size());
		for (int i = 0; i < size(); i++) {
			ret.add(Generics.newPair(i, value(i)));
		}
		return ret;
	}

	@Override
	public double prob(int index) {
		return vals[index] / sum;
	}

	@Override
	public double probAt(int loc) {
		return prob(loc);
	}

	@Override
	public void prune(Set<Integer> toRemove) {
		for (int i : toRemove) {
			double value = vals[i];
			vals[i] -= value;
			sum -= value;
		}
	}

	@Override
	public void pruneExcept(Set<Integer> toKeep) {
		for (int i = 0; i < vals.length; i++) {
			double value = vals[i];
			if (!toKeep.contains(i)) {
				vals[i] -= value;
				sum -= value;
			}
		}
	}

	@Override
	public SparseVector ranking() {
		return ranking(false);
	}

	public SparseVector ranking(boolean ascending) {
		SparseVector ret = toSparseVector().copy();
		ret.sortValues();

		for (int i = 0; i < ret.size(); i++) {
			double rank = i + 1;
			if (ascending) {
				rank = ret.size() - i;
			}
			ret.setAt(i, rank);
		}
		ret.sortIndexes();
		return ret;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		vals = FileUtils.readDoubleArray(ois);
		summation();
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	@Override
	public double set(int i, double v) {
		double ret = vals[i];
		vals[i] = v;
		return ret;
	}

	@Override
	public void setAll(double v) {
		sum = ArrayUtils.setAll(vals, v);
	}

	@Override
	public double setAt(int loc, double v) {
		return set(loc, v);
	}

	@Override
	public double setAt(int loc, int i, double v) {
		return setAt(loc, v);
	}

	@Override
	public void setIndexes(int[] indexes) {
		new UnsupportedOperationException("unsupported");
	}

	@Override
	public double setSum(double sum) {
		double ret = this.sum;
		this.sum = sum;
		return ret;
	}

	@Override
	public void setValues(double[] values) {
		this.vals = values;
	}

	@Override
	public int size() {
		return vals.length;
	}

	@Override
	public int sizeOfNonzero() {
		int ret = 0;
		for (int i = 0; i < vals.length; i++) {
			double value = vals[i];
			if (value == 0) {
				continue;
			}
			ret++;
		}
		return ret;
	}

	@Override
	public DenseVector subVector(int size) {
		return subVector(0, size);
	}

	@Override
	public DenseVector subVector(int start, int size) {
		DenseVector ret = new DenseVector(size);
		for (int i = start, j = 0; i < start + size; i++, j++) {
			ret.add(j, value(i));
		}
		return ret;
	}

	@Override
	public DenseVector subVector(int[] idxs) {
		DenseVector ret = new DenseVector(idxs.length);
		for (int j = 0; j < idxs.length; j++) {
			ret.add(j, value(idxs[j]));
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

	public DenseMatrix toDenseMatrix() {
		return new DenseMatrix(new DenseVector[] { this });
	}

	public SparseVector toSparseVector() {
		int[] newIndexes = new int[sizeOfNonzero()];
		double[] newValues = new double[newIndexes.length];
		double sum = 0;
		for (int i = 0, loc = 0; i < vals.length; i++) {
			double value = vals[i];
			if (value != 0) {
				newIndexes[loc] = i;
				newValues[loc] = value;
				sum += value;
				loc++;
			}
		}
		SparseVector ret = new SparseVector(newIndexes, newValues);
		ret.setSum(sum);
		return ret;
	}

	@Override
	public String toString() {
		return toString(20, true, null);
	}

	public String toString(int size, boolean sparse, NumberFormat nf) {
		if (nf == null) {
			nf = NumberFormat.getInstance();
			nf.setMinimumFractionDigits(4);
			nf.setGroupingUsed(false);
		}

		StringBuffer sb = new StringBuffer();
		sb.append(String.format("(%d/%d, %s) ->", sizeOfNonzero(), vals.length, nf.format(sum)));

		// int numPrint = 0;

		for (int i = 0, j = 0; i < vals.length; i++) {
			if (j == size) {
				break;
			}

			double v = vals[i];

			if (sparse) {
				if (v != 0) {
					if (v % 1 == 0) {
						sb.append(String.format(" %d:%d", i, (int) v));
					} else {
						sb.append(String.format(" %d:%s", i, nf.format(v)));
					}
					j++;
				}
			} else {
				if (v % 1 == 0) {
					sb.append(String.format(" %d:%d", i, (int) v));
				} else {
					sb.append(String.format(" %d:%s", i, nf.format(v)));
				}
				j++;
			}

		}

		return sb.toString();
	}

	@Override
	public double value(int index) {
		return vals[index];
	}

	@Override
	public double valueAt(int loc) {
		return value(loc);
	}

	@Override
	public double[] values() {
		return vals;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.writeDoubles(oos, vals);
	}

	@Override
	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
