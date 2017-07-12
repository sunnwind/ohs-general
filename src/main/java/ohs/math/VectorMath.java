package ohs.math;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.math.ArrayWokers.EqualColumnProductWorker2;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.Matrix;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.matrix.Vector;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * @author Heung-Seon Oh
 * 
 */
public class VectorMath {

	public static double abs(DenseVector a, DenseVector b) {
		double sum = ArrayMath.abs(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double add(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += add(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double add(DenseMatrix a, DenseMatrix b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += addAfterMultiply(a.row(i), 1, b.row(i), 1, c.row(i));
		}
		return sum;
	}

	public static double add(DenseMatrix a, DenseVector b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += addAfterMultiply(a.row(i), 1, b, 1, c.row(i));
		}
		return sum;
	}

	public static double add(DenseMatrix a, double b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += addAfterMultiply(a.row(i), 1, b, 1, c.row(i));
		}
		return sum;
	}

	public static double add(DenseVector a, DenseVector b) {
		double sum = ArrayMath.add(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double add(DenseVector a, DenseVector b, DenseVector c) {
		double sum = ArrayMath.addAfterMultiply(a.values(), 1, b.values(), 1, c.values());
		c.setSum(sum);
		return sum;
	}

	public static double add(DenseVector a, double b, DenseVector c) {
		double sum = ArrayMath.addAfterMultiply(a.values(), 1, b, 1, c.values());
		c.setSum(sum);
		return sum;
	}

	public static void add(Vector a, Counter<Integer> b) {
		addAfterMultiply(a, 1, b);
	}

	public static SparseVector add(Vector a, Vector b) {
		return add(new Vector[] { a, b });
	}

	public static SparseVector add(Vector[] a) {
		Counter<Integer> b = Generics.newCounter();
		add(a, b);
		return VectorUtils.toSparseVector(b);
	}

	public static void add(Vector[] a, Counter<Integer> b) {
		for (Vector v : a) {
			add(v, b);
		}
	}

	public static double addAfterMultiply(DenseMatrix a, double ac, DenseMatrix b, double bc, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += addAfterMultiply(a.row(i), ac, b.row(i), bc, c.row(i));
		}
		return sum;
	}

	public static double addAfterMultiply(DenseMatrix a, double ac, DenseVector b, double bc, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += addAfterMultiply(a.row(i), ac, b, bc, c.row(i));
		}
		return sum;
	}

	public static double addAfterMultiply(DenseVector a, DenseVector ac, DenseVector b, DenseVector bc, DenseVector c) {
		double sum = ArrayMath.addAfterMultiply(a.values(), ac.values(), b.values(), bc.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	public static double addAfterMultiply(DenseVector a, DenseVector ac, double b, DenseVector bc, DenseVector c) {
		double sum = ArrayMath.addAfterMultiply(a.values(), ac.values(), b, bc.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	public static double addAfterMultiply(DenseVector a, double ac, DenseVector b) {
		double sum = ArrayMath.addAfterMultiply(a.values(), ac, b.values());
		b.setSum(sum);
		return sum;
	}

	public static double addAfterMultiply(DenseVector a, double ac, DenseVector b, double bc, DenseVector c) {
		double sum = ArrayMath.addAfterMultiply(a.values(), ac, b.values(), bc, c.values());
		c.setSum(sum);
		return sum;
	}

	public static double addAfterMultiply(DenseVector a, double ac, double b, double bc, DenseVector c) {
		double sum = ArrayMath.addAfterMultiply(a.values(), ac, b, bc, c.values());
		c.setSum(sum);
		return sum;
	}

	public static double addAfterMultiply(SparseMatrix a, double ac, CounterMap<Integer, Integer> b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			int m = a.indexAt(i);
			sum += addAfterMultiply(a.rowAt(i), ac, b.getCounter(m));
		}
		return sum;
	}

	public static SparseMatrix addAfterMultiply(SparseMatrix a, double ac, SparseMatrix b, double bc) {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap();
		addAfterMultiply(a, ac, cm);
		addAfterMultiply(b, bc, cm);
		return VectorUtils.toSparseMatrix(cm);
	}

	public static double addAfterMultiply(SparseVector a, double ac, Counter<Integer> b) {
		double sum = 0;
		for (int i = 0; i < a.size(); i++) {
			int j = a.indexAt(i);
			double v = a.valueAt(i) * ac;
			b.incrementCount(j, v);
			sum += v;
		}
		return sum;
	}

	public static void addAfterMultiply(SparseVector a, double ac, Counter<Integer> b, double bc, Counter<Integer> c) {
		addAfterMultiply(a, ac, c);

		for (Entry<Integer, Double> e : b.entrySet()) {
			c.incrementCount(e.getKey(), bc * e.getValue().doubleValue());
		}
	}

	public static SparseVector addAfterMultiply(SparseVector a, double ac, SparseVector b, double bc) {
		Counter<Integer> c = Generics.newCounter(a.size() + b.size());
		addAfterMultiply(new SparseVector[] { a, b }, new double[] { ac, bc }, c);
		return VectorUtils.toSparseVector(c);
	}

	public static void addAfterMultiply(SparseVector a, double ac, SparseVector b, double bc, Counter<Integer> c) {
		addAfterMultiply(new SparseVector[] { a, b }, new double[] { ac, bc }, c);
	}

	public static SparseVector addAfterMultiply(SparseVector[] a, double[] b) {
		Counter<Integer> c = Generics.newCounter();
		addAfterMultiply(a, b, c);
		return VectorUtils.toSparseVector(c);
	}

	public static void addAfterMultiply(SparseVector[] a, double[] b, Counter<Integer> c) {
		for (int i = 0; i < a.length; i++) {
			addAfterMultiply(a[i], b[i], c);
		}
	}

	public static void addAfterMultiply(Vector a, double ac, Counter<Integer> b) {
		for (int loc = 0; loc < a.size(); loc++) {
			int i = a.indexAt(loc);
			b.setCount(i, ac * a.valueAt(loc) + b.getCount(i));
		}
	}

	public static double addAfterMultiplyByThreads(DenseMatrix a, double ac, DenseMatrix b, double bc, DenseMatrix c, int num_threads) {
		ArrayMath.addAfterMultiplyByThreads(a.values(), ac, b.values(), bc, c.values(), num_threads);
		return c.sumColumns().sum();
	}

	public static double addAfterOuterProduct(SparseVector a, DenseVector b, DenseMatrix c) {
		int i = 0;
		double v = 0;
		double out = 0;
		double sum = 0;

		for (int m = 0; m < a.size(); m++) {
			i = a.indexAt(m);
			v = a.valueAt(m);

			for (int j = 0; j < b.size(); j++) {
				out = v * b.value(j);
				c.add(i, j, out);
			}
		}

		for (int k = 0; k < c.rowSize(); k++) {
			sum += c.row(k).summation();
		}
		return sum;
	}

	public static double addAfterOuterProductByThreads(DenseVector a, DenseVector b, DenseMatrix c, int num_threads) {
		ArrayMath.addAfterOuterProductByThreads(a.values(), b.values(), c.values(), num_threads);
		return c.sumRows().sum();
	}

	public static double addAfterSquared(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += addAfterSquared(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double addAfterSquared(DenseVector a, DenseVector b) {
		double sum = ArrayMath.addAfterSquared(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static SparseVector average(Collection<SparseVector> a) {
		Counter<Integer> b = Generics.newCounter();
		average(a, b);
		return VectorUtils.toSparseVector(b);
	}

	public static void average(Collection<SparseVector> a, Counter<Integer> b) {
		for (Vector x : a) {
			add(x, b);
		}
		b.scale(1f / a.size());
	}

	public static double cosine(Vector a, Vector b) {
		double[] norms = new double[2];
		return ArrayMath.cosine(dotProduct(a, b, norms), norms);
	}

	public static double cumulate(DenseVector a, DenseVector b) {
		double sum = ArrayMath.cumulate(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double cumulateAfterNormalize(DenseVector a, DenseVector b) {
		double sum = ArrayMath.cumulateAfterNormalize(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double divide(DenseMatrix a, DenseMatrix b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += divide(a.row(i), b.row(i), c.row(i));
		}
		return sum;
	}

	public static double divide(DenseMatrix a, DenseVector b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += divide(a.row(i), b, c.row(i));
		}
		return sum;
	}

	public static double divide(DenseVector a, DenseVector b, DenseVector c) {
		double sum = ArrayMath.divide(a.values(), b.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	public static double dotProduct(Vector a, Vector b) {
		return dotProduct(a, b, new double[0]);
	}

	public static double dotProduct(Vector a, Vector b, double[] norms) {
		double ret = 0;
		if (isSparse(a) && isSparse(b)) {
			ret = ArrayMath.dotProduct(a.indexes(), a.values(), b.indexes(), b.values(), norms);
		} else if (!isSparse(a) && !isSparse(b)) {
			ret = ArrayMath.dotProduct(a.values(), b.values(), norms);
		} else {
			SparseVector s = null;
			DenseVector d = null;

			if (isSparse(a)) {
				d = (DenseVector) b;
				s = (SparseVector) a;
			} else {
				d = (DenseVector) a;
				s = (SparseVector) b;
			}

			ret = ArrayMath.dotProduct(s.indexes(), s.values(), d.values(), norms);

			if (isSparse(b)) {
				ArrayUtils.swap(norms, 0, 1);
			}

		}
		return ret;
	}

	public static double dotProductColumns(DenseMatrix a, int j1, DenseMatrix b, int j2) {
		return ArrayMath.dotProductColumns(a.values(), j1, b.values(), j2);
	}

	public static double dotProductRows(DenseMatrix a, int i1, DenseMatrix b, int i2) {
		return dotProduct(a.row(i1), b.row(i2));
	}

	public static double entropy(Vector x) {
		return ArrayMath.entropy(x.values());
	}

	public static double euclideanDistance(Vector a, Vector b) {
		double ret = 0;
		int i = 0, j = 0;
		double v1 = 0;
		double v2 = 0;
		int m = 0;
		int n = 0;
		double diff = 0;

		while (i < a.size() && j < b.size()) {
			m = a.indexAt(i);
			n = b.indexAt(j);
			v1 = a.valueAt(i);
			v2 = b.valueAt(j);
			diff = 0;

			if (m == n) {
				diff = v1 - v2;
				i++;
				j++;
			} else if (m > n) {
				diff = -v2;
				j++;
			} else if (m < n) {
				diff = v1;
				i++;
			}
			ret += diff * diff;
		}
		ret = Math.sqrt(ret);
		return ret;
	}

	public static double geometricMean(Vector x) {
		return ArrayMath.geometricMean(x.values());
	}

	public static double identity(DenseMatrix a, int init) {
		return ArrayMath.identity(a.values(), init);
	}

	public static boolean isSparse(Matrix x) {
		return x instanceof SparseMatrix;
	}

	public static boolean isSparse(Vector x) {
		return x instanceof SparseVector;
	}

	public static double jsDivergence(Vector a, Vector b) {
		return lambdaDivergence(a, b, 0.5);
	}

	public static double klDivergence(Vector a, Vector b, boolean symmetric) {
		double ret = 0;
		int i = 0, j = 0;

		int ai = 0;
		int bi = 0;
		double av = 0;
		double bv = 0;

		while (i < a.size() && j < b.size()) {
			ai = a.indexAt(i);
			bi = b.indexAt(j);

			if (ai == bi) {
				av = a.valueAt(i);
				bv = b.valueAt(j);
				if (av == bv || av <= 0 || bv <= 0) {

				} else {
					double div = 0;
					if (symmetric) {
						div = av * Math.log(av / bv) + bv * Math.log(bv / av);
					} else {
						div = av * Math.log(av / bv);
					}
					ret += div;
				}
				i++;
				j++;
			} else if (ai > bi) {
				j++;
			} else if (ai < bi) {
				i++;
			}
		}

		return ret;
	}

	public static double lambdaDivergence(Vector a, Vector b, double lambda) {
		double ret = 0;
		int i = 0, j = 0;
		int ai = 0;
		int bi = 0;
		double av = 0;
		double bv = 0;

		double term1 = 0;
		double term2 = 0;
		double avg = 0;

		while (i < a.size() && j < b.size()) {
			ai = a.indexAt(i);
			bi = b.indexAt(j);

			if (ai == bi) {
				av = a.valueAt(i);
				bv = b.valueAt(j);
				if (av != bv && av > 0 && bv > 0) {
					avg = lambda * av + (1 - lambda) * bv;
					term1 = av * Math.log(av / avg);
					term2 = bv * Math.log(bv / avg);
					ret += (lambda * term1 + (1 - lambda) * term2);
				}
				i++;
				j++;
			} else if (ai > bi) {
				j++;
			} else if (ai < bi) {
				i++;
			}
		}

		return ret;
	}

	public static double leakReLU(DenseMatrix a, double k, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += leakReLU(a.row(i), k, b.row(i));
		}
		return sum;
	}

	public static double leakReLU(DenseVector a, double k, DenseVector b) {
		double sum = ArrayMath.leakReLU(a.values(), k, b.values());
		b.setSum(sum);
		return sum;
	}

	public static double leakReLUGradient(DenseMatrix a, double k, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += leakReLUGradient(a.row(i), k, b.row(i));
		}
		return sum;
	}

	public static double leakReLUGradient(DenseVector a, double k, DenseVector b) {
		double sum = ArrayMath.leakReLUGradient(a.values(), k, b.values());
		return sum;
	}

	public static double log(DenseVector a, DenseVector b) {
		double sum = ArrayMath.log(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static void main(String[] args) {
		System.out.println("process begins.");

		int num_iters = 100000;
		int size = 100000;
		{
			Timer timer = Timer.newTimer();

			for (int i = 0; i < num_iters; i++) {
				DenseVector a1 = new DenseVector(size);
				DenseVector a2 = new DenseVector(size);
				DenseVector a3 = new DenseVector(size);

				a1.setAll(1);
				a2.setAll(1);
				add(a1, a2, a3);
			}

			System.out.println(timer.stop());
		}

		{
			Timer timer = Timer.newTimer();

			DenseVector a1 = new DenseVector(size);
			DenseVector a2 = new DenseVector(size);
			DenseVector a3 = new DenseVector(size);
			for (int i = 0; i < num_iters; i++) {
				a1.setAll(1);
				a2.setAll(1);

				add(a1, a2, a3);
			}

			System.out.println(timer.stop());
		}

		// int[] indexes = ArrayUtils.range(10);
		// double[] values = ArrayMath.random(0f, 1f, 10);
		//
		// SparseVector sv = new SparseVector(indexes, values);
		// sv.sortByValue();
		// System.out.println(sv.toString());
		// sv.sortByIndex();
		// ;
		// System.out.println(sv.toString());
		//
		// Vector x1 = new SparseVector(indexes, values);
		System.out.println("process ends.");

	}

	public static double mask(DenseMatrix x, double p) {
		double sum = 0;
		for (int i = 0; i < x.rowSize(); i++) {
			sum += mask(x.row(i), p);
		}
		return sum;
	}

	public static double mask(DenseVector x, double p) {
		double sum = ArrayMath.mask(x.values(), p);
		x.setSum(sum);
		return sum;
	}

	public static double mean(DenseMatrix a, DenseVector b) {
		double sum = ArrayMath.mean(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double mean(Vector x) {
		return ArrayMath.mean(x.values());
	}

	public static double meanColumns(DenseMatrix a, DenseVector b) {
		double sum = ArrayMath.meanColumns(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double multiply(DenseMatrix a, DenseMatrix b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += multiply(a.row(i), b.row(i), c.row(i));
		}
		return sum;
	}

	public static double multiply(DenseMatrix a, DenseVector b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += multiply(a.row(i), b, c.row(i));
		}
		return sum;
	}

	public static double multiply(DenseMatrix a, double ac, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += multiply(a.row(i), ac, b.row(i));
		}
		return sum;
	}

	public static double multiply(DenseVector a, DenseVector b, DenseVector c) {
		double sum = ArrayMath.multiply(a.values(), b.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	public static double multiply(DenseVector a, DenseVector b, double c, DenseVector d) {
		double sum = ArrayMath.multiply(a.values(), b.values(), c, d.values());
		d.setSum(sum);
		return sum;
	}

	public static double multiply(DenseVector a, double ac, DenseVector c) {
		double sum = ArrayMath.multiply(a.values(), ac, c.values());
		c.setSum(sum);
		return sum;
	}

	public static SparseVector multiply(SparseVector a, SparseVector b) {
		Counter<Integer> c = Generics.newCounter(Math.min(a.size(), b.size()));

		// SparseVector small = a;
		// SparseVector large = b;
		//
		// if (a.size() > b.size()) {
		// small = b;
		// large = a;
		// }

		int ai = 0;
		int bi = 0;
		double av = 0;
		double bv = 0;
		int i = 0, j = 0;

		while (i < a.size() && j < b.size()) {
			ai = a.indexAt(i);
			bi = b.indexAt(j);
			av = a.valueAt(i);
			bv = b.valueAt(j);

			if (ai == bi) {
				c.setCount(ai, (av * bv));

				i++;
				j++;
			} else if (ai > bi) {
				j++;
			} else if (ai < bi) {
				i++;
			}
		}
		return new SparseVector(c);
	}

	public static double multiplyAfterAdd(DenseMatrix a, double ac, DenseMatrix b, double bc, DenseMatrix c, DenseMatrix d) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += multiplyAfterAdd(a.row(i), ac, b.row(i), bc, c.row(i), c.row(i));
		}
		return sum;
	}

	/**
	 * 
	 * (a * ac + b * bc) * c
	 * 
	 * @param a
	 * @param b
	 * @param c
	 * @param d
	 * @return
	 */
	public static double multiplyAfterAdd(DenseVector a, DenseVector b, DenseVector c, DenseVector d) {
		double sum = ArrayMath.multiplyAfterAdd(a.values(), 1, b.values(), 1, c.values(), d.values());
		return sum;
	}

	public static double multiplyAfterAdd(DenseVector a, DenseVector ac, DenseVector b, DenseVector bc, DenseVector c, DenseVector d) {
		double sum = ArrayMath.multiplyAfterAdd(a.values(), ac.values(), b.values(), bc.values(), c.values(), d.values());
		return sum;
	}

	public static double multiplyAfterAdd(DenseVector a, double ac, DenseVector b, double bc, DenseVector c, DenseVector d) {
		double sum = ArrayMath.multiplyAfterAdd(a.values(), ac, b.values(), bc, c.values(), d.values());
		return sum;
	}

	public static void normalizeByMinMax(List<SparseVector> xs, int indexSize) {
		DenseVector index_max = new DenseVector(indexSize);
		DenseVector index_min = new DenseVector(indexSize);

		index_max.setAll(-Double.MAX_VALUE);
		index_min.setAll(Double.MAX_VALUE);

		index_max.setSum(0);
		index_min.setSum(0);

		for (int i = 0; i < xs.size(); i++) {
			SparseVector x = xs.get(i);

			for (int j = 0; j < x.size(); j++) {
				int index = x.indexAt(j);
				double value = x.valueAt(j);

				if (value > index_max.value(index)) {
					index_max.set(index, value);
				}

				if (value < index_min.value(index)) {
					index_min.set(index, value);
				}
			}
		}

		for (int i = 0; i < xs.size(); i++) {
			SparseVector x = xs.get(i);

			for (int j = 0; j < x.size(); j++) {
				int index = x.indexAt(j);
				double value = x.valueAt(j);

				double max = index_max.value(index);
				double min = index_min.value(index);

				if (max == -Double.MAX_VALUE || min == Double.MAX_VALUE) {
					continue;
				}

				if (min == max) {
					continue;
				}

				double newValue = (value - min) / (max - min);
				x.setAt(j, newValue);
			}
			x.summation();
		}
	}

	public static void normalizeBySigmoid(SparseVector x) {
		double sum = ArrayMath.normalizeBySigmoid(x.values(), x.values());
		x.setSum(sum);
	}

	public static double normL1(Vector a) {
		return ArrayMath.normL1(a.values());
	}

	public static double normL2(DenseMatrix a) {
		return ArrayMath.normL2(a.values());
	}

	public static double normL2(Vector a) {
		return ArrayMath.normL2(a.values());
	}

	public static DenseMatrix outerProduct(DenseVector a, DenseVector b) {
		DenseMatrix c = new DenseMatrix(a.size(), b.size());
		outerProduct(a, b, c, false);
		return c;
	}

	/**
	 * @param a
	 *            M x 1
	 * @param b
	 *            1 x N
	 * @param c
	 *            M x N
	 * @return
	 */

	public static double outerProduct(DenseVector a, DenseVector b, DenseMatrix c, boolean add) {
		return ArrayMath.outerProduct(a.values(), b.values(), c.values(), add);
	}

	public static double outerProduct(SparseVector a, DenseVector b, DenseMatrix c) {
		int i = 0;
		double v = 0;
		double out = 0;
		double sum = 0;

		for (int m = 0; m < a.size(); m++) {
			i = a.indexAt(m);
			v = a.valueAt(m);

			for (int j = 0; j < b.size(); j++) {
				out = v * b.value(j);

				if (i == j) {
					c.set(i, j, out);
				} else {
					c.set(i, j, out);
					c.set(j, i, out);
				}
			}
		}

		for (int k = 0; k < c.rowSize(); k++) {
			sum += c.row(k).summation();
		}
		return sum;
	}

	public static double pow(DenseMatrix a, double b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += pow(a.row(i), b, c.row(i));
		}
		return sum;
	}

	public static double pow(DenseVector a, double b, DenseVector c) {
		double sum = ArrayMath.pow(a.values(), b, c.values());
		c.setSum(sum);
		return sum;
	}

	/**
	 * @param a
	 *            M x K
	 * @param b
	 *            K x N
	 * @return
	 */
	public static DenseMatrix product(DenseMatrix a, DenseMatrix b) {
		DenseMatrix c = new DenseMatrix(a.rowSize(), b.colSize());
		product(a, b, c, false);
		return c;
	}

	/**
	 * @param a
	 *            M x K
	 * @param b
	 *            K x N
	 * @param c
	 *            M x N
	 * @return
	 */
	public static double product(DenseMatrix a, DenseMatrix b, DenseMatrix c, boolean add) {
		ArrayMath.product(a.values(), b.values(), c.values(), add);
		return c.sumRows().sum();
	}

	public static DenseVector product(DenseMatrix a, DenseVector b) {
		DenseVector c = new DenseVector(a.rowSize());
		product(a, b, c, false);
		return c;
	}

	/**
	 * @param a
	 *            M x K
	 * @param b
	 *            K x 1
	 * @param c
	 *            M x 1
	 * @return
	 */
	public static double product(DenseMatrix a, DenseVector b, DenseVector c, boolean add) {
		double sum = ArrayMath.product(a.values(), b.values(), c.values(), add);
		c.setSum(sum);
		return sum;
	}

	public static double product(DenseVector a) {
		return ArrayMath.product(a.values());
	}

	/**
	 * @param a
	 *            1 x N
	 * @param b
	 *            N x K
	 * @param c
	 *            1 x K
	 */
	public static double product(DenseVector a, DenseMatrix b, DenseVector c, boolean add) {
		return product(a.toDenseMatrix(), b, c.toDenseMatrix(), add);
	}

	/**
	 * @param a
	 *            M x K
	 * @param b
	 *            K x N
	 * @param c
	 *            M x N
	 * @return
	 */
	public static double product(SparseMatrix a, DenseMatrix b, DenseMatrix c) {
		int a_rows = a.rowSize();
		// int a_cols = a[0].length;
		int b_rows = b.rowSize();
		int b_cols = b.colSize();
		double dot = 0;
		DenseVector rowSums = new DenseVector(c.rowSize());
		DenseVector bj = new DenseVector(b_rows); // column j of B

		for (int j = 0; j < b_cols; j++) {
			VectorUtils.copyColumn(b, j, bj);
			for (int i = 0; i < a_rows; i++) {
				dot = dotProduct(a.rowAt(i), bj);
				c.set(i, j, dot);
				rowSums.add(i, dot);
			}
		}

		for (int i = 0; i < c.rowSize(); i++) {
			c.row(i).setSum(rowSums.value(i));
		}

		return rowSums.sum();
	}

	public static SparseMatrix product(SparseMatrix a, SparseMatrix b) {
		SparseMatrix sm1 = a;
		SparseMatrix sm2 = b.transpose();

		ListMap<Integer, Integer> lm = Generics.newListMap(b.colSize());

		for (int n = 0; n < sm2.rowSize(); n++) {
			int j = sm2.indexAt(n);
			SparseVector sv2 = sm2.rowAt(n);
			for (int k : sv2.indexes()) {
				lm.put(k, n);
			}
		}

		CounterMap<Integer, Integer> c = Generics.newCounterMap(a.rowSize());

		Set<Integer> ns = Generics.newHashSet();

		for (int m = 0; m < sm1.rowSize(); m++) {
			int i = sm1.indexAt(m);
			SparseVector sv1 = sm1.rowAt(m);
			ns.clear();

			for (int k : sv1.indexes()) {
				List<Integer> tmp = lm.get(k, false);

				if (tmp != null) {
					ns.addAll(tmp);
				}
			}

			for (int n : ns) {
				int j = sm2.indexAt(n);
				SparseVector sv2 = sm2.rowAt(n);
				double dot = dotProduct(sv1, sv2);
				c.setCount(i, j, dot);
			}
		}

		return new SparseMatrix(c);
	}

	/**
	 * @param a
	 *            M x K
	 * @param b
	 *            K x N
	 * @param c
	 *            M x N
	 * @return
	 */
	public static double product(SparseVector a, DenseMatrix b, DenseVector c) {
		double sum = ArrayMath.product(a.indexes(), a.values(), b.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	/**
	 * @param a
	 *            M x N
	 * @param b
	 *            N x K
	 * @param c
	 *            M x K
	 * @return
	 */
	public static double productByThreads(DenseMatrix a, DenseMatrix b, DenseMatrix c, int num_threads) {
		ArrayMath.productByThreads(a.values(), b.values(), c.values(), num_threads);
		return c.sumRows().sum();
	}

	/**
	 * @param a
	 *            M x K
	 * @param b
	 *            K x 1
	 * @param c
	 *            M x 1
	 * @param num_threads
	 * @return
	 */
	public static double productByThreads(DenseMatrix a, DenseVector b, DenseVector c, int num_threads) {
		double sum = ArrayMath.productByThreads(a.values(), b.values(), c.values(), num_threads);
		c.setSum(sum);
		return sum;
	}

	/**
	 * @param a
	 *            1 x K
	 * @param b
	 *            K x M
	 * @param c
	 *            1 x M
	 * @param num_threads
	 * @return
	 */
	public static double productByThreads(DenseVector a, DenseMatrix b, DenseVector c, int num_threads) {
		double sum = ArrayMath.productByThreads(a.values(), b.values(), c.values(), num_threads);
		c.setSum(sum);
		return sum;
	}

	public static SparseMatrix productByThreads(SparseMatrix a, SparseMatrix b, int thread_size) throws Exception {
		SparseMatrix sm1 = a;
		SparseMatrix sm2 = b.transpose();

		CounterMap<Integer, Integer> c = Generics.newCounterMap(a.rowSize());

		AtomicInteger loc = new AtomicInteger(0);
		List<Future<Double>> fs = Generics.newArrayList(thread_size);
		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new EqualColumnProductWorker2(sm1, sm2, c, loc, false)));
		}

		double sum = 0;

		for (int k = 0; k < fs.size(); k++) {
			sum += fs.get(k).get().doubleValue();
		}

		tpe.shutdown();

		return new SparseMatrix(c);
	}

	public static double productByThreads(SparseVector a, DenseMatrix b, DenseVector c, int num_threads) {
		double sum = ArrayMath.productByThreads(a.indexes(), a.values(), b.values(), c.values(), num_threads);
		c.setSum(sum);
		return sum;
	}

	/**
	 * @param a
	 *            K x M
	 * @param b
	 *            K x N
	 * @param c
	 *            M x N
	 * @return
	 */
	public static double productColumns(DenseMatrix a, DenseMatrix b, DenseMatrix c, boolean add) {
		ArrayMath.productColumns(a.values(), b.values(), c.values(), add);
		return c.sumRows().sum();
	}

	/**
	 * @param a
	 *            K x M
	 * @param b
	 *            K x 1
	 * @param c
	 *            M x 1
	 */
	public static double productColumns(DenseMatrix a, DenseVector b, DenseVector c) {
		double sum = ArrayMath.productColumns(a.values(), b.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	/**
	 * @param a
	 *            K x M
	 * @param b
	 *            K x N
	 * @param c
	 *            M x N
	 * @return
	 */
	public static double productColumns(SparseMatrix a, DenseMatrix b, DenseMatrix c, boolean add) {
		ArrayMath.productColumns(a.values(), b.values(), c.values(), add);
		return c.sumRows().sum();
	}

	/**
	 * @param a
	 *            M x K
	 * @param b
	 *            N x K
	 * @param c
	 *            M x N
	 * @return
	 */
	public static double productRows(DenseMatrix a, DenseMatrix b, DenseMatrix c, boolean add) {
		ArrayMath.productRows(a.values(), b.values(), c.values(), add);
		return c.sumRows().sum();
	}

	public static double random(double min, double max, DenseMatrix x) {
		double sum = 0;
		for (int i = 0; i < x.rowSize(); i++) {
			sum += random(min, max, x.row(i));
		}
		return sum;
	}

	/**
	 * @param min
	 *            inclusive
	 * @param max
	 *            exclusive
	 * @param x
	 * @return
	 */
	public static double random(double min, double max, DenseVector x) {
		double sum = ArrayMath.random(min, max, x.values());
		x.setSum(sum);
		return sum;
	}

	public static DenseVector random(double min, double max, int size) {
		DenseVector x = new DenseVector(size);
		random(min, max, x);
		return x;
	}

	public static DenseMatrix random(double min, double max, int row_size, int col_size) {
		DenseMatrix x = new DenseMatrix(row_size, col_size);
		random(min, max, x);
		return x;
	}

	/**
	 * @param T
	 *            Column-normalized transition probabilities
	 * @param cents
	 * @param max_iter
	 * @param min_dist
	 * @param damping_factor
	 * @return
	 */
	public static void randomWalk(SparseMatrix T, DenseVector cents, DenseVector biases, int max_iter, double min_dist,
			double damping_factor) {

		DenseVector old_cents = cents.copy();
		double old_dist = Double.MAX_VALUE;
		double uniform_cent = (1 - damping_factor) / T.rowSize();

		for (int m = 0; m < max_iter; m++) {
			for (int i = 0; i < T.rowSize(); i++) {
				SparseVector sv = T.rowAt(i);
				double cent_from_others = dotProduct(sv, old_cents);
				cents.set(i, damping_factor * cent_from_others);
			}

			double sum = 0;
			if (biases == null) {
				sum = add(cents, uniform_cent, cents);
			} else {
				sum = addAfterMultiply(biases, damping_factor, cents, 1, cents);
			}

			if (sum != 1) {
				cents.multiply(1d / sum);
			}

			double dist = euclideanDistance(old_cents, cents);

			System.out.printf("%d: %s - %s = %s\n", m + 1, old_dist, dist, old_dist - dist);

			if (dist < min_dist) {
				break;
			}

			if (dist > old_dist) {
				VectorUtils.copy(old_cents, cents);
				break;
			}

			old_dist = dist;
			VectorUtils.copy(cents, old_cents);
		}
	}

	public static void randomWalk(SparseMatrix T, DenseVector cents, int max_iter) {
		randomWalk(T, cents, null, max_iter, 0.0000001, 0.85);
	}

	public static SparseVector rank(Vector a) {
		SparseVector ret = null;
		if (isSparse(a)) {
			ret = (SparseVector) a.copy();
			ret.sortValues();
			for (int i = 0; i < ret.size(); i++) {
				ret.setAt(i, i + 1);
			}
			ret.setSum(0);
		} else {
			SparseVector m = ((DenseVector) a).toSparseVector();
			m.sortValues();

			ret = m.copy();
			for (int i = 0; i < m.size(); i++) {
				ret.setAt(i, i + 1);
			}
			ret.setSum(0);
		}
		ret.sortIndexes();
		return ret;
	}

	public static double reLU(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += reLU(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double reLU(DenseVector a, DenseVector b) {
		double sum = ArrayMath.reLU(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double reLUGradient(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += reLUGradient(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double reLUGradient(DenseVector a, DenseVector b) {
		double sum = ArrayMath.reLUGradient(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static void scale(List<Vector> xs, double upper, double lower) {
		int max_index = Integer.MIN_VALUE;

		for (int i = 0; i < xs.size(); i++) {
			Vector x = xs.get(i);
			for (int j = 0; j < x.size(); j++) {
				int index = x.indexAt(j);
				if (index > max_index) {
					max_index = index;
				}
			}
		}

		double[] feature_max = new double[max_index + 1];
		double[] feature_min = new double[max_index + 1];

		Arrays.fill(feature_max, Double.MIN_VALUE);
		Arrays.fill(feature_min, Double.MAX_VALUE);

		for (int i = 0; i < xs.size(); i++) {
			Vector x = xs.get(i);
			for (int j = 0; j < x.size(); j++) {
				int index = x.indexAt(j);
				double value = x.valueAt(j);
				feature_max[index] = Math.max(value, feature_max[index]);
				feature_min[index] = Math.min(value, feature_min[index]);
			}
		}

		for (int i = 0; i < xs.size(); i++) {
			Vector x = xs.get(i);
			for (int j = 0; j < x.size(); j++) {
				int index = x.indexAt(j);
				double value = x.valueAt(j);

				double max = feature_max[index];
				double min = feature_min[index];

				if (max == min) {

				} else if (value == min) {

				} else if (value == max) {

				} else {
					value = lower + (upper - lower) * (value - min) / (max - min);
				}
				x.setAt(j, index, value);
			}
		}
	}

	public static double sigmoid(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += sigmoid(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double sigmoid(DenseVector a, DenseVector b) {
		double sum = ArrayMath.sigmoid(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static DenseMatrix sigmoidGradient(DenseMatrix a) {
		DenseMatrix b = new DenseMatrix(a.rowSize(), a.colSize());
		sigmoidGradient(a, b);
		return b;
	}

	public static double sigmoidGradient(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += sigmoidGradient(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double sigmoidGradient(DenseVector a, DenseVector b) {
		double sum = ArrayMath.sigmoidGradient(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double softmax(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += softmax(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double softmax(DenseMatrix a, int c, int o) {
		return ArrayMath.softmax(a.values(), c, o);
	}

	public static double softmax(DenseVector a, DenseVector b) {
		double sum = ArrayMath.softmax(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double softmax(Vector a) {
		double sum = ArrayMath.softmax(a.values(), a.values());
		a.setSum(sum);
		return sum;
	}

	public static double sqrt(DenseVector a, DenseVector b) {
		double sum = ArrayMath.sqrt(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double subtract(DenseMatrix a, DenseMatrix b, DenseMatrix c) {
		return addAfterMultiply(a, 1, b, -1, c);
	}

	public static double subtract(DenseMatrix a, DenseVector b, DenseMatrix c) {
		return addAfterMultiply(a, 1, b, -1, c);
	}

	public static double subtract(DenseVector a, DenseVector b, DenseVector c) {
		return addAfterMultiply(a, 1, b, -1, c);
	}

	public static double subtract(DenseVector a, double b, DenseVector c) {
		return addAfterMultiply(a, 1, b, -1, c);
	}

	public static double sum(DenseMatrix a) {
		return ArrayMath.sum(a.values());
	}

	public static double sumAfterLog(DenseVector a) {
		return ArrayMath.sumAfterLog(a.values());
	}

	public static double sumColumn(DenseMatrix a, int j) {
		return ArrayMath.sumColumn(a.values(), j);
	}

	public static DenseVector sumColumns(DenseMatrix a) {
		DenseVector b = new DenseVector(a.colSize());
		sumColumns(a, b);
		return b;
	}

	public static double sumColumns(DenseMatrix a, DenseVector b) {
		double sum = ArrayMath.sumColumns(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double sumSquared(DenseMatrix a) {
		return ArrayMath.sumSquared(a.values());
	}

	public static double sumSquared(DenseVector a) {
		return ArrayMath.sumSquared(a.values());
	}

	public static double tanh(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += tanh(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double tanh(DenseVector a, DenseVector b) {
		double sum = ArrayMath.tanh(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double tanhGradient(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += tanhGradient(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double tanhGradient(DenseVector a, DenseVector b) {
		double sum = ArrayMath.tanhGradient(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static void transpos(DenseMatrix a, DenseMatrix b) {
		for (int i = 0; i < a.rowSize(); i++) {
			for (int j = 0; j < a.colSize(); j++) {
				b.set(j, i, a.value(i, j));
			}
		}

		for (int i = 0; i < b.rowSize(); i++) {
			b.row(i).summation();
		}
	}

	public static DenseMatrix transpose(DenseMatrix a) {
		DenseMatrix b = new DenseMatrix(a.colSize(), a.rowSize());
		transpos(a, b);
		return b;
	}

	public static double unitVector(DenseMatrix a, DenseMatrix b) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += unitVector(a.row(i), b.row(i));
		}
		return sum;
	}

	public static double unitVector(DenseVector a, DenseVector b) {
		double sum = ArrayMath.unitVector(a.values(), b.values());
		b.setSum(sum);
		return sum;
	}

	public static double unitVector(SparseMatrix a) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			SparseVector row = a.rowAt(i);
			sum += unitVector(row);
		}
		return sum;
	}

	public static double unitVector(Vector x) {
		double sum = ArrayMath.unitVector(x.values(), x.values());
		x.setSum(sum);
		return sum;
	}

	public static double variance(DenseMatrix a, DenseVector b, DenseVector c) {
		double sum = ArrayMath.variance(a.values(), b.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	public static double variance(Vector x) {
		return variance(x, mean(x));
	}

	public static double variance(Vector x, double mean) {
		return ArrayMath.variance(x.values(), mean);
	}

	/**
	 * @param a
	 * @param b
	 *            means
	 * @param c
	 *            variance
	 * @return
	 */
	public static double varianceColumns(DenseMatrix a, DenseVector b, DenseVector c) {
		double sum = ArrayMath.varianceColumns(a.values(), b.values(), c.values());
		c.setSum(sum);
		return sum;
	}

	public static double zTransform(DenseVector a, double mean, double var, double eps, DenseVector b) {
		double sum = ArrayMath.zTransform(a.values(), mean, var, eps, b.values());
		b.setSum(sum);
		return sum;
	}

	public double max(DenseMatrix a, double b, DenseMatrix c) {
		double sum = 0;
		for (int i = 0; i < a.rowSize(); i++) {
			sum += max(a.row(i), b, c.row(i));
		}
		return sum;
	}

	public double max(DenseVector a, double b, DenseVector c) {
		double sum = ArrayMath.max(a.values(), b, c.values());
		c.setSum(sum);
		return sum;
	}

}
