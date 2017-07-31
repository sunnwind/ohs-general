package ohs.math;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.math.ThreadWokers.SetAllWorker;
import ohs.utils.ByteSize;
import ohs.utils.Conditions;
import ohs.utils.Generics;

public class ArrayUtils {

	public static int THREAD_SIZE = 5;

	public static int argMaxLength(int[][] a) {
		return argMinMaxLength(a)[1];
	}

	public static int argMinLength(int[][] a) {
		return argMinMaxLength(a)[0];
	}

	public static long getSingleIndex(long i, long max_i, long j) {
		return i * max_i + j;
	}

	public static long[] getTwoIndexes(long k, long max_i) {
		long i = (k / max_i);
		long j = (k % max_i);
		return new long[] { i, j };
	}

	public static int[] argMinMaxLength(int[][] a) {
		int max = -Integer.MAX_VALUE;
		int max_loc = 0;

		int min = Integer.MAX_VALUE;
		int min_loc = 0;

		for (int i = 0; i < a.length; i++) {
			int[] b = a[i];
			if (min > b.length) {
				min = b.length;
				min_loc = i;
			}

			if (max < b.length) {
				max = b.length;
				max_loc = i;
			}
		}
		return new int[] { min_loc, max_loc };
	}

	public static ByteSize byteSize(double[] a) {
		return new ByteSize(Double.BYTES * a.length);
	}

	public static ByteSize byteSize(int[] a) {
		return new ByteSize(Integer.BYTES * a.length);
	}

	public static double[] copy(double[] a) {
		double[] ret = new double[a.length];
		copy(a, ret);
		return ret;
	}

	public static double copy(double[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = a[i];
			sum += b[i];
		}
		return sum;
	}

	public static double[] copy(double[] a, int start, int end) {
		double[] b = new double[end - start];
		copy(a, start, end, b);
		return b;
	}

	public static double copy(double[] a, int start, int end, double[] b) {
		int size = end - start;
		if (size > b.length) {
			throw new IllegalArgumentException();
		}
		double sum = 0;
		for (int i = start, j = 0; i < end; i++, j++) {
			b[j] = a[i];
			sum += b[j];
		}
		return sum;
	}

	public static int copy(double[] a, int[] b) {
		int sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = (int) a[i];
			sum += b[i];
		}
		return sum;
	}

	public static double copy(double[] a, List<Double> b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			b.add(a[i]);
			sum += a[i];
		}
		return sum;
	}

	public static double[][] copy(double[][] a) {
		double[][] b = new double[a.length][];
		for (int i = 0; i < a.length; i++) {
			b[i] = copy(a[i]);
		}
		return b;
	}

	public static double copy(double[][] a, double[][] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += copy(a[i], b[i]);
		}
		return sum;
	}

	public static int copy(double[][] a, int[][] b) {
		int sum = 0;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				b[i][j] = (int) a[i][j];
				sum += b[i][j];
			}
		}
		return sum;
	}

	public static double copy(float[] a, float[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = a[i];
			sum += b[i];
		}
		return sum;
	}

	public static int[] copy(int[] a) {
		int[] b = new int[a.length];
		copy(a, b);
		return b;
	}

	public static double copy(int[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = a[i];
			sum += b[i];
		}
		return sum;
	}

	public static int[] copy(int[] a, int start, int end) {
		int size = end - start;
		int[] ret = new int[size];
		copy(a, start, end, ret);
		return ret;
	}

	public static int copy(int[] a, int start, int end, int[] b) {
		int size = end - start;
		if (size > b.length) {
			throw new IllegalArgumentException();
		}
		int sum = 0;
		for (int i = start, j = 0; i < end; i++, j++) {
			b[j] = a[i];
			sum += b[j];
		}
		return sum;
	}

	public static int copy(int[] a, int[] b) {
		int sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = a[i];
			sum += b[i];
		}
		return sum;
	}

	public static int copy(int[] a, List<Integer> b) {
		int sum = 0;
		for (int i = 0; i < a.length; i++) {
			b.add(a[i]);
			sum += a[i];
		}
		return sum;
	}

	public static int copy(int[] a, Set<Integer> b) {
		int sum = 0;
		for (int value : a) {
			b.add(value);
			sum += value;
		}
		return sum;
	}

	public static double copy(int[][] a, double[][] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				b[i][j] = a[i][j];
				sum += b[i][j];
			}
		}
		return sum;
	}

	public static double copyColumn(double[] a, double[][] b, int bj) {
		int[] dims = dimensions(b);
		double sum = 0;
		for (int i = 0; i < dims[0]; i++) {
			b[i][bj] = a[i];
			sum += b[i][bj];
		}
		return sum;
	}

	public static double[] copyColumn(double[][] a, int j) {
		double[] ret = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			ret[i] = a[i][j];
		}
		return ret;
	}

	public static double copyColumn(double[][] a, int j, double[] b) {
		if (a.length != b.length) {
			throw new IllegalArgumentException();
		}
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			b[i] = a[i][j];
			sum += b[i];
		}
		return sum;
	}

	public static double copyColumn(double[][] a, int aj, double[][] b, int bj) {
		if (!ArrayChecker.isEqualSize(a, b)) {
			throw new IllegalArgumentException();
		}
		int rowDim = a.length;
		int colDim = a[0].length;
		double sum = 0;
		for (int i = 0; i < rowDim; i++) {
			b[i][bj] = a[i][aj];
			sum += b[i][bj];
		}
		return sum;
	}

	public static double copyDoubles(Collection<Double> a, double[] b) {
		int loc = 0;
		Iterator<Double> iter = a.iterator();
		double sum = 0;
		while (iter.hasNext()) {
			b[loc] = iter.next();
			sum += b[loc];
			loc++;
		}
		return sum;
	}

	public static double copyFloats(Collection<Float> a, float[] b) {
		int loc = 0;
		Iterator<Float> iter = a.iterator();
		double sum = 0;
		while (iter.hasNext()) {
			b[loc] = iter.next();
			sum += b[loc];
			loc++;
		}
		return sum;
	}

	public static int[] copyIntegers(Collection<Integer> a) {
		int[] ret = new int[a.size()];
		copyIntegers(a, ret);
		return ret;
	}

	public static int copyIntegers(Collection<Integer> a, int[] b) {
		int sum = 0;
		int loc = 0;
		Iterator<Integer> iter = a.iterator();
		while (iter.hasNext()) {
			b[loc] = iter.next();
			sum += b[loc];
			loc++;
		}
		return sum;
	}

	public static long[] copyLongs(Collection<Long> a) {
		long[] b = new long[a.size()];
		copyLongs(a, b);
		return b;
	}

	public static long copyLongs(Collection<Long> a, long[] b) {
		long sum = 0;
		int loc = 0;
		Iterator<Long> iter = a.iterator();
		while (iter.hasNext()) {
			b[loc] = iter.next();
			sum += b[loc];
			loc++;
		}
		return sum;
	}

	public static double copyRow(double[] a, double[][] b, int bi) {
		return copy(a, b[bi]);
	}

	public static double copyRow(double[][] a, int ai, double[] b) {
		return copy(a[ai], b);
	}

	public static double copyRow(double[][] a, int ai, double[][] b, int bi) {
		return copy(a[ai], b[bi]);
	}

	public static double copyRows(double[][] a, int start, int end, double[] b) {
		int size = 0;
		for (int i = start; i < end; i++) {
			size += a[i].length;
		}

		if (size != b.length) {
			throw new IllegalArgumentException();
		}

		double sum = 0;
		int loc = 0;

		for (int i = start; i < end; i++) {
			double[] c = a[i];
			for (int j = 0; j < c.length; j++) {
				b[loc] = c[j];
				sum += b[loc];
				loc++;
			}
		}
		return sum;
	}

	public static short[] copyShorts(Collection<Short> a) {
		short[] b = new short[a.size()];
		copyShorts(a, b);
		return b;
	}

	public static int copyShorts(Collection<Short> a, short[] b) {
		int sum = 0;
		int loc = 0;
		Iterator<Short> iter = a.iterator();
		while (iter.hasNext()) {
			b[loc] = iter.next();
			sum += b[loc];
			loc++;
		}
		return sum;
	}

	public static int[] dimensions(double[][] a) {
		int[] ret = new int[2];
		ret[0] = a.length;
		ret[1] = a[0].length;
		return ret;
	}

	public static int[] enlarge(int[] a, int add_size) {
		int[] ret = new int[a.length + add_size];
		for (int i = 0; i < a.length; i++) {
			ret[i] = a[i];
		}
		return ret;
	}

	public static double get(double[][] a, int[] index) {
		return a[index[0]][index[1]];
	}

	public static NumberFormat getDoubleNumberFormat(int num_fractions) {
		NumberFormat ret = NumberFormat.getInstance();
		ret.setMinimumFractionDigits(num_fractions);
		ret.setGroupingUsed(false);
		return ret;
	}

	public static NumberFormat getIntegerNumberFormat() {
		NumberFormat ret = NumberFormat.getInstance();
		ret.setMinimumFractionDigits(0);
		ret.setGroupingUsed(false);
		return ret;
	}

	public static int[] indexesOfZero(double[] x) {
		List<Integer> ids = Generics.newArrayList();
		for (int i = 0; i < x.length; i++) {
			if (x[i] == 0) {
				ids.add(i);
			}
		}
		int[] ret = new int[ids.size()];
		copyIntegers(ids, ret);
		return ret;
	}

	public static int indexOf(double[] a, int start, int end, double b, boolean complement) {
		for (int i = start; i < end; i++) {
			if (complement ? a[i] != b : a[i] == b) {
				return i;
			}
		}
		return -1;
	}

	public static int indexOf(int[] a, int start, int end, int b, boolean complement) {
		for (int i = start; i < end; i++) {
			if (complement ? a[i] != b : a[i] == b) {
				return i;
			}
		}
		return -1;
	}

	public static int lastIndexOf(double[] a, int start, int end, double b, boolean complement) {
		for (int i = a.length - 1; i >= 0; i--)
			if (complement ? a[i] != b : a[i] == b) {
				return i;
			}
		return -1;
	}

	public static int lastIndexOf(int[] a, int start, int end, int b, boolean complement) {
		for (int i = a.length - 1; i >= 0; i--)
			if (complement ? a[i] != b : a[i] == b) {
				return i;
			}
		return -1;
	}

	public static int length(double[][] a) {
		int ret = 0;
		for (double[] b : a) {
			ret += b.length;
		}
		return ret;
	}

	public static long length(int[][] a) {
		long ret = 0;
		for (int[] b : a) {
			ret += b.length;
		}
		return ret;
	}

	public static void main(String[] args) {
		System.out.println("process begins.");
		{
			int[] a = range(20);

			System.out.println(toString(a));

			shuffle(a, 0, 10);

			System.out.println(toString(a));

			quickSort(a, 5, 10);

			System.out.println(toString(a));

			System.exit(0);

		}

		{
			int[] dims = { 3, 2, 3 };
			int[] indexes = { 0, 1, 2 };

			int singleIndex = singleIndex(dims, indexes);

			int[] indexes2 = multipleIndexes(singleIndex, dims);

			System.out.println(toString(indexes));
			System.out.println(singleIndex);
			System.out.println(toString(indexes2));

			System.exit(0);
		}

		{
			double[][] x = { { Double.NaN, Double.POSITIVE_INFINITY }, { Double.NEGATIVE_INFINITY, 10.25 } };

			System.out.println(toString(x));
		}

		{
			int[] dims = { 2, 2, 3, 3 };
			int[] indexes = { 1, 1, 2, 1 };
			int singleIndex = singleIndex(dims, indexes);

			int[] indexes2 = multipleIndexes(singleIndex, dims);

			System.out.println(toString(indexes));
			System.out.println(toString(indexes2));
		}

		{
			int[] dims = { 2, 2, 3, 3 };
			int[] indexes = { 0, 1, 2, 0 };
			int singleIndex = singleIndex(dims, indexes);

			int[] indexes2 = multipleIndexes(singleIndex, dims);

			System.out.println(toString(indexes));
			System.out.println(toString(indexes2));
		}

		{

			double[] a = ArrayMath.random(0f, 1f, 10000000);

		}

		System.out.println("process ends.");

	}

	public static int maxColumnSize(int[][] a) {
		int ret = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i].length > ret) {
				ret = a[i].length;
			}
		}
		return ret;
	}

	public static int maxLength(int[][] a) {
		return a[argMaxLength(a)].length;
	}

	public static int minLength(int[][] a) {
		return a[argMinLength(a)].length;
	}

	public static int[] multipleIndexes(int singleIndex, int[] dims) {
		int[] ret = new int[dims.length];
		multipleIndexes(singleIndex, dims, ret);
		return ret;
	}

	public static void multipleIndexes(int singleIndex, int[] dims, int[] indexes) {
		if (!ArrayChecker.isEqualSize(dims, indexes)) {
			throw new IllegalArgumentException();
		}

		int ret = 0;
		int base = 1;
		for (int i = dims.length - 1; i >= 0; i--) {
			if (i != dims.length - 1) {
				base *= dims[i + 1];
			}
		}

		int quotient = 0;
		int remainer = 0;

		for (int i = 0; i < dims.length - 1; i++) {
			if (singleIndex >= base) {
				quotient = singleIndex / base;
				remainer = (singleIndex - quotient * base);
				singleIndex = remainer;
			}

			if (i == dims.length - 2) {
				indexes[i] = quotient;
				indexes[i + 1] = remainer;
			} else {
				indexes[i] = quotient;
			}

			base /= dims[i + 1];
		}

	}

	public static int[] nonzeroIndexes(double[] x) {
		List<Integer> ids = new ArrayList<Integer>();
		for (int i = 0; i < x.length; i++) {
			if (x[i] != 0) {
				ids.add(i);
			}
		}

		int[] ret = new int[ids.size()];
		copyIntegers(ids, ret);
		return ret;
	}

	public static int[] nonzeroIndexes(int[] x) {
		List<Integer> set = new ArrayList<Integer>();
		for (int i = 0; i < x.length; i++) {
			if (x[i] != 0) {
				set.add(i);
			}
		}
		int[] ret = new int[set.size()];
		copyIntegers(set, ret);
		return ret;
	}

	public static void print(int[] x, boolean sparse, boolean vertical) {
		System.out.println(toString(x, sparse, vertical, " ") + "\n");
	}

	public static void print(int[][] x) {
		System.out.println(toString("None", x) + "\n");
	}

	public static void print(String label, double[] x) {
		System.out.println(toString(label, x) + "\n");
	}

	public static void print(String label, double[][] x) {
		System.out.println(toString(label, x) + "\n");
	}

	public static void print(String label, double[][] x, int rows, int cols, boolean sparse, NumberFormat nf) {
		System.out.println(toString(label, x, rows, cols, sparse, nf) + "\n");
	}

	public static void quickSort(double[] a) {
		quickSortHere(a, 0, a.length - 1);
	}

	public static void quickSort(double[] a, int start, int end) {
		quickSortHere(a, start, end - 1);
	}

	public static void quickSort(int[] a) {
		quickSort(a, 0, a.length);
	}

	public static void quickSort(int[] idxs, double[] vals, boolean sort_by_index) {
		if (idxs.length != vals.length) {
			throw new IllegalArgumentException();
		}

		quickSortHere(idxs, vals, 0, idxs.length - 1, sort_by_index);
	}

	public static void quickSort(int[] a, int start, int end) {
		quickSortHere(a, start, end - 1);
	}

	private static void quickSortHere(double[] a, int low, int high) {
		if (low >= high)
			return;
		int p = quickSortPartition(a, low, high);
		quickSortHere(a, low, p);
		quickSortHere(a, p + 1, high);
	}

	private static void quickSortHere(int[] idxs, double[] vals, int low, int high, boolean sort_by_index) {
		if (low >= high)
			return;
		int p = quickSortPartition(idxs, vals, low, high, sort_by_index);
		quickSortHere(idxs, vals, low, p, sort_by_index);
		quickSortHere(idxs, vals, p + 1, high, sort_by_index);
	}

	private static void quickSortHere(int[] a, int low, int high) {
		if (low >= high)
			return;
		int p = quickSortPartition(a, low, high);
		quickSortHere(a, low, p);
		quickSortHere(a, p + 1, high);
	}

	private static int quickSortPartition(double[] a, int low, int high) {
		int i = low - 1;
		int j = high + 1;

		// descending order
		int rand_idx = (int) (Math.random() * (high - low)) + low;
		double pivot = a[rand_idx];

		while (i < j) {
			i++;
			while (a[i] > pivot) {
				i++;
			}

			j--;
			while (a[j] < pivot) {
				j--;
			}

			if (i < j) {
				swap(a, i, j);
			}
		}
		return j;
	}

	private static int quickSortPartition(int[] idxs, double[] vals, int low, int high, boolean sort_by_index) {
		int i = low - 1;
		int j = high + 1;

		if (sort_by_index) {
			// ascending order
			int rand_idx = (int) (Math.random() * (high - low)) + low;
			int pivot = idxs[rand_idx];

			while (i < j) {
				i++;
				while (idxs[i] < pivot) {
					i++;
				}

				j--;
				while (idxs[j] > pivot) {
					j--;
				}

				if (i < j) {
					swap(idxs, i, j);
					swap(vals, i, j);
				}
			}
		} else {
			// descending order
			int rand_idx = (int) (Math.random() * (high - low)) + low;
			double pivot = vals[rand_idx];

			while (i < j) {
				i++;
				while (vals[i] > pivot) {
					i++;
				}

				j--;
				while (vals[j] < pivot) {
					j--;
				}

				if (i < j) {
					swap(idxs, i, j);
					swap(vals, i, j);
				}
			}
		}
		return j;
	}

	private static int quickSortPartition(int[] a, int low, int high) {
		int i = low - 1;
		int j = high + 1;

		// descending order
		int rand_idx = (int) (Math.random() * (high - low)) + low;
		int pivot = a[rand_idx];

		while (i < j) {
			i++;
			while (a[i] > pivot) {
				i++;
			}

			j--;
			while (a[j] < pivot) {
				j--;
			}

			if (i < j) {
				swap(a, i, j);
			}
		}
		return j;
	}

	public static double range(double[] a, double start, double increment) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			a[i] = start + i * increment;
			sum += a[i];
		}
		return sum;
	}

	public static int[] range(int size) {
		return range(size, 0, 1);
	}

	public static int[] range(int... a) {
		int[] b = range(a.length);
		for (int i = 0; i < b.length; i++) {
			b[i] = a[i];
		}
		return b;
	}

	public static double[] range(int size, double start, double increment) {
		double[] a = new double[size];
		range(a, start, increment);
		return a;
	}

	// public static double random(double min, double max, double[] x) {
	// Random random = new Random();
	// double range = max - min;
	// double sum = 0;
	// for (int i = 0; i < x.length; i++) {
	// x[i] = range * random.nextDouble() + min;
	// sum += x[i];
	// }
	// return sum;
	// }
	//
	// public static double random(double min, double max, double[][] x) {
	// double sum = 0;
	// for (int i = 0; i < x.length; i++) {
	// sum += random(min, max, x[i]);
	// }
	// return sum;
	// }
	//
	// public static double[] random(double min, double max, int size) {
	// double[] x = new double[size];
	// random(min, max, x);
	// return x;
	// }
	//
	// public static double[][] random(double min, double max, int rows, int
	// columns) {
	// double[][] x = new double[rows][columns];
	// random(min, max, x);
	// return x;
	// }
	//
	// public static int[] random(int min, int max, int size) {
	// int[] x = new int[size];
	// random(min, max, x);
	// return x;
	// }
	//
	// public static int random(int min, int max, int[] x) {
	// Random random = new Random();
	// double range = max - min + 1;
	// int sum = 0;
	// for (int i = 0; i < x.length; i++) {
	// x[i] = (int) (range * random.nextDouble()) + min;
	// sum += x[i];
	// }
	// return sum;
	// }

	public static int[] range(int size, int start) {
		return range(size, start, 1);
	}

	public static int[] range(int size, int start, int increment) {
		int[] a = new int[size];
		range(a, start, increment);
		return a;
	}

	public static int range(int[] a, int start, int increment) {
		int sum = 0;
		for (int i = 0; i < a.length; i++) {
			a[i] = start + i * increment;
			sum += a[i];
		}
		return sum;
	}

	public static int[] rankedIndexes(double[] a) {
		int[] b = range(a.length);
		quickSort(b, copy(a), false);
		return b;
	}

	public static double reshape(double[] a, double[][] b) {
		if (sizeOfEntries(b) != a.length) {
			throw new IllegalArgumentException();
		}
		double sum = 0;

		int a_size = b.length;
		int b_rows = b.length;
		int b_cols = b[0].length;
		int i = 0;
		int j = 0;

		for (int k = 0; k < a.length; k++) {
			i = k / b_rows;
			j = k % b_rows;
			b[i][j] = a[k];
			sum += a[i];
		}
		return sum;
	}

	public static double[] reshape(double[][] a) {
		double[] b = new double[sizeOfEntries(a)];
		reshape(a, b);
		return b;
	}

	public static double reshape(double[][] a, double[] b) {
		if (sizeOfEntries(a) != b.length) {
			throw new IllegalArgumentException();
		}

		double sum = 0;
		for (int i = 0, k = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				b[k] = a[i][j];
				sum += b[k];
				k++;
			}
		}
		return sum;
	}

	public static void reverse(double[] a) {
		int mid = a.length / 2;
		for (int i = 0; i < mid; i++) {
			swap(a, i, a.length - 1 - i);
		}
	}

	public static void reverse(int[] a) {
		int mid = a.length / 2;
		for (int i = 0; i < mid; i++) {
			swap(a, i, a.length - 1 - i);
		}
	}

	public static double setAll(double[] a, double value) {
		Arrays.fill(a, value);
		// Arrays.parallelSetAll(a, x -> value);
		return a.length * value;
	}

	public static void setAll(double[][] a, double value) {
		for (int i = 0; i < a.length; i++) {
			setAll(a[i], value);
		}
	}

	public static void setAllAtDiagonal(double[][] a, double value) {
		int row_dim = a.length;
		int col_dim = a[0].length;

		int dim = Math.min(row_dim, col_dim);
		for (int i = 0; i < dim; i++) {
			a[i][i] = value;
		}
	}

	public static double setAllByThreads(double[][] a, double value, int num_threads) {
		int a_rows = a.length;

		if (num_threads == 0) {
			num_threads = THREAD_SIZE;
		}

		if (num_threads > a_rows) {
			num_threads = a_rows;
		}

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(num_threads);

		List<Future<Double>> fs = Generics.newArrayList();
		AtomicInteger shared_i = new AtomicInteger(0);
		for (int i = 0; i < num_threads; i++) {
			fs.add(tpe.submit(new SetAllWorker(a, value, shared_i)));
		}
		double sum = 0;
		try {
			for (int k = 0; k < fs.size(); k++) {
				sum += fs.get(k).get().doubleValue();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		} finally {
			fs.clear();
		}

		tpe.shutdown();

		return sum;
	}

	public static void setAllRows(double[][] a, int[] is, double value) {
		for (int i : is) {
			setAll(a[i], value);
		}
	}

	public static void shuffle(double[] a) {
		shuffle(a, 0, a.length);
	}

	public static void shuffle(double[] a, int start, int end) {
		for (int i = start; i < end; i++) {
			int j = ArrayMath.random(start, end);
			swap(a, i, j);
		}
	}

	public static void shuffle(int[] a) {
		shuffle(a, 0, a.length);
	}

	public static void shuffle(int[] a, int start, int end) {
		for (int i = start; i < end; i++) {
			int j = ArrayMath.random(start, end);
			swap(a, i, j);
		}
	}

	public static int singleIndex(int[] dims, int[] indexes) {
		if (!ArrayChecker.isEqualSize(dims, indexes)) {
			throw new IllegalArgumentException();
		}

		int ret = 0;
		int base = 1;

		for (int i = dims.length - 1; i >= 0; i--) {
			if (indexes[i] >= dims[i]) {
				throw new IllegalArgumentException();
			}
			if (i == dims.length - 1) {
				ret += indexes[i];
			} else {
				base *= dims[i + 1];
				ret += base * indexes[i];
			}
		}
		return ret;
	}

	public static int sizeOfEntries(double[][] a) {
		int[] dims = dimensions(a);
		return dims[0] * dims[1];
	}

	public static int sizeOfNonzero(double[] a) {
		int ret = 0;

		for (int i = 0; i < a.length; i++) {
			if (a[i] != 0) {
				ret++;
			}
		}
		return ret;
	}

	public static int sizeOfNonzeros(int[] a) {
		int ret = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] != 0) {
				ret++;
			}
		}
		return ret;
	}

	public static int sizeOfZeros(double[] a) {
		return indexesOfZero(a).length;
	}

	public static void swap(double[] x, int i, int j) {
		double v1 = x[i];
		double v2 = x[j];
		x[i] = v2;
		x[j] = v1;
	}

	public static void swap(int[] x, int i, int j) {
		int v2 = x[i];
		int v1 = x[j];
		x[i] = v1;
		x[j] = v2;
	}

	public static void swapColumns(double[][] x, int i, int j) {
		for (int k = 0; k < x.length; k++) {
			double temp = x[k][i];
			x[k][i] = x[k][j];
			x[k][j] = temp;
		}
	}

	public static void swapRows(double[][] x, int i, int j) {
		for (int k = 0; k < x[0].length; k++) {
			double temp = x[i][k];
			x[i][k] = x[j][k];
			x[j][k] = temp;
		}
	}

	public static String toString(double[] x) {
		return toString("None", x, x.length, false, false, getDoubleNumberFormat(4));
	}

	public static String toString(double[][] x) {
		return toString("None", x, x.length, x[0].length, false, getDoubleNumberFormat(4));
	}

	public static String toString(int[] x) {
		return toString(x, false, false, " ");
	}

	public static String toString(int[] x, boolean sparse, boolean vertical, String delim) {
		StringBuffer sb = new StringBuffer();
		if (vertical) {
			delim = "\n";
		}

		if (sparse) {
			for (int i = 0; i < x.length; i++) {
				if (x[i] != 0) {
					sb.append(String.format("%s%d:%s", delim, i, x[i]));
				}
			}
		} else {
			for (int i = 0; i < x.length; i++) {
				sb.append(String.format("%s%s", delim, x[i]));
			}
		}
		return sb.toString().trim();
	}

	public static String toString(String label, double[] x) {
		return toString(label, x, x.length, false, false, getDoubleNumberFormat(5));
	}

	public static String toString(String label, double[] x, int len, boolean sparse, boolean vertical, NumberFormat nf) {
		StringBuffer sb = new StringBuffer();

		String delim = Conditions.value(vertical, "\t", "\n");

		sb.append(String.format("(%s)", label));

		if (sparse) {
			for (int i = 0; i < x.length && i < len; i++) {
				sb.append(String.format("%s%d:%s", delim, i, nf.format(x[i])));
			}
		} else {
			for (int i = 0; i < x.length && i < len; i++) {
				sb.append(String.format("%s%s", delim, nf.format(x[i])));
			}
		}
		return sb.toString();
	}

	public static String toString(String label, double[][] x) {
		return toString(label, x, x.length, x[0].length, false, getDoubleNumberFormat(5));
	}

	public static String toString(String label, double[][] x, int rows, int cols, boolean sparse, NumberFormat nf) {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("(%s)", label));
		sb.append(String.format("\ndim:\t(%d, %d)\n", x.length, x[0].length));

		if (sparse) {
			for (int i = 0; i < x.length && i < rows; i++) {
				StringBuffer sb2 = new StringBuffer();
				sb2.append(i);

				int num_nonzero_cols = 0;

				for (int j = 0; j < x[i].length && num_nonzero_cols < cols; j++) {
					Double v = new Double(x[i][j]);
					if (v != 0) {
						if (Double.isFinite(v) || Double.isInfinite(v) || Double.isNaN(v)) {
							sb2.append(String.format("\t%d:%s", j, v.toString()));
							num_nonzero_cols++;
						} else {
							sb2.append(String.format("\t%d:%s", j, nf.format(v.doubleValue())));
							num_nonzero_cols++;
						}
					}
				}

				if (num_nonzero_cols > 0) {
					sb.append(sb2.toString());
					sb.append("\n");
				}
			}
		} else {
			sb.append("#");
			for (int i = 0; i < x[0].length && i < cols; i++) {
				sb.append(String.format("\t%d", i));
			}
			sb.append("\n");

			for (int i = 0; i < x.length && i < rows; i++) {
				sb.append(i);
				for (int j = 0; j < x[i].length && j < cols; j++) {
					Double v = new Double(x[i][j]);
					if (!Double.isFinite(v)) {
						sb.append(String.format("\t%s", v.toString()));
					} else {
						sb.append(String.format("\t%s", nf.format(v.doubleValue())));
					}
				}
				sb.append("\n");
			}
		}

		return sb.toString().trim();
	}

	public static String toString(String label, int[][] x) {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("(%s)", label));
		sb.append(String.format("dim:\t(%d, %d)", x.length, x[0].length));
		for (int i = 0; i < x.length; i++) {
			sb.append("\n");
			sb.append(toString(x[i], false, false, "\t"));
		}
		return sb.toString();
	}

	public static double[] values(double[] a, int start, int end) {
		double[] b = new double[end - start];
		for (int i = start, j = 0; i < end; i++, j++) {
			b[j] = a[i];
		}
		return b;
	}

	public static double[] values(double[] a, int[] locs) {
		double[] b = new double[locs.length];
		for (int i = 0; i < locs.length; i++) {
			b[i] = a[locs[i]];
		}
		return b;
	}

	public static int[] values(int[] a, int start, int end) {
		int[] b = new int[end - start];
		for (int i = start, j = 0; i < end; i++, j++) {
			b[j] = a[i];
		}
		return b;
	}

	public static int[] values(int[] a, int[] locs) {
		int[] b = new int[locs.length];
		for (int i = 0; i < locs.length; i++) {
			b[i] = a[locs[i]];
		}
		return b;
	}
}
