package ohs.math;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;

public class ThreadWokers {
	public static class AddAfterMultiplyWorker1 implements Callable<Double> {
		private double[][] a;

		private double ac;

		private double[][] b;

		private double bc;

		private double[][] c;

		private AtomicInteger shared_i;

		public AddAfterMultiplyWorker1(double[][] a, double ac, double[][] b, double bc, double[][] c,
				AtomicInteger shared_i) {
			this.a = a;
			this.ac = ac;
			this.b = b;
			this.bc = bc;
			this.c = c;
			this.shared_i = shared_i;
		}

		@Override
		public Double call() throws Exception {
			double sum = 0;
			int a_rows = a.length;
			int i = 0;
			while ((i = shared_i.getAndIncrement()) < a_rows) {
				sum += ArrayMath.addAfterMultiply(a[i], ac, b[i], bc, c[i]);
			}
			return sum;
		}
	}

	public static class AddAfterMultiplyWorker2 implements Callable<Double> {
		private double[][] a;

		private double[] ac;

		private double[][] b;

		private double[] bc;

		private double[][] c;

		private AtomicInteger shared_i;

		public AddAfterMultiplyWorker2(double[][] a, double[] ac, double[][] b, double[] bc, double[][] c,
				AtomicInteger shared_i) {
			this.a = a;
			this.ac = ac;
			this.b = b;
			this.bc = bc;
			this.c = c;
			this.shared_i = shared_i;
		}

		@Override
		public Double call() throws Exception {
			double sum = 0;
			int a_rows = a.length;
			int i = 0;
			while ((i = shared_i.getAndIncrement()) < a_rows) {
				sum += ArrayMath.addAfterMultiply(a[i], ac, b[i], bc, c[i]);
			}
			return sum;
		}
	}

	public static class AddAfterMultiplyWorker3 implements Callable<Double> {
		private double[][] a;

		private double[][] ac;

		private double[][] b;

		private double[][] bc;

		private double[][] c;

		private AtomicInteger shared_i;

		public AddAfterMultiplyWorker3(double[][] a, double[][] ac, double[][] b, double[][] bc, double[][] c,
				AtomicInteger shared_i) {
			this.a = a;
			this.ac = ac;
			this.b = b;
			this.bc = bc;
			this.c = c;
			this.shared_i = shared_i;
		}

		@Override
		public Double call() throws Exception {
			double sum = 0;
			int a_rows = a.length;
			int i = 0;
			while ((i = shared_i.getAndIncrement()) < a_rows) {
				sum += ArrayMath.addAfterMultiply(a[i], ac[i], b[i], bc[i], c[i]);
			}
			return sum;
		}
	}

	public static class ColumnByColumnProductWorker implements Callable<Double> {
		private double[][] a;

		private double[][] b;

		private double[][] c;

		private AtomicInteger shared_j;

		public ColumnByColumnProductWorker(double[][] a, double[][] b, double[][] c, AtomicInteger shared_j) {
			this.a = a;
			this.b = b;
			this.c = c;

			this.shared_j = shared_j;
		}

		@Override
		public Double call() throws Exception {
			int a_rows = a.length;
			int a_cols = a[0].length;
			int b_cols = b[0].length;

			double[] bj = new double[a_rows];
			int j = 0;
			double sum = 0;

			while ((j = shared_j.getAndIncrement()) < b_cols) {
				ArrayUtils.copyColumn(b, j, bj);
				for (int i = 0; i < a_cols; i++) {
					double dot = 0;
					for (int k = 0; k < a_rows; k++) {
						dot += a[k][i] * bj[k];
					}
					c[i][j] = dot;
					sum += c[i][j];
				}
			}

			return sum;
		}
	}

	public static class OuterProductWorker implements Callable<Double> {
		private double[] a;

		private double[] b;

		private double[][] c;

		private AtomicInteger cnt;

		public OuterProductWorker(double[] a, double[] b, double[][] c, AtomicInteger cnt) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.cnt = cnt;
		}

		@Override
		public Double call() throws Exception {
			int a_size = a.length;
			int b_size = b.length;

			int i = 0;
			double sum = 0;

			while ((i = cnt.getAndIncrement()) < a_size) {
				for (int j = 0; j < b_size; j++) {
					c[i][j] = a[i] * b[j];
					sum += c[i][j];
				}
			}
			return sum;
		}
	}

	public static class ProductWorker3 implements Callable<Double> {

		private int[][] ai;

		private double[][] av;

		private double[][] b;

		double[] bj;

		private double[][] c;

		private int j;

		@Override
		public Double call() throws Exception {
			int a_rows = av.length;
			int a_cols = av[0].length;
			int b_rows = b.length;
			int b_cols = b[0].length;

			if (bj == null) {
				bj = new double[b_rows];
			}

			double sum = 0;

			ArrayUtils.copyColumn(b, j, bj);
			for (int i = 0; i < a_rows; i++) {
				c[i][j] = ArrayMath.dotProduct(ai[i], av[i], bj);
				sum += c[i][j];
			}
			return sum;
		}

		public void setData(int[][] ai, double[][] av, double[][] b, int j, double[][] c) {
			this.ai = ai;
			this.av = av;
			this.b = b;
			this.c = c;
			this.j = j;
		}

		public String toString() {
			return Thread.currentThread().getName();
		}
	}

	public static class RowByColumnProductWorker implements Callable<Double> {
		private double[][] a;

		private double[][] b;

		private double[][] c;

		private AtomicInteger col_cnt;

		public RowByColumnProductWorker(double[][] a, double[][] b, double[][] c, AtomicInteger col_cnt) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.col_cnt = col_cnt;

		}

		@Override
		public Double call() throws Exception {
			int a_rows = a.length;
			int b_rows = b.length;
			int b_cols = b[0].length;

			double sum = 0;
			double[] bj = new double[b_rows];
			int j = 0;

			while ((j = col_cnt.getAndIncrement()) < b_cols) {
				ArrayUtils.copyColumn(b, j, bj);
				for (int i = 0; i < a_rows; i++) {
					c[i][j] = ArrayMath.dotProduct(a[i], bj);
					sum += c[i][j];
				}
			}

			return sum;
		}

		public String toString() {
			StringBuffer sb = new StringBuffer();
			sb.append(String.format("name: %s\n", Thread.currentThread().getName()));
			sb.append(String.format("target column: %d", col_cnt));
			return sb.toString();
		}
	}

	public static class RowByRowProductWorker implements Callable<Double> {
		private double[][] a;

		private double[][] b;

		private double[][] c;

		private AtomicInteger row_cnt;

		public RowByRowProductWorker(double[][] a, double[][] b, double[][] c, AtomicInteger row_cnt) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.row_cnt = row_cnt;
		}

		@Override
		public Double call() throws Exception {
			int a_rows = a.length;
			int b_rows = b.length;

			int j = 0;
			double sum = 0;

			while ((j = row_cnt.getAndIncrement()) < b_rows) {
				for (int i = 0; i < a_rows; i++) {
					c[i][j] = ArrayMath.dotProduct(a[i], b[j]);
					sum += c[i][j];
				}
			}
			return sum;
		}
	}

	public static class RowBySingleColumnProductWorker implements Callable<Double> {
		private double[][] a;

		private double[] b;

		private double[] c;

		private AtomicInteger row_cnt;

		public RowBySingleColumnProductWorker(double[][] a, double[] b, double[] c, AtomicInteger row_cnt) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.row_cnt = row_cnt;
		}

		@Override
		public Double call() throws Exception {
			double sum = 0;
			int a_rows = a.length;
			int i = 0;

			while ((i = row_cnt.getAndIncrement()) < a_rows) {
				c[i] = ArrayMath.dotProduct(a[i], b);
				sum += c[i];
			}
			return sum;
		}
	}

	public static class SelfSRowProductWorker implements Callable<Double> {
		private SparseMatrix a;

		private CounterMap<Integer, Integer> b;

		private ListMap<Integer, Integer> ii;

		private AtomicInteger row_cnt;

		public SelfSRowProductWorker(SparseMatrix a, CounterMap<Integer, Integer> b, ListMap<Integer, Integer> ii,
				AtomicInteger row_cnt) {
			this.a = a;
			this.b = b;
			this.ii = ii;
			this.row_cnt = row_cnt;
		}

		@Override
		public Double call() throws Exception {
			int m = 0;
			double sum = 0;

			while ((m = row_cnt.getAndIncrement()) < a.rowSize()) {
				int i = a.indexAt(m);
				SparseVector r1 = a.rowAt(m);

				List<Integer> ns = getLocations(r1, m);

				Counter<Integer> c = null;

				synchronized (b) {
					c = b.getCounter(i);
				}

				for (int n : ns) {
					int j = a.indexAt(n);
					SparseVector r2 = a.rowAt(n);
					double dot = VectorMath.dotProduct(r1, r2);
					if (dot != 0) {
						c.setCount(j, dot);
					}
				}

				int prog = BatchUtils.progress(m, a.rowSize());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d]\n", prog, m, a.rowSize());
				}
			}
			return sum;
		}

		private List<Integer> getLocations(SparseVector a, int m) {
			List<Integer> ns = null;

			Set<Integer> set = Generics.newHashSet();
			for (int j : a.indexes()) {
				for (int n : ii.get(j)) {
					if (n > m) {
						set.add(n);
					}
				}
			}
			ns = Generics.newArrayList(set);
			Collections.sort(ns);
			return ns;
		}
	}

	public static class SetAllWorker implements Callable<Double> {
		private double[][] a;

		private AtomicInteger shared_i;

		private double value;

		public SetAllWorker(double[][] a, double value, AtomicInteger shared_i) {
			this.a = a;
			this.value = value;
			this.shared_i = shared_i;
		}

		@Override
		public Double call() throws Exception {
			int a_rows = a.length;
			int i = 0;
			double sum = 0;

			while ((i = shared_i.getAndIncrement()) < a_rows) {
				sum += ArrayUtils.setAll(a[i], value);
			}
			return sum;
		}
	}

	public static class SRowByColumnProductWorker implements Callable<Double> {
		private SparseMatrix a;

		private DenseVector b;

		private DenseVector c;

		private AtomicInteger row_cnt;

		private boolean print_log;

		public SRowByColumnProductWorker(SparseMatrix a, DenseVector b, DenseVector c, AtomicInteger row_cnt,
				boolean print_log) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.row_cnt = row_cnt;
			this.print_log = print_log;
		}

		@Override
		public Double call() throws Exception {
			int m = 0;
			double sum = 0;

			while ((m = row_cnt.getAndIncrement()) < a.rowSize()) {
				int i = a.indexAt(m);
				SparseVector row_a = a.rowAt(m);
				double dot = VectorMath.dotProduct(row_a, b);
				sum += dot;
				c.set(i, dot);

				int prog = BatchUtils.progress(m, a.rowSize());
				if (prog > 0 && print_log) {
					System.out.printf("[%d percent, %d/%d]\n", prog, m, a.rowSize());
				}
			}
			return sum;
		}
	}

	public static class SRowBySRowProductWorker implements Callable<Double> {
		private SparseMatrix a;

		private AtomicInteger a_row_cnt;

		private SparseMatrix b;

		private CounterMap<Integer, Integer> c;

		public SRowBySRowProductWorker(SparseMatrix a, SparseMatrix b, CounterMap<Integer, Integer> c,
				AtomicInteger a_row_cnt) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.a_row_cnt = a_row_cnt;
		}

		@Override
		public Double call() throws Exception {
			int m = 0;
			double sum = 0;

			while ((m = a_row_cnt.getAndIncrement()) < a.rowSize()) {
				int i = a.indexAt(m);
				SparseVector row_a = a.rowAt(m);

				Counter<Integer> tmp = null;

				synchronized (c) {
					tmp = c.getCounter(i);
				}

				for (int n = 0; n < b.rowSize(); n++) {
					int j = b.indexAt(n);
					SparseVector row_b = b.rowAt(n);
					double dot = VectorMath.dotProduct(row_a, row_b);
					if (dot != 0) {
						tmp.setCount(j, dot);
					}
				}

				int prog = BatchUtils.progress(m, a.rowSize());

				if (prog > 0) {
					System.out.printf("[%d percent, %d/%d]\n", prog, m, a.rowSize());
				}
			}
			return sum;
		}
	}

	public static boolean PRINT_LOG = false;
}
