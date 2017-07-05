package ohs.math;

import java.sql.BatchUpdateException;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;

public class ArrayWokers {
	public static class AddAfterMultiplyWorker1 implements Callable<Double> {
		private double[][] a;

		private double[][] b;

		private double[][] c;

		private double ac;

		private double bc;

		private AtomicInteger shared_i;

		public AddAfterMultiplyWorker1(double[][] a, double ac, double[][] b, double bc, double[][] c, AtomicInteger shared_i) {
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

		private double[][] b;

		private double[][] c;

		private double[] ac;

		private double[] bc;

		private AtomicInteger shared_i;

		public AddAfterMultiplyWorker2(double[][] a, double[] ac, double[][] b, double[] bc, double[][] c, AtomicInteger shared_i) {
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

		private double[][] b;

		private double[][] c;

		private double[][] ac;

		private double[][] bc;

		private AtomicInteger shared_i;

		public AddAfterMultiplyWorker3(double[][] a, double[][] ac, double[][] b, double[][] bc, double[][] c, AtomicInteger shared_i) {
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

	public static class EqualColumnProductWorker implements Callable<Double> {
		private double[][] a;

		private double[][] b;

		private double[][] c;

		private AtomicInteger shared_j;

		public EqualColumnProductWorker(double[][] a, double[][] b, double[][] c, AtomicInteger shared_j) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.shared_j = shared_j;
		}

		@Override
		public Double call() throws Exception {
			int a_rows = a.length;
			int b_rows = b.length;

			int j = 0;
			double sum = 0;

			while ((j = shared_j.getAndIncrement()) < b_rows) {
				for (int i = 0; i < a_rows; i++) {
					c[i][j] = ArrayMath.dotProduct(a[i], b[j]);
					sum += c[i][j];
				}
			}
			return sum;
		}
	}

	public static class EqualColumnProductWorker2 implements Callable<Double> {
		private SparseMatrix a;

		private SparseMatrix b;

		private CounterMap<Integer, Integer> c;

		private AtomicInteger row_cnt;

		private boolean symmetric;

		public EqualColumnProductWorker2(SparseMatrix a, SparseMatrix b, CounterMap<Integer, Integer> c, AtomicInteger loc,
				boolean symmetric) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.row_cnt = loc;
			this.symmetric = symmetric;
		}

		@Override
		public Double call() throws Exception {
			int m = 0;
			double sum = 0;

			while ((m = row_cnt.getAndIncrement()) < a.rowSize()) {
				int i = a.indexAt(m);
				SparseVector sv1 = a.rowAt(m);

				Counter<Integer> cc = null;

				synchronized (c) {
					cc = c.getCounter(i);
				}

				if (symmetric) {

				} else {
					for (int n = 0; n < b.rowSize(); n++) {
						int j = b.indexAt(n);
						SparseVector sv2 = b.rowAt(n);
						double dot = VectorMath.dotProduct(sv1, sv2);
						if (dot != 0) {
							cc.setCount(j, dot);
						}
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

	public static class EqualRowProductWorker implements Callable<Double> {
		private double[][] a;

		private double[][] b;

		private double[][] c;

		private AtomicInteger shared_j;

		public EqualRowProductWorker(double[][] a, double[][] b, double[][] c, AtomicInteger shared_j) {
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

		private AtomicInteger shared_i;

		public OuterProductWorker(double[] a, double[] b, double[][] c, AtomicInteger shared_i) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.shared_i = shared_i;
		}

		@Override
		public Double call() throws Exception {
			int a_size = a.length;
			int b_size = b.length;

			int i = 0;
			double sum = 0;

			while ((i = shared_i.getAndIncrement()) < a_size) {
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

		private double[][] c;

		double[] bj;

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

		private AtomicInteger shared_j;

		public RowByColumnProductWorker(double[][] a, double[][] b, double[][] c, AtomicInteger shared_j) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.shared_j = shared_j;

		}

		@Override
		public Double call() throws Exception {
			int a_rows = a.length;
			int b_rows = b.length;
			int b_cols = b[0].length;

			double sum = 0;
			double[] bj = new double[b_rows];
			int j = 0;

			while ((j = shared_j.getAndIncrement()) < b_cols) {
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
			sb.append(String.format("target column: %d", shared_j));
			return sb.toString();
		}
	}

	public static class RowBySingleColumnProductWorker implements Callable<Double> {
		private double[][] a;

		private double[] b;

		private double[] c;

		private AtomicInteger shared_i;

		public RowBySingleColumnProductWorker(double[][] a, double[] b, double[] c, AtomicInteger shared_i) {
			this.a = a;
			this.b = b;
			this.c = c;
			this.shared_i = shared_i;
		}

		@Override
		public Double call() throws Exception {
			double sum = 0;
			int a_rows = a.length;
			int i = 0;

			while ((i = shared_i.getAndIncrement()) < a_rows) {
				c[i] = ArrayMath.dotProduct(a[i], b);
				sum += c[i];
			}
			return sum;
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
}
