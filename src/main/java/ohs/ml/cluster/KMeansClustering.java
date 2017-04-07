package ohs.ml.cluster;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class KMeansClustering extends Clustering {

	private class Worker implements Callable<Double> {

		private SparseMatrix X;

		private SparseMatrix C;

		private AtomicInteger data_cnt;

		private Timer timer;

		public Worker(SparseMatrix X, SparseMatrix C, AtomicInteger data_cnt, Timer timer) {
			this.X = X;
			this.C = C;
			this.data_cnt = data_cnt;
			this.timer = timer;
		}

		@Override
		public Double call() throws Exception {
			DenseVector yh = new DenseVector(C.rowSize());
			double ret = 0;
			int i = 0;
			int cnt = 0;

			while ((i = data_cnt.getAndIncrement()) < X.rowSize()) {
				cnt++;
				SparseVector x = X.rowAt(i);

				for (int j = 0; j < C.rowSize(); j++) {
					SparseVector c = C.rowAt(j);
					double sim = sm.getSimilarity(x, c);
					yh.set(j, sim);
				}

				int idx = yh.argMax();
				double max_sim = yh.value(idx);

				Y.set(i, idx);
				ret += max_sim;

				int prog = BatchUtils.progress(i + 1, X.rowSize());

				// if (prog > 0) {
				// System.out.printf("[%d percent, %d/%d, %s]\n", prog, loc + 1, X.rowSize(), timer.stop());
				// }
			}
			// ret /= cnt;
			return ret;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DocumentCollection dc = new DocumentCollection(MIRPath.OHSUMED_COL_DC_DIR);
		List<SparseVector> dvs = Generics.newLinkedList();
		for (Pair<String, IntegerArray> p : dc.get(0, 20000)) {
			Counter<Integer> c = Generics.newCounter();
			for (int w : p.getSecond()) {
				if (w == DocumentCollection.SENT_END) {
					continue;
				}
				c.incrementCount(w, 1);
			}

			SparseVector dv = new SparseVector(c);
			dvs.add(dv);
		}

		for (SparseVector dv : dvs) {
			double norm = 0;
			double doc_cnt = dc.getVocab().getDocCnt();
			for (int j = 0; j < dv.size(); j++) {
				int w = dv.indexAt(j);
				double cnt = dv.valueAt(j);
				double doc_freq = dc.getVocab().getDocFreq(w);
				double weight = TermWeighting.tfidf(cnt, doc_cnt, doc_freq);
				norm += (weight * weight);
				dv.setAt(j, weight);
			}
			norm = Math.sqrt(norm);
			dv.multiply(1f / norm);
		}

		int feat_size = 0;

		for (SparseVector dv : dvs) {
			feat_size = Math.max(feat_size, dv.argMax());
		}
		feat_size += 1;

		SparseMatrix X = new SparseMatrix(dvs);

		KMeansClustering kc = new KMeansClustering(X, 10, feat_size);
		kc.setThreadSize(1);
		kc.cluster(10000);

		System.out.println("process ends.");
	}

	private SparseMatrix X;

	private IntegerArray Y;

	// private DenseMatrix C;

	private SparseMatrix C;

	private int feat_size;

	public KMeansClustering(SparseMatrix X, int cluster_size, int feat_size) {
		this.X = X;
		this.feat_size = feat_size;
		// C = new DenseMatrix(cluster_size, feat_size);

		int[] idxs = ArrayUtils.range(cluster_size);
		SparseVector[] rows = new SparseVector[cluster_size];

		for (int i = 0; i < rows.length; i++) {
			SparseVector row = new SparseVector(ArrayUtils.range(feat_size), new double[feat_size]);
			rows[i] = row;

			int input_size = rows.length;
			double bound = 1f / Math.sqrt(input_size);
			ArrayMath.random(-bound, bound, row.values());
		}

		C = new SparseMatrix(idxs, rows);

		Y = new IntegerArray(new int[X.size()]);

	}

	public IntegerArray cluster(int max_iter) throws Exception {
		System.out.println("cluster");
		Timer timer = Timer.newTimer();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);
		List<Future<Double>> fs = Generics.newArrayList(thread_size);

		IntegerArray Yo = new IntegerArray(new int[Y.size()]);
		IntegerArray Ym = new IntegerArray(new int[Y.size()]);

		ArrayUtils.copy(Y.values(), Yo.values());
		ArrayUtils.copy(Y.values(), Ym.values());

		double diff_min = Double.MAX_VALUE;

		for (int iter = 0; iter < max_iter; iter++) {
			AtomicInteger data_cnt = new AtomicInteger(0);

			for (int i = 0; i < thread_size; i++) {
				fs.add(tpe.submit(new Worker(X, C, data_cnt, timer)));
			}

			for (int i = 0; i < thread_size; i++) {
				fs.get(i).get();
			}
			fs.clear();

			double diff = 0;
			for (int i = 0; i < Y.size(); i++) {
				if (Y.get(i) != Yo.get(i)) {
					diff++;
				}
			}

			if (diff < diff_min) {
				diff_min = diff;
				ArrayUtils.copy(Y.values(), Ym.values());
			}

			if (diff == 0) {
				break;
			}

			ArrayUtils.copy(Y.values(), Yo.values());

			IntegerArrayMatrix G = DataSplitter.group(Y);
			Counter<Integer> dataCnts = Generics.newCounter();

			DenseVector nc = new DenseVector(feat_size);

			for (int i = 0; i < C.rowSize(); i++) {
				SparseVector c = C.rowAt(i);
				c.setAll(0);

				IntegerArray locs = G.get(i);

				nc.clear();

				for (int loc : locs) {
					SparseVector x = X.get(loc);

					for (int j = 0; j < x.size(); j++) {
						int idx = x.indexAt(j);
						double val = x.valueAt(j);
						nc.add(idx, val);
					}

				}
				nc.multiply(1f / locs.size());

				C.setRowAt(i, nc.toSparseVector());

				dataCnts.setCount(i, locs.size());
			}

			System.out.printf("[%d, %f, %f, %s] - %s\n", iter, diff, diff_min, timer.stop(), dataCnts.toString(dataCnts.size()));
		}

		tpe.shutdown();

		return Ym;
	}

	public SparseMatrix getCentroids() {
		return C;
	}

	public void initialize() {
		DenseVector c_avg = new DenseVector(C.colSize());
		for (SparseVector x : X) {
			for (int i = 0; i < x.size(); i++) {
				int idx = x.indexAt(i);
				double val = x.valueAt(i);
				c_avg.add(idx, val);
			}
		}
		c_avg.multiply(1f / X.rowSize());

		SparseVector dists = new SparseVector(X.rowSize());

		for (int i = 0; i < X.rowSize(); i++) {
			double dist = VectorMath.euclideanDistance(X.rowAt(i), c_avg);
			dists.addAt(i, i, dist);
		}
	}

}
