package ohs.ml.cluster;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.RawDocumentCollection;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorMath;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.cluster.SimilarityMetric.Type;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Pair;
import ohs.types.number.DoubleArray;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * http://people.revoledu.com/kardi/tutorial/Clustering/Numerical%20Example.htm
 * 
 * https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-agglomerative-clustering-1.html
 * 
 * @author ohs
 */
public class HierarchicalAgglomerativeClustering extends HierarchicalClustering {

	private class Worker1 implements Callable<Double> {

		private CounterMap<Integer, Integer> dataSims;

		private AtomicInteger data_cnt;

		private SparseMatrix X;

		private Timer timer;

		public Worker1(CounterMap<Integer, Integer> dataSims, SparseMatrix X, AtomicInteger data_cnt, Timer timer) {
			this.dataSims = dataSims;
			this.X = X;
			this.data_cnt = data_cnt;
			this.timer = timer;
		}

		@Override
		public Double call() throws Exception {
			double ret = 0;
			int i = 0;

			while ((i = data_cnt.getAndIncrement()) < X.rowSize()) {
				SparseVector x1 = X.rowAt(i);

				Counter<Integer> c = Generics.newCounter(X.rowSize() - i + 1);

				for (int j = i + 1; j < X.rowSize(); j++) {
					SparseVector x2 = X.rowAt(j);
					double sim = sm.getSimilarity(x1, x2);
					c.setCount(j, sim);
				}

				synchronized (dataSims) {
					dataSims.setCounter(i, c);
				}
			}

			return ret;
		}
	}

	private class Worker2 implements Callable<Double> {

		private CounterMap<Integer, Integer> dataSims;

		private AtomicInteger c_cnt;

		private IntegerArray p_idxs;

		private CounterMap<Integer, Integer> childMap;

		private CounterMap<Integer, Integer> clusterSims;

		private Timer timer;

		public Worker2(IntegerArray p_idxs, CounterMap<Integer, Integer> childMap, CounterMap<Integer, Integer> dataSims,
				AtomicInteger c_cnt, CounterMap<Integer, Integer> clusterSims, Timer timer) {
			this.p_idxs = p_idxs;
			this.childMap = childMap;
			this.dataSims = dataSims;
			this.clusterSims = clusterSims;
			this.c_cnt = c_cnt;
			this.timer = timer;
		}

		@Override
		public Double call() throws Exception {
			double ret = 0;
			int i = 0;

			while ((i = c_cnt.getAndIncrement()) < p_idxs.size()) {
				int p_idx1 = p_idxs.get(i);
				Counter<Integer> c1 = childMap.getCounter(p_idx1);
				Counter<Integer> cent1 = Generics.newCounter();

				if (lc == LinkageCriteria.GROUP_AVERAGE) {
					for (int c_idx : c1.keySet()) {
						SparseVector x = X.rowAt(c_idx);
						VectorMath.add(x, cent1);
					}
				}

				Counter<Integer> c3 = Generics.newCounter(p_idxs.size() - i + 1);

				for (int j = i + 1; j < p_idxs.size(); j++) {
					int p_idx2 = p_idxs.get(j);
					Counter<Integer> c2 = childMap.getCounter(p_idx2);
					double sim = 0;

					if (lc == LinkageCriteria.GROUP_AVERAGE) {
						Counter<Integer> cent2 = Generics.newCounter(cent1);

						for (int c_idx : c2.keySet()) {
							SparseVector x = X.rowAt(c_idx);
							VectorMath.add(x, cent2);
						}

						SparseVector cent = new SparseVector(cent2);
						double dot_product = VectorMath.dotProduct(cent, cent);
						int size = c1.size() + c2.size();
						sim = (dot_product - size) / (size * (size - 1));
					} else {
						double min_sim = Double.MAX_VALUE;
						double max_sim = 0;
						double avg_sim = 0;
						int cnt = 0;

						for (int c_idx1 : c1.keySet()) {
							for (int c_idx2 : c2.keySet()) {
								double data_sim = dataSims.getCount(c_idx1, c_idx2);

								if (data_sim == 0) {
									data_sim = dataSims.getCount(c_idx2, c_idx1);
								}
								min_sim = Math.min(min_sim, data_sim);
								max_sim = Math.max(max_sim, data_sim);
								avg_sim += data_sim;
								cnt++;
							}
						}

						avg_sim /= cnt;

						if (lc == LinkageCriteria.CENTROID) {
							sim = avg_sim;
						} else if (lc == LinkageCriteria.MAX) {
							sim = max_sim;
						} else if (lc == LinkageCriteria.MIN) {
							sim = min_sim;
						}
					}
					c3.setCount(p_idx2, sim);
				}

				synchronized (clusterSims) {
					if (c3.size() > 0) {
						clusterSims.setCounter(p_idx1, c3);
					}
				}
			}

			return ret;
		}
	}

	public static DocumentCollection dc;

	public static RawDocumentCollection rdc;

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		dc = new DocumentCollection(MIRPath.OHSUMED_COL_DC_DIR);
		rdc = new RawDocumentCollection(MIRPath.OHSUMED_COL_DC_DIR);

		List<SparseVector> dvs = Generics.newLinkedList();
		for (Pair<String, IntegerArray> p : dc.getRange(0, 5000, false)) {
			Counter<Integer> c = Generics.newCounter();
			for (int w : p.getSecond()) {
				if (w == DocumentCollection.SENT_END) {
					continue;
				}
				c.incrementCount(w, 1);
			}

			SparseVector dv = new SparseVector(c);

			if (dv.size() > 0) {
				dvs.add(dv);
			}
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

		HierarchicalAgglomerativeClustering kc = new HierarchicalAgglomerativeClustering(X);
		kc.setThreadSize(10);
		kc.setSimilarityMetric(new SimilarityMetric(Type.COSINE));
		kc.setMaxClusterSize(3);
		// kc.setMinSimilarity(0.3);
		kc.setLinkageCriteria(LinkageCriteria.CENTROID);
		kc.cluster();

		System.out.println("process ends.");
	}

	private CounterMap<Integer, Integer> dataSims;

	private SparseMatrix X;

	private IntegerArray parentMap;

	private IntegerArray mergeOrders;

	private DoubleArray mergeSims;

	private CounterMap<Integer, Integer> childMap;

	public HierarchicalAgglomerativeClustering(SparseMatrix X) {
		this.X = X;

		int data_size = X.rowSize();

		parentMap = new IntegerArray(new int[data_size]);
		mergeOrders = new IntegerArray(new int[data_size]);
		mergeSims = new DoubleArray(new double[data_size]);
		childMap = Generics.newCounterMap(data_size);

		for (int i = 0; i < data_size; i++) {
			childMap.getCounter(i).setCount(i, 1);
			parentMap.set(i, -1);
			mergeOrders.set(i, -1);
		}
		dataSims = Generics.newCounterMap(data_size - 1);
	}

	public IntegerArray cluster() throws Exception {
		System.out.println("cluster");
		Timer timer = Timer.newTimer();

		computeDataSimilarityMatrix();

		int merge_cnt = 0;

		while (childMap.size() > 1) {
			if (childMap.size() == max_cluster_size) {
				break;
			}

			CounterMap<Integer, Integer> clusterSims = computeClusterSimilarityMatrix();
			Pair<Integer, Integer> p = clusterSims.argMax();
			double sim = clusterSims.getCount(p.getFirst(), p.getSecond());

			int p_idx = p.getFirst();
			int c_idx = p.getSecond();

			if (c_idx < p_idx) {
				p_idx = p.getSecond();
				c_idx = p.getFirst();
			}

			if (sim < min_sim) {
				break;
			}

			// List<String> attrs = rdc.getAttrData().get(0);
			// SparseVector x1 = getCentroid(p_idx);
			// SparseVector x2 = getCentroid(c_idx);

			// System.out.println(rdc.getValues(p_idx).get(1));
			// System.out.println(rdc.getValues(p_idx).get(3));
			// // System.out.println(rdc.getValues(p_idx).get(5));
			// System.out.println(VectorUtils.toCounter(x1, dc.getVocab()));
			// System.out.println("----------------------------");
			// System.out.println(rdc.getValues(c_idx).get(1));
			// System.out.println(rdc.getValues(c_idx).get(3));
			// // System.out.println(rdc.getValues(c_idx).get(5));
			// System.out.println(VectorUtils.toCounter(x2, dc.getVocab()));
			// System.out.println("============================");
			// System.out.printf("SIM : [%f]\n", sim);
			// System.out.println();

			Counter<Integer> c1 = childMap.removeKey(c_idx);
			Counter<Integer> c2 = childMap.getCounter(p_idx);

			int size1 = c1.size();
			int size2 = c2.size();

			c2.incrementAll(c1);

			int size3 = c2.size();

			System.out.printf("(%d, %d) + (%d, %d) with sim(%f) => (%d, %d)\n", p_idx, size1, c_idx, size2, sim, p_idx, size3);

			if (parentMap.get(c_idx) == -1) {
				parentMap.set(c_idx, p_idx);
			}

			mergeOrders.set(c_idx, merge_cnt++);
			mergeSims.set(c_idx, sim);

			// System.out.printf("[%d/%d clusters, %d -> %d, %s]\n", childMap.size(), X.rowSize(), c_idx, p_idx, timer.stop());

			// System.out.println(mergeOrders);
			// System.out.println(parentMap);
			// System.out.println(childMap);
			// System.out.println();

			// System.out.println(mergeOrders);
			// System.out.println(parentMap);
			// System.out.println(childMap);
			// System.out.println();

		}

		System.out.println(mergeOrders);
		System.out.println(parentMap);
		System.out.println(childMap);
		System.out.println();
		return null;
	}

	private CounterMap<Integer, Integer> computeClusterSimilarityMatrix() throws Exception {
		Timer timer = Timer.newTimer();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);
		List<Future<Double>> fs = Generics.newArrayList(thread_size);

		AtomicInteger c_cnt = new AtomicInteger(0);
		IntegerArray p_idxs = new IntegerArray(childMap.keySet());
		p_idxs.sort(false);

		CounterMap<Integer, Integer> ret = Generics.newCounterMap(p_idxs.size());

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker2(p_idxs, childMap, dataSims, c_cnt, ret, timer)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

		return ret;
	}

	private void computeDataSimilarityMatrix() throws Exception {
		System.out.printf("compute data sims.\n");
		Timer timer = Timer.newTimer();

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);
		List<Future<Double>> fs = Generics.newArrayList(thread_size);
		AtomicInteger data_cnt = new AtomicInteger(0);

		for (int k = 0; k < thread_size; k++) {
			fs.add(tpe.submit(new Worker1(dataSims, X, data_cnt, timer)));
		}

		for (int k = 0; k < thread_size; k++) {
			fs.get(k).get();
		}

		tpe.shutdown();
	}

}
