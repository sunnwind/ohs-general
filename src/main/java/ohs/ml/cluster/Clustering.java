package ohs.ml.cluster;

import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.ml.cluster.SimilarityMetric.Type;

public abstract class Clustering {

	public static void add(SparseVector a, DenseVector b) {
		for (int i = 0; i < a.size(); i++) {
			int idx = a.indexAt(i);
			double val = a.valueAt(i);
			b.add(idx, val);
		}
	}

	protected SimilarityMetric sm = new SimilarityMetric(Type.COSINE);

	protected int thread_size = 10;

	public SimilarityMetric getSimiarityMetric() {
		return sm;
	}

	public void setSimilarityMetric(SimilarityMetric sm) {
		this.sm = sm;
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}
}
