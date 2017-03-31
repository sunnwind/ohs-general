package ohs.ml.cluster;

import ohs.math.VectorMath;
import ohs.matrix.Vector;

/**
 * 
 * https://en.wikipedia.org/wiki/Hierarchical_clustering
 * 
 * @author ohs
 */
public class SimilarityMetric {

	public static enum Type {
		COSINE, EUCLIDEAN
	}

	private Type type;

	public SimilarityMetric(Type type) {
		this.type = type;
	}

	public double getSimilarity(Vector a, Vector b) {
		double ret = 0;
		if (type == Type.COSINE) {
			ret = VectorMath.cosine(a, b);
		}
		return ret;
	}

}
