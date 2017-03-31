package ohs.ml.cluster;

public class HierarchicalClustering extends Clustering {
	public static enum LinkageCriteria {
		MAX, MIN, CENTROID, GROUP_AVERAGE
	}

	protected LinkageCriteria lc = LinkageCriteria.CENTROID;

	protected int max_cluster_size = Integer.MAX_VALUE;

	protected double min_sim = 0;

	public void setMaxClusterSize(int max_cluster_size) {
		this.max_cluster_size = max_cluster_size;
	}

	public void setMinSimilarity(double min_sim) {
		this.min_sim = min_sim;
	}

	public void setLinkageCriteria(LinkageCriteria lc) {
		this.lc = lc;
	}
}
