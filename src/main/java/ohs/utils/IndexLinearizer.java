package ohs.utils;

public class IndexLinearizer {

	private long max_i;

	public IndexLinearizer(long max_i) {
		this.max_i = max_i;
	}

	public long getSingleIndex(long i, long j) {
		return i * max_i + j;
	}

	public long[] getTwoIndexes(long k) {
		long i = (k / max_i);
		long j = (k % max_i);
		return new long[] { i, j };
	}

}
