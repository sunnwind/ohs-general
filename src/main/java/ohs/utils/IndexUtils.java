package ohs.utils;

public class IndexUtils {

	public static long getSingleIndex(long i, long max_i, long j) {
		return i * max_i + j;
	}

	public static long[] getTwoIndexes(long k, long max_i) {
		long i = (k / max_i);
		long j = (k % max_i);
		return new long[] { i, j };
	}
}
