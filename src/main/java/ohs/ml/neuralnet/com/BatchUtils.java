package ohs.ml.neuralnet.com;

import java.util.List;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Pair;
import ohs.utils.Generics;

public class BatchUtils {

	public static Pair<DenseMatrix, DenseVector> getBatch(DenseMatrix X, DenseVector Y, int[] range, int[] data_locs) {

		int batch_size = range[1] - range[0];
		List<DenseVector> xx = Generics.newArrayList(batch_size);
		List<Double> yy = Generics.newArrayList(batch_size);

		for (int i = range[0]; i < range[1]; i++) {
			int loc = data_locs[i];
			DenseVector x = X.row(loc);
			int answer = (int) Y.value(loc);

			xx.add(x);
			yy.add((double) answer);
		}
		return Generics.newPair(new DenseMatrix(xx), new DenseVector(yy));
	}

	public static int[][] getBatch(int[][] X, int[] range, int[] data_locs) {
		int batch_size = range[1] - range[0];
		int[][] miniX = new int[batch_size][];
		int loc = 0;
		for (int i = range[0], j = 0; i < range[1]; i++, j++) {
			loc = data_locs[i];
			miniX[j] = X[loc];
		}
		return miniX;
	}

	public static int getMaxBatchSize(int[][] ranges) {
		int ret = 0;
		for (int i = 0; i < ranges.length; i++) {
			int[] range = ranges[i];
			int size = range[1] - range[0];
			ret = Math.max(ret, size);
		}
		return ret;
	}

	public static int[][] getBatchRanges(int data_size, int batch_size) {
		int[][] batch_ranges = null;

		if (data_size < batch_size) {
			batch_ranges = new int[1][];
			batch_ranges[0] = new int[] { 0, data_size };
		} else {
			int batch_cnt = (int) Math.ceil(1f * data_size / batch_size);

			int num_remains = data_size % batch_size;

			if (num_remains != 0 && num_remains < batch_size / 2f) {
				batch_cnt -= 1;
			}

			batch_ranges = new int[batch_cnt][2];

			for (int i = 0; i < batch_cnt; i++) {
				int start = i * batch_size;
				int end = (i + 1) * batch_size;

				if (i == batch_cnt - 1) {
					end = data_size;
				}
				batch_ranges[i] = new int[] { start, end };
			}
		}

		return batch_ranges;
	}

	public static int[] getIndexes(int[] data_locs, int[] range) {
		int start = range[0];
		int end = range[1];
		int[] ret = new int[end - start];
		for (int i = start, j = 0; i < end; i++, j++) {
			ret[j] = data_locs[i];
		}
		return ret;
	}

	public static SparseMatrix getSparseMiniBatchX(List<SparseVector> X, int[] locs, int[] range) {
		int size = range[1] - range[0];
		SparseVector[] rows = new SparseVector[size];
		for (int i = range[0], j = 0; i < range[1]; i++, j++) {
			int loc = locs[i];
			rows[j] = X.get(loc).copy();
		}
		return new SparseMatrix(rows);
	}

	public static int progress(long cnt, long max_cnt) {
		long chunk_size = (long) Math.ceil(1f * max_cnt / Math.min(100, max_cnt));
		int ret = 0;
		if (cnt == 0) {

		} else if (cnt == max_cnt) {
			ret = 100;
		} else if (cnt % chunk_size == 0) {
			ret = (int) (cnt / chunk_size);
		}
		return ret;
	}
}
