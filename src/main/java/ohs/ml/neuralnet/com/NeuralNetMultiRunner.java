package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.matrix.DenseMatrix;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public class NeuralNetMultiRunner {
	public class Worker implements Callable<Pair<Double, Integer>> {

		private NeuralNet nn;

		public Worker(NeuralNet nn) {
			this.nn = nn;
		}

		@Override
		public Pair<Double, Integer> call() throws Exception {

			int range_loc = 0;

			while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
				int[] r = ranges[range_loc];

				if (tt == TaskType.CLASSIFICATION) {
					DenseMatrix X_ = TaskType.toDenseMatrix(X);
					IntegerArray Y_ = TaskType.toIntegerArray(Y);

					DenseMatrix Xm = X_.rows(r[0], r[1] - r[0]);
					IntegerArray Yhm = nn.classify(Xm);

					for (int i = 0; i < Yhm.size(); i++) {
						int j = i + r[0];
						Y_.set(j, Yhm.get(i));
					}
				} else if (tt == TaskType.SEQ_CLASSIFICATION) {
					IntegerMatrix X_ = TaskType.toIntegerMatrix(X);
					IntegerArray Y_ = TaskType.toIntegerArray(Y);

					IntegerMatrix Xm = X_.subMatrix(r[1], r[0]);
					IntegerArray Yhm = nn.classify(Xm);

					for (int i = 0; i < Yhm.size(); i++) {
						int j = i + r[0];
						Y_.set(j, Yhm.get(i));
					}
				} else if (tt == TaskType.SEQ_LABELING) {
					IntegerMatrix X_ = TaskType.toIntegerMatrix(X);
					IntegerMatrix Y_ = TaskType.toIntegerMatrix(Y);

					for (int i = r[0]; i < r[1]; i++) {
						IntegerArray Xm = X_.get(i);
						IntegerArray Yhm = nn.classify(Xm);

						if (Xm.size() != Yhm.size()) {
							System.out.println();
						}

						Y_.set(i, Yhm);
					}
				}
			}

			return Generics.newPair(0d, 0);
		}

	}

	private List<NeuralNet> nns;

	private ThreadPoolExecutor tpe;

	private List<Worker> ws;

	private TaskType tt;

	private Object Y;

	private Object X;

	private int[][] ranges;

	private AtomicInteger range_cnt;

	public NeuralNetMultiRunner(List<NeuralNet> nns) {
		this.nns = nns;
		this.tt = nns.get(0).getTaskType();

		tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(nns.size());

		ws = Generics.newArrayList(nns.size());

		for (int i = 0; i < nns.size(); i++) {
			ws.add(new Worker(nns.get(i)));
		}
	}

	public Object classify(Object X) throws Exception {
		this.X = X;
		int size = 0;

		if (tt == TaskType.CLASSIFICATION) {
			size = TaskType.toDenseMatrix(X).rowSize();
			Y = new IntegerArray(new int[size]);
		} else if (tt == TaskType.SEQ_CLASSIFICATION) {
			size = TaskType.toIntegerMatrix(X).size();
			Y = new IntegerArray(new int[size]);
		} else if (tt == TaskType.SEQ_LABELING) {
			IntegerMatrix X_ = TaskType.toIntegerMatrix(X);
			size = X_.size();
			IntegerMatrix Y_ = new IntegerMatrix(size);

			for (int i = 0; i < size; i++) {
				Y_.add(new IntegerArray());
			}
			Y = Y_;
		}

		ranges = BatchUtils.getBatchRanges(size, 20);

		range_cnt = new AtomicInteger(0);

		List<Future<Pair<Double, Integer>>> fs = Generics.newArrayList(ws.size());

		for (int j = 0; j < ws.size(); j++) {
			fs.add(tpe.submit(ws.get(j)));
		}

		for (int j = 0; j < fs.size(); j++) {
			Pair<Double, Integer> res = fs.get(j).get();
		}

		return Y;
	}

	public void shutdown() {
		tpe.shutdown();
	}
}