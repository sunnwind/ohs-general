package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.matrix.DenseMatrix;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.types.number.IntegerTensor;
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
					DenseMatrix _X = (DenseMatrix) X;
					IntegerArray _Yh = (IntegerArray) Yh;

					DenseMatrix Xm = _X.rows(r[0], r[1] - r[0]);
					IntegerArray Yhm = nn.classify(Xm);

					for (int i = 0; i < Yhm.size(); i++) {
						int j = i + r[0];
						_Yh.set(j, Yhm.get(i));
					}
				} else if (tt == TaskType.SEQ_CLASSIFICATION) {
					IntegerTensor _X = (IntegerTensor) X;
					IntegerArray _Yh = TaskType.toIntegerArray(Yh);

					for (int i = r[0]; i < r[1]; i++) {
						IntegerArray Yhm = nn.classify(_X.get(i).toIntegerTensor());
						int j = i + r[0];
						_Yh.set(j, Yhm.get(0));
					}
				} else if (tt == TaskType.SEQ_LABELING) {
					IntegerTensor _X = (IntegerTensor) X;
					IntegerMatrix _Yh = (IntegerMatrix) Yh;

					for (int i = r[0]; i < r[1]; i++) {
						IntegerTensor Xm = _X.get(i).toIntegerTensor();
						IntegerArray Yhm = nn.classify(Xm);
						_Yh.set(i, Yhm);
					}
				}
			}

			return Generics.newPair(0d, 0);
		}

	}

	private List<NeuralNet> nns;

	private AtomicInteger range_cnt;

	private int[][] ranges;

	private ThreadPoolExecutor tpe;

	private TaskType tt;

	private List<Worker> ws;

	private Object X;

	private Object Yh;

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
			DenseMatrix _X = (DenseMatrix) X;
			size = _X.rowSize();
			Yh = new IntegerArray(new int[size]);
		} else if (tt == TaskType.SEQ_CLASSIFICATION) {
			IntegerTensor _X = (IntegerTensor) X;
			size = _X.size();
			Yh = new IntegerArray(new int[size]);
		} else if (tt == TaskType.SEQ_LABELING) {
			IntegerTensor _X = (IntegerTensor) X;
			size = _X.size();
			IntegerMatrix _Y = new IntegerMatrix(size);

			for (int i = 0; i < size; i++) {
				_Y.add(new IntegerArray());
			}
			Yh = _Y;
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

		return Yh;
	}

	public void shutdown() {
		tpe.shutdown();
	}
}