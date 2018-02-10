package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
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
				DenseTensor Xm = X.subTensor(r[0], r[1]);
				DenseTensor Yhm = nn.classify(Xm);

				for (int i = 0; i < Yhm.size(); i++) {
					Yh.set(r[0] + i, Yhm.get(i));
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

	private DenseTensor X;

	private DenseTensor Yh;

	public NeuralNetMultiRunner(List<NeuralNet> nns) {
		this.nns = nns;
		this.tt = nns.get(0).getTaskType();

		tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(nns.size());

		ws = Generics.newArrayList(nns.size());

		for (int i = 0; i < nns.size(); i++) {
			ws.add(new Worker(nns.get(i)));
		}
	}

	public DenseTensor classify(DenseTensor X) throws Exception {
		this.X = X;

		Yh = new DenseTensor();
		Yh.ensureCapacity(X.size());

		for (int i = 0; i < X.size(); i++) {
			Yh.add(new DenseMatrix());
		}

		ranges = BatchUtils.getBatchRanges(X.size(), 20);

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