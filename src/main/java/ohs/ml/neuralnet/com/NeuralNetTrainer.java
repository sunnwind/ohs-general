package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.eval.Performance;
import ohs.ml.eval.PerformanceEvaluator;
import ohs.ml.neuralnet.cost.CrossEntropyCostFunction;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;
import ohs.utils.Timer;

/**
 * http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
 * 
 * http://cs231n.stanford.edu/syllabus.html
 * 
 * http://cs231n.github.io/
 * 
 * @author ohs
 */
public class NeuralNetTrainer {

	public class Worker implements Callable<Pair<Double, Integer>> {

		private CrossEntropyCostFunction cf;

		private NeuralNet nn;

		private ParameterUpdater pu;

		public Worker(ParameterUpdater pu) {
			this.pu = pu;
			this.nn = pu.getNeuralNet();
			cf = new CrossEntropyCostFunction();
		}

		@Override
		public Pair<Double, Integer> call() throws Exception {
			int correct_cnt = 0;
			double cost = 0;

			if (X instanceof DenseMatrix) {
				int range_loc = 0;

				DenseMatrix X_ = (DenseMatrix) X;
				IntegerArray Y_ = (IntegerArray) Y;

				while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
					int[] range = ranges[range_loc];
					int[] locs = BatchUtils.getIndexes(data_locs, range);

					DenseMatrix Xm = X_.rows(locs);
					IntegerArray Ym = Y_.get(locs);

					DenseMatrix Yh = (DenseMatrix) nn.forward(Xm);
					DenseMatrix D = cf.evaluate(Yh, Ym);

					cost += cf.getCost();
					correct_cnt += cf.getCorrectCnt();

					nn.backward(D);
					pu.update();
				}
			} else if (X instanceof IntegerMatrix) {
				if (Y instanceof IntegerArray) {
					int range_loc = 0;
					IntegerMatrix X_ = (IntegerMatrix) X;
					IntegerArray Y_ = (IntegerArray) Y;

					while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
						int[] range = ranges[range_loc];
						int[] locs = BatchUtils.getIndexes(data_locs, range);

						IntegerMatrix Xm = X_.subMatrix(locs);
						IntegerArray Ym = Y_.subArray(locs);

						DenseMatrix Yh = (DenseMatrix) nn.forward(Xm);
						DenseMatrix D = cf.evaluate(Yh, Ym);

						cost += cf.getCost();
						correct_cnt += cf.getCorrectCnt();

						nn.backward(D);
						pu.update();
					}
				} else if (Y instanceof IntegerMatrix) {
					int data_loc = 0;
					int batch_size = param.getBatchSize();
					IntegerMatrix X_ = (IntegerMatrix) X;
					IntegerMatrix Y_ = (IntegerMatrix) Y;

					while ((data_loc = range_cnt.getAndIncrement()) < data_locs.length) {
						int loc = data_locs[data_loc];
						IntegerArray x = X_.get(loc);
						IntegerArray y = Y_.get(loc);

						int[][] ranges = BatchUtils.getBatchRanges(x.size(), batch_size);

						for (int i = 0; i < ranges.length; i++) {
							int[] range = ranges[i];

							IntegerArray Xm = x.subArray(range[0], range[1]);
							IntegerArray Ym = y.subArray(range[0], range[1]);

							DenseMatrix Yh = (DenseMatrix) nn.forward(Xm);
							DenseMatrix D = cf.evaluate(Yh, Ym);

							cost += cf.getCost();
							correct_cnt += cf.getCorrectCnt();

							nn.backward(D);
							pu.update();
						}
					}
				}
			}

			readyForNextIteration();

			return Generics.newPair(cost, correct_cnt);
		}

		private void readyForNextIteration() {
			for (Layer l : nn) {
				if (l instanceof RnnLayer) {
					RnnLayer n = (RnnLayer) l;
					n.resetH0();
				} else if (l instanceof LstmLayer) {
					LstmLayer n = (LstmLayer) l;
					n.resetH0();
				} else if (l instanceof BidirectionalRecurrentLayer) {
					BidirectionalRecurrentLayer n = (BidirectionalRecurrentLayer) l;
					n.resetH0();
				}
			}
		}
	}

	public static NeuralNet copy(NeuralNet nn) {
		NeuralNet ret = new NeuralNet();
		for (Layer l : nn) {
			ret.add(l.copy());
		}
		return ret;

	}

	public static List<NeuralNet> copy(NeuralNet nn, int size) {
		List<NeuralNet> ret = Generics.newArrayList();
		for (int i = 0; i < size; i++) {
			ret.add(copy(nn));
		}
		return ret;
	}

	private int[] data_locs;

	private int data_size;

	private PerformanceEvaluator eval = new PerformanceEvaluator();

	private Indexer<String> labelIndexer;

	private NeuralNet nn;

	public int pad_label = -1;

	private NeuralNetParams param;

	private AtomicInteger range_cnt;

	private int[][] ranges;

	private Timer timer1 = Timer.newTimer();

	private ThreadPoolExecutor tpe;

	private DenseMatrix W;

	private List<Worker> ws;

	private Object X;

	private Object Y;

	public NeuralNetTrainer(NeuralNet nn, NeuralNetParams param, int data_size, Indexer<String> labelIndexer)
			throws Exception {
		prepare(nn, param, data_size, labelIndexer);
	}

	public void finish() {
		tpe.shutdown();

		nn.setIsTesting(true);

		// VectorUtils.copy(W_best, W);
	}

	public void prepare(NeuralNet nn, NeuralNetParams param, int data_size, Indexer<String> labelIndexer)
			throws Exception {
		this.nn = nn;
		this.param = param;
		this.data_size = data_size;
		this.labelIndexer = labelIndexer;

		int thread_size = param.getThreadSize();
		List<NeuralNet> nns = copy(nn, thread_size - 1);

		for (NeuralNet n : nns) {
			n.prepare();
		}

		nns.add(nn);

		List<ParameterUpdater> pus = Generics.newArrayList(nns.size());

		for (NeuralNet n : nns) {
			ParameterUpdater pu = new ParameterUpdater(n, data_size);
			pu.setLearningRate(param.getLearnRate());
			pu.setWeightDecay(param.getRegLambda(), param.getLearnRate(), data_size);
			pus.add(pu);
		}

		{
			List<DenseVector> l = Generics.newLinkedList();

			for (DenseMatrix w : nn.getW(true)) {
				l.addAll(w);
			}
			W = new DenseMatrix(l);
		}

		tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		ws = Generics.newArrayList(thread_size);

		for (int i = 0; i < thread_size; i++) {
			ws.add(new Worker(pus.get(i)));
		}
	}

	// public void train(NeuralNetParams param, NeuralNet nn, Object X, Object Y,
	// Object Xt, Object Yt, int max_iter,
	// Indexer<String> labelIndexer) throws Exception {
	// Timer timer1 = Timer.newTimer();
	// this.param = param;
	// this.X = X;
	// this.Y = Y;
	//
	// this.data_size = 0;
	//
	// if (X instanceof DenseMatrix) {
	// data_size = ((DenseMatrix) X).rowSize();
	// data_locs = ArrayUtils.range(data_size);
	// ranges = BatchUtils.getBatchRanges(data_size, param.getBatchSize());
	// } else {
	// IntegerArrayMatrix XX = ((IntegerArrayMatrix) X);
	// for (IntegerArray x : XX) {
	// data_size += x.size();
	// }
	// data_locs = ArrayUtils.range(XX.size());
	// }
	//
	// int thread_size = param.getThreadSize();
	//
	// List<NeuralNet> nns = copy(nn, thread_size - 1);
	//
	// for (NeuralNet n : nns) {
	// n.prepareTraining();
	// }
	//
	// nns.add(nn);
	//
	// W = nn.getParameters();
	// rW1 = W.copy(true);
	// rW2 = W.copy(true);
	//
	// DenseMatrix W_best = W.copy(true);
	// DenseMatrix W_no_bias = nn.getW();
	//
	// ThreadPoolExecutor tpe = (ThreadPoolExecutor)
	// Executors.newFixedThreadPool(thread_size);
	//
	// List<Worker> ws = Generics.newArrayList(thread_size);
	//
	// for (int i = 0; i < thread_size; i++) {
	// ws.add(new Worker(nns.get(i)));
	// }
	//
	// double best_acc = 0;
	// double best_cost = 0;
	// int best_cor_cnt = 0;
	//
	// PerformanceEvaluator eval = new PerformanceEvaluator();
	// List<Performance> perfs = Generics.newLinkedList();
	//
	// for (int iter = 1; iter <= max_iter; iter++) {
	// Timer timer2 = Timer.newTimer();
	//
	// ArrayUtils.shuffle(data_locs);
	//
	// range_cnt = new AtomicInteger(0);
	//
	// List<Future<Pair<Double, Integer>>> fs = Generics.newArrayList(ws.size());
	// for (int i = 0; i < ws.size(); i++) {
	// fs.add(tpe.submit(ws.get(i)));
	// }
	//
	// double cost = 0;
	// int cor_cnt = 0;
	//
	// for (int i = 0; i < fs.size(); i++) {
	// Pair<Double, Integer> res = fs.get(i).get();
	// cost += res.getFirst();
	// cor_cnt += res.getSecond();
	// }
	//
	// if (param.getRegLambda() > 0) {
	// cost +=
	// CrossEntropyCostFunction.getL2RegularizationTerm(param.getRegLambda(),
	// W_no_bias, data_size);
	// }
	//
	// double acc = 1f * cor_cnt / data_size;
	//
	// if (best_cost < cost) {
	// best_cost = cost;
	// best_cor_cnt = cor_cnt;
	// best_acc = acc;
	//
	// VectorUtils.copy(W, W_best);
	// }
	//
	// double norm = VectorMath.normL2(W_no_bias);
	// System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s (%s), norm: %f,
	// learn-rate: %f\n", iter, cost, acc, cor_cnt,
	// data_size, timer2.stop(), timer1.stop(), norm, param.getLearningRate());
	//
	// if (Xt != null && Yt != null) {
	// nn.setIsTesting(true);
	//
	// IntegerArray yh = null;
	// IntegerArray y = null;
	//
	// if (X instanceof DenseMatrix) {
	// yh = nn.classify(Xt);
	// y = (IntegerArray) Yt;
	// } else {
	// IntegerArrayMatrix Xm = (IntegerArrayMatrix) Xt;
	// IntegerArrayMatrix Ym = (IntegerArrayMatrix) Yt;
	//
	// y = new IntegerArray(Xm.size());
	// yh = new IntegerArray(Xm.size());
	//
	// for (int i = 0; i < Xm.size(); i++) {
	// IntegerArray xm = Xm.get(i);
	// IntegerArray ym = Ym.get(i);
	// IntegerArray yhm = nn.classify(xm);
	//
	// for (int j = 0; j < ym.size(); j++) {
	// y.add(ym.get(j));
	// yh.add(yhm.get(j));
	// }
	// }
	// }
	//
	// Performance p = eval.evalute(y, yh, labelIndexer);
	// perfs.add(p);
	//
	// System.out.println(p.toString());
	// nn.setIsTesting(false);
	// }
	// }
	//
	// tpe.shutdown();
	//
	// nn.setIsTesting(true);
	//
	// VectorUtils.copy(W_best, W);
	// }

	public void setPadLabel(int pad_label) {
		this.pad_label = pad_label;
	}

	public void train(Object X, Object Y, Object Xt, Object Yt, int max_iter) throws Exception {
		this.X = X;
		this.Y = Y;

		int data_size = 0;

		if (X instanceof DenseMatrix) {
			data_size = ((DenseMatrix) X).rowSize();
			data_locs = ArrayUtils.range(data_size);
			ranges = BatchUtils.getBatchRanges(data_size, param.getBatchSize());
		} else {
			if (Y instanceof IntegerArray) {
				data_size = ((IntegerMatrix) X).size();
				data_locs = ArrayUtils.range(data_size);
				ranges = BatchUtils.getBatchRanges(data_size, param.getBatchSize());
			} else if (Y instanceof IntegerMatrix) {
				data_size = ((IntegerMatrix) X).sizeOfEntries();
				data_locs = ArrayUtils.range(((IntegerMatrix) X).size());
			}
		}

		double best_acc = 0;
		double best_cost = 0;
		int best_cor_cnt = 0;

		List<Performance> perfs = Generics.newLinkedList();

		for (int i = 1; i <= max_iter; i++) {
			Timer timer2 = Timer.newTimer();

			ArrayUtils.shuffle(data_locs);

			range_cnt = new AtomicInteger(0);

			List<Future<Pair<Double, Integer>>> fs = Generics.newArrayList(ws.size());
			for (int j = 0; j < ws.size(); j++) {
				fs.add(tpe.submit(ws.get(j)));
			}

			double cost = 0;
			int cor_cnt = 0;

			for (int j = 0; j < fs.size(); j++) {
				Pair<Double, Integer> res = fs.get(j).get();
				cost += res.getFirst();
				cor_cnt += res.getSecond();
			}

			if (param.getRegLambda() > 0) {
				cost += CrossEntropyCostFunction.getL2RegularizationTerm(param.getRegLambda(), W, data_size);
			}

			double acc = 1f * cor_cnt / data_size;

			if (best_cost < cost) {
				best_cost = cost;
				best_cor_cnt = cor_cnt;
				best_acc = acc;

				// VectorUtils.copy(W, W_best);
			}

			double norm = ArrayMath.normL2(W.values());
			System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s (%s), norm: %f, learn-rate: %f\n", i, cost,
					acc, cor_cnt, data_size, timer2.stop(), timer1.stop(), norm, param.getLearnRate());

			if (Xt != null && Yt != null) {
				nn.setIsTesting(true);

				IntegerArray yh = null;
				IntegerArray y = null;

				if (X instanceof DenseMatrix) {
					yh = nn.classify(Xt);
					y = (IntegerArray) Yt;
				} else if (X instanceof IntegerMatrix) {
					if (Y instanceof IntegerArray) {
						IntegerMatrix Xm = (IntegerMatrix) Xt;
						IntegerArray Ym = (IntegerArray) Yt;

						y = new IntegerArray(Xm.size());
						yh = new IntegerArray(Xm.size());

						int[][] ranges = BatchUtils.getBatchRanges(Xm.size(), param.getBatchSize());

						for (int j = 0; j < ranges.length; j++) {
							int[] range = ranges[j];
							IntegerMatrix xm = Xm.subMatrix(range[0], range[1]);
							IntegerArray yhm = nn.classify(xm);

							y.addAll(Ym.subArray(range[0], range[1]));
							yh.addAll(yhm);
						}
					} else if (Y instanceof IntegerMatrix) {
						IntegerMatrix Xm = (IntegerMatrix) Xt;
						IntegerMatrix Ym = (IntegerMatrix) Yt;

						y = new IntegerArray(Xm.sizeOfEntries());
						yh = new IntegerArray(Xm.sizeOfEntries());

						for (int j = 0; j < Xm.size(); j++) {
							IntegerArray xm = Xm.get(j);
							IntegerArray ym = Ym.get(j);
							IntegerArray yhm = nn.classify(xm);

							for (int k = 0; k < ym.size(); k++) {
								y.add(ym.get(k));
								yh.add(yhm.get(k));
							}
						}
					}
				}

				Performance p = eval.evalute(y, yh, labelIndexer);
				perfs.add(p);

				System.out.println(p.toString());
				nn.setIsTesting(false);
			}
		}
	}

}
