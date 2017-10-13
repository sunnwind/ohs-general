package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.eval.Performance;
import ohs.ml.eval.PerformanceEvaluator;
import ohs.ml.neuralnet.cost.CrossEntropyCostFunction;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
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
					int[] r = ranges[range_loc];
					int[] locs = BatchUtils.getIndexes(data_locs, r);

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
						int[] r = ranges[range_loc];
						int[] locs = BatchUtils.getIndexes(data_locs, r);

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

						if (batch_size == Integer.MAX_VALUE) {
							DenseMatrix Yh = (DenseMatrix) nn.forward(x);
							DenseMatrix D = cf.evaluate(Yh, y);

							cost += cf.getCost();
							correct_cnt += cf.getCorrectCnt();

							nn.backward(D);
							pu.update();
						} else {
							int[][] rs = BatchUtils.getBatchRanges(x.size(), batch_size);
							for (int i = 0; i < rs.length; i++) {
								int[] r = rs[i];
								IntegerArray Xm = x.subArray(r[0], r[1]);
								IntegerArray Ym = y.subArray(r[0], r[1]);

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

	private int burnin_iters = 100;

	private NeuralNet nn;

	private NeuralNetParams param;

	private List<ParameterUpdater> pus;

	private AtomicInteger range_cnt;

	private int[][] ranges;

	private Timer timer1 = Timer.newTimer();

	private int total_iters = 0;

	private ThreadPoolExecutor tpe;

	private DenseMatrix W;

	private List<Worker> ws;

	private Object X;

	private Object Y;

	public NeuralNetTrainer(NeuralNet nn, NeuralNetParams param, int data_size) throws Exception {
		prepare(nn, param, data_size);
	}

	public void finish() {
		tpe.shutdown();

		nn.setIsTesting(true);

		// VectorUtils.copy(W_best, W);
	}

	private void prepare(NeuralNet nn, NeuralNetParams param, int data_size) throws Exception {
		this.nn = nn;
		this.param = param;
		this.data_size = data_size;

		int thread_size = param.getThreadSize();
		List<NeuralNet> nns = copy(nn, thread_size - 1);

		for (NeuralNet n : nns) {
			n.prepare();
		}

		nns.add(nn);

		pus = Generics.newArrayList(nns.size());

		for (NeuralNet n : nns) {
			ParameterUpdater pu = new ParameterUpdater(n, data_size);
			pu.setLearningRate(param.getLearnRate());
			pu.setWeightDecay(param.getRegLambda(), param.getLearnRate(), data_size);
			pu.setOptimizerType(param.getOptimizerType());
			pu.setGradientClipCutoff(param.getGradientClipCutoff());
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

	public void train(Object X, Object Y, Object Xt, Object Yt, int max_iters) throws Exception {
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

		int s = total_iters;
		int e = total_iters + max_iters;
		int cnt = 0;

		for (int i = s; i < e; i++) {
			total_iters++;
			cnt++;

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
			System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s (%s), norm: %f, learn-rate: %f\n", i + 1, cost,
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

						int[][] rs = BatchUtils.getBatchRanges(Xm.size(), param.getBatchSize());

						for (int j = 0; j < rs.length; j++) {
							int[] r = rs[j];
							IntegerMatrix xm = Xm.subMatrix(r[0], r[1]);
							IntegerArray yhm = nn.classify(xm);

							y.addAll(Ym.subArray(r[0], r[1]));
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

				Performance p = eval.evalute(y, yh, nn.getLabelIndexer());
				perfs.add(p);

				System.out.println(p.toString());
				nn.setIsTesting(false);

				if (cnt == burnin_iters) {
					for (ParameterUpdater pu : pus) {
						pu.resetGradientAccumulators();
					}
				}
			}
		}
	}

}
