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
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.cost.CrossEntropyCostFunction;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.RecurrentLayer;
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
					IntegerMatrix X_ = (IntegerMatrix) X;
					IntegerMatrix Y_ = (IntegerMatrix) Y;

					int range_loc = 0;

					while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
						int[] r = ranges[range_loc];
						int[] locs = BatchUtils.getIndexes(data_locs, r);

						IntegerMatrix Xm = X_.subMatrix(locs);
						IntegerMatrix Ym = Y_.subMatrix(locs);

						if (is_full_seq_batch) {
							for (int i = 0; i < Xm.size(); i++) {
								IntegerArray xm = Xm.get(i);
								IntegerArray ym = Ym.get(i);
								DenseMatrix ymh = (DenseMatrix) nn.forward(xm);
								DenseMatrix D = cf.evaluate(ymh, ym);

								cost += cf.getCost();
								correct_cnt += cf.getCorrectCnt();
								nn.backward(D);
							}
							pu.update();
						} else {
							for (int i = 0; i < Xm.size(); i++) {
								IntegerArray xm = Xm.get(i);
								IntegerArray ym = Ym.get(i);
								int[][] rs = BatchUtils.getBatchRanges(xm.size(), batch_size);

								for (int j = 0; j < rs.length; j++) {
									int[] r2 = rs[j];
									IntegerArray x = xm.subArray(r2[0], r2[1]);
									IntegerArray y = ym.subArray(r2[0], r2[1]);

									DenseMatrix Yh = (DenseMatrix) nn.forward(x);
									DenseMatrix D = cf.evaluate(Yh, y);

									cost += cf.getCost();
									correct_cnt += cf.getCorrectCnt();

									nn.backward(D);
									pu.update();
								}
							}
						}
						readyForNextIteration();
					}
				}
			}

			// readyForNextIteration();

			return Generics.newPair(cost, correct_cnt);
		}

		private void readyForNextIteration() {
			for (Layer l : nn) {
				if (l instanceof RecurrentLayer) {
					RecurrentLayer n = (RecurrentLayer) l;
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

	private int batch_size;

	private int burnin_iters = 100;

	private int[] data_locs;

	private PerformanceEvaluator eval = new PerformanceEvaluator();

	// private NeuralNetParams param;

	private double learn_rate;

	private NeuralNet nn;

	private List<ParameterUpdater> pus;

	private AtomicInteger range_cnt;

	private int[][] ranges;

	private double reg_lambda;

	private Timer timer1 = Timer.newTimer();

	private int total_iters = 0;

	private ThreadPoolExecutor tpe;

	private DenseMatrix W;

	private List<Worker> ws;

	private Object X;

	private Object Y;

	public NeuralNetTrainer(NeuralNet nn, NeuralNetParams param) throws Exception {
		prepare(nn, param.getThreadSize(), param.getBatchSize(), param.getLearnRate(), param.getRegLambda(),
				param.getGradientClipCutoff(), param.getOptimizerType(), param.isFullSequenceBatch());
	}

	public void finish() {
		tpe.shutdown();

		nn.setIsTesting(true);

		// VectorUtils.copy(W_best, W);
	}

	private void prepare(NeuralNet nn, int thread_size, int batch_size, double learn_rate, double reg_lambda,
			double grad_clip_cutoff, OptimizerType ot, boolean is_sent_batch) throws Exception {
		this.nn = nn;

		this.learn_rate = learn_rate;
		this.reg_lambda = reg_lambda;
		this.batch_size = batch_size;
		this.is_full_seq_batch = is_sent_batch;

		List<NeuralNet> nns = copy(nn, thread_size - 1);

		for (NeuralNet n : nns) {
			n.prepare();
		}

		nns.add(nn);

		pus = Generics.newArrayList(nns.size());

		int tmp_data_size = 1000000;

		for (NeuralNet n : nns) {
			ParameterUpdater pu = new ParameterUpdater(n, tmp_data_size);
			pu.setLearningRate(learn_rate);
			pu.setWeightDecay(reg_lambda, learn_rate, tmp_data_size);
			pu.setOptimizerType(ot);
			pu.setGradientClipCutoff(grad_clip_cutoff);
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

	private boolean is_full_seq_batch;

	public void train(Object X, Object Y, Object Xt, Object Yt, int max_iters) throws Exception {
		this.X = X;
		this.Y = Y;

		int data_size = 0;

		if (X instanceof DenseMatrix) {
			data_size = ((DenseMatrix) X).rowSize();
			data_locs = ArrayUtils.range(data_size);
			ranges = BatchUtils.getBatchRanges(data_size, batch_size);
		} else if (X instanceof IntegerMatrix) {
			if (Y instanceof IntegerArray) {
				data_size = ((IntegerMatrix) Y).size();
				data_locs = ArrayUtils.range(data_size);
				ranges = BatchUtils.getBatchRanges(data_size, batch_size);
			} else if (Y instanceof IntegerMatrix) {
				IntegerMatrix Y_ = ((IntegerMatrix) Y);
				data_size = Y_.sizeOfEntries();
				data_locs = ArrayUtils.range(Y_.size());
				ranges = BatchUtils.getBatchRanges(Y_.size(), 100);
			}
		}

		double best_acc = 0;
		double best_cost = 0;
		int best_cor_cnt = 0;

		List<Performance> perfs = Generics.newLinkedList();

		int s = total_iters;
		int e = total_iters + max_iters;
		int iters = 0;

		for (int i = s; i < e; i++) {
			total_iters++;
			iters++;

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

			if (reg_lambda > 0) {
				cost += CrossEntropyCostFunction.getL2RegularizationTerm(reg_lambda, W, data_size);
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
					acc, cor_cnt, data_size, timer2.stop(), timer1.stop(), norm, learn_rate);

			if (Xt != null && Yt != null) {
				nn.setIsTesting(true);

				IntegerArray yh = null;
				IntegerArray y = null;

				if (X instanceof DenseMatrix) {
					yh = nn.classify(Xt);
					y = (IntegerArray) Yt;
				} else if (X instanceof IntegerMatrix) {
					if (Y instanceof IntegerArray) {
						IntegerMatrix Xt_ = (IntegerMatrix) Xt;
						IntegerArray Yt_ = (IntegerArray) Yt;

						y = new IntegerArray(Xt_.size());
						yh = new IntegerArray(Xt_.size());

						int[][] rs = BatchUtils.getBatchRanges(Xt_.size(), batch_size);

						for (int j = 0; j < rs.length; j++) {
							int[] r = rs[j];
							IntegerMatrix xm = Xt_.subMatrix(r[0], r[1]);
							IntegerArray yhm = nn.classify(xm);

							y.addAll(Yt_.subArray(r[0], r[1]));
							yh.addAll(yhm);
						}
					} else if (Y instanceof IntegerMatrix) {
						IntegerMatrix Xt_ = (IntegerMatrix) Xt;
						IntegerMatrix Yt_ = (IntegerMatrix) Yt;

						y = new IntegerArray(Xt_.sizeOfEntries());
						yh = new IntegerArray(Xt_.sizeOfEntries());

						for (int j = 0; j < Xt_.size(); j++) {
							IntegerArray Xtm = Xt_.get(j);
							IntegerArray Ytm = Yt_.get(j);
							IntegerArray Yhm = nn.classify(Xtm);

							for (int k = 0; k < Ytm.size(); k++) {
								y.add(Ytm.get(k));
								yh.add(Yhm.get(k));
							}
						}
					}
				}

				Performance p = eval.evalute(y, yh, nn.getLabelIndexer());
				perfs.add(p);

				System.out.println(p.toString());
				nn.setIsTesting(false);

				if (iters % burnin_iters == 0) {
					for (ParameterUpdater pu : pus) {
						pu.resetGradientAccumulators();
					}
				}
			}
		}
	}

}
