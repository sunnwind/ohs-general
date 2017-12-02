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
import ohs.ml.neuralnet.com.ParameterUpdater.OptimizerType;
import ohs.ml.neuralnet.cost.CrossEntropyCostFunction;
import ohs.ml.neuralnet.layer.Layer;
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
			int range_loc = 0;
			int[][] rs = ranges;

			while ((range_loc = range_cnt.getAndIncrement()) < rs.length) {
				int[] r = rs[range_loc];
				int[] locs = BatchUtils.getIndexes(data_locs, r);

				DenseTensor Xm = X.subTensor(locs);
				DenseMatrix Ym = Y.subMatrix(locs);

				DenseTensor Yhm = (DenseTensor) nn.forward(Xm);
				DenseTensor D = cf.evaluate(Yhm, Ym);
				cost += cf.getCost();
				correct_cnt += cf.getCorrectCount();
				nn.backward(D);
				pu.update(Ym.sizeOfEntries());
			}

			return Generics.newPair(cost, correct_cnt);
		}

	}

	private int batch_size;

	private int mini_batch_size;

	private int[] data_locs;

	private double learn_rate;

	private double learn_rate_decay;

	private double reg_lambda;

	private int grad_acc_reset_size;

	private int learn_rate_decay_size;

	private int[][] ranges;

	private int total_iters = 0;

	private boolean is_full_seq_batch;

	private boolean is_random_batch;

	private PerformanceEvaluator eval = new PerformanceEvaluator();

	private Timer timer1 = Timer.newTimer();

	private NeuralNetMultiRunner nnmr;

	private NeuralNet onn;

	private List<NeuralNet> nns;

	private List<ParameterUpdater> pus;

	private List<Worker> ws;

	private AtomicInteger range_cnt;

	private ThreadPoolExecutor tpe;

	private TaskType tt;

	private DenseTensor W;

	private DenseTensor Wbest;

	private DenseMatrix Wnobias;

	private DenseTensor X;

	private DenseMatrix Y;

	private double best_score = 0;

	private boolean copy_best_model = true;

	private int eval_window_size = 1;

	public NeuralNetTrainer(NeuralNet nn, NeuralNetParams nnp) throws Exception {
		prepare(nn, nnp.getThreadSize(), nnp.getBatchSize(), nnp.getLearnRate(), nnp.getRegLambda(),
				nnp.getGradientClipCutoff(), nnp.getOptimizerType(), nnp.isFullSequenceBatch(), nnp.isRandomBatch(),
				nnp.getGradientAccumulatorResetSize(), nnp.getWeightDecayL2(), nnp.getLearnRateDecay(),
				nnp.getLearnRateDecaySize(), nnp.getGradientDecay(), nnp.getUseAverageGradients(),
				nnp.getUseHardGradientClipping());
	}

	public Performance evaluate(DenseTensor X, DenseMatrix Y) throws Exception {
		for (NeuralNet nn : nns) {
			nn.setIsTesting(true);
		}

		DenseMatrix Yh = nnmr.classify(X);

		Performance p = null;

		if (Y.colSize() > 1) {
			p = eval.evaluteSequences(Y, Yh, onn.getLabelIndexer());
		} else {
			p = eval.evalute(Y, Yh, onn.getLabelIndexer());
		}

		for (NeuralNet nn : nns) {
			nn.setIsTesting(false);
		}

		return p;
	}

	public void finish() {
		tpe.shutdown();

		for (NeuralNet nn : nns) {
			nn.setIsTesting(true);
		}

		nnmr.shutdown();

		if (copy_best_model) {
			VectorUtils.copy(Wbest, W);
		}
	}

	private void prepare(NeuralNet nn, int thread_size, int batch_size, double learn_rate, double reg_lambda,
			double grad_clip_cutoff, OptimizerType ot, boolean is_full_seq_batch, boolean is_random_batch,
			int grad_acc_reset_size, double weight_decay, double learn_rate_decay, int learn_rate_decay_size,
			double grad_decay, boolean use_avg_grad, boolean use_hard_grad_clipping) throws Exception {
		this.onn = nn;
		this.learn_rate = learn_rate;
		this.reg_lambda = reg_lambda;
		this.batch_size = batch_size;
		this.is_full_seq_batch = is_full_seq_batch;
		this.is_random_batch = is_random_batch;
		this.grad_acc_reset_size = grad_acc_reset_size;
		this.learn_rate_decay = learn_rate_decay;
		this.learn_rate_decay_size = learn_rate_decay_size;

		this.tt = nn.getTaskType();

		nns = Generics.newArrayList(thread_size);
		nns.add(nn);

		for (int i = 0; i < thread_size - 1; i++) {
			NeuralNet n = nn.copy();
			n.prepareTraining();

			nns.add(n);
		}

		pus = Generics.newArrayList(thread_size);

		for (NeuralNet n : nns) {
			ParameterUpdater pu = new ParameterUpdater(n);
			pu.setLearningRate(learn_rate);
			// pu.setWeightDecay(reg_lambda, learn_rate, tmp_data_size);
			pu.setOptimizerType(ot);
			pu.setGradientClipCutoff(grad_clip_cutoff);
			pu.setL2WeightDecay(weight_decay);
			pu.setGradientDecay(grad_decay);
			pu.setUseAverageGradients(use_avg_grad);
			pu.setUseHardGradClipping(use_hard_grad_clipping);
			pus.add(pu);
		}

		{
			List<DenseVector> l = Generics.newLinkedList();

			for (DenseMatrix w : nn.getW(true)) {
				l.addAll(w);
			}
			Wnobias = new DenseMatrix(l);
		}

		Wbest = nn.getW(false).copy(true);

		W = nn.getW(false);

		tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		ws = Generics.newArrayList(thread_size);

		for (int i = 0; i < thread_size; i++) {
			ws.add(new Worker(pus.get(i)));
		}

		nnmr = new NeuralNetMultiRunner(nns);
	}

	public void setCopyBestModel(boolean copy_best_model) {
		this.copy_best_model = copy_best_model;
	}

	public void setEvaluationWindowSize(int eval_window_size) {
		this.eval_window_size = eval_window_size;
	}

	public void train(DenseTensor X, DenseMatrix Y, DenseTensor Xt, DenseMatrix Yt, int max_iters) throws Exception {
		if (Y.colSize() > 1) {
			if (is_full_seq_batch) {
				data_locs = ArrayUtils.range(Y.rowSize());
				ranges = BatchUtils.getBatchRanges(Y.rowSize(), batch_size);
			} else {
				List<DenseMatrix> _X = Generics.newArrayList();
				List<DenseVector> _Y = Generics.newArrayList();

				for (int i = 0; i < X.size(); i++) {
					DenseMatrix Xm = X.get(i);
					DenseVector Ym = Y.get(i);

					for (int j = 0; j < Xm.rowSize();) {
						int size = batch_size;

						if (Xm.rowSize() - j < batch_size) {
							size = Xm.rowSize() - j;
						}

						_X.add(Xm.subMatrix(j, size));
						_Y.add(Ym.subVector(j, size));
						j += size;
					}
				}

				X = new DenseTensor(_X);
				Y = new DenseMatrix(_Y);

				data_locs = ArrayUtils.range(Y.rowSize());
				ranges = BatchUtils.getBatchRanges(Y.rowSize(), 1);
			}
		} else {
			data_locs = ArrayUtils.range(Y.rowSize());
			ranges = BatchUtils.getBatchRanges(Y.rowSize(), batch_size);
		}

		this.X = X;
		this.Y = Y;

		int data_size = 0;
		if (Y.colSize() > 1) {
			/*
			 * sequence labeling
			 */
			data_size = Y.sizeOfEntries();
		} else {
			data_size = Y.size();
		}

		int s = total_iters;
		int e = total_iters + max_iters;
		int iters = 0;

		for (int i = s; i < e; i++) {
			total_iters++;
			iters++;

			Timer timer2 = Timer.newTimer();

			if (is_random_batch) {
				ArrayUtils.shuffle(data_locs);
			}

			range_cnt = new AtomicInteger(0);

			List<Future<Pair<Double, Integer>>> fs = Generics.newArrayList(ws.size());

			for (Worker w : ws) {
				fs.add(tpe.submit(w));
			}

			double cost = 0;
			int cor_cnt = 0;

			for (Future<Pair<Double, Integer>> f : fs) {
				Pair<Double, Integer> res = f.get();
				cost += res.getFirst();
				cor_cnt += res.getSecond();
			}

			if (reg_lambda > 0) {
				cost += CrossEntropyCostFunction.getL2RegularizationTerm(reg_lambda, Wnobias, data_size);
			}

			double acc = 1d * cor_cnt / data_size;

			double norm = ArrayMath.normL2(W.values());

			System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s (%s), norm: %f, learn-rate: %f\n", i + 1, cost,
					acc, cor_cnt, data_size, timer2.stop(), timer1.stop(), norm, learn_rate);

			{
				Performance p = null;

				if (Xt != null && Yt != null && total_iters % eval_window_size == 0) {
					Timer timer3 = Timer.newTimer();
					p = evaluate(Xt, Yt);
					System.out.println(p.toString());
					System.out.printf("testing time:\t%s\n", timer3.stop());
				}

				boolean is_better = false;
				double ratio = 1;

				if (p == null) {
					double score = cost;
					if (score < best_score) {
						best_score = score;
						is_better = true;
					}

					ratio = (score / best_score);

				} else {
					double score = p.getMicroF1();
					if (score > best_score) {
						best_score = score;
						is_better = true;
					}
					ratio = best_score / score;
				}

				String status = "old";

				if (is_better) {
					status = "new";
					VectorUtils.copy(W, Wbest);
				} else {
//					if (ratio < 0.9) {
//						VectorUtils.copy(Wbest, W);
//					}
				}

				System.out.printf("best score:\t[%f, %s]\n", best_score, status);
			}

			if (iters % grad_acc_reset_size == 0) {
				for (ParameterUpdater pu : pus) {
					pu.resetGradientAccumulators();
				}
			}

			if (total_iters % learn_rate_decay_size == 0 && learn_rate_decay < 1.0) {
				learn_rate *= learn_rate_decay;

				for (ParameterUpdater pu : pus) {
					pu.setLearningRate(learn_rate);
				}
			}
		}
	}

}