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

			while ((range_loc = range_cnt.getAndIncrement()) < ranges.length) {
				int[] r = ranges[range_loc];
				int[] locs = BatchUtils.getIndexes(data_locs, r);

				DenseTensor Xm = X.subTensor(locs);
				DenseMatrix Ym = Y.subMatrix(locs);
				DenseTensor Yhm = (DenseTensor) nn.forward(Xm);
				DenseTensor D = cf.evaluate(Yhm, Ym);
				cost += cf.getCost();
				correct_cnt += cf.getCorrectCount();
				nn.backward(D);
				pu.update();
			}

			return Generics.newPair(cost, correct_cnt);
		}

	}

	public static NeuralNet copy(NeuralNet nn) {
		NeuralNet ret = new NeuralNet();
		for (Layer l : nn) {
			ret.add(l.copy());
		}
		return ret;
	}

	private int batch_size;

	private int mini_batch_size;

	private int[] data_locs;

	private PerformanceEvaluator eval = new PerformanceEvaluator();

	private int grad_acc_reset_size;

	private boolean is_full_seq_batch;

	private boolean is_random_batch;

	private double learn_rate;

	private double learn_rate_decay;

	private int learn_rate_decay_size;

	private NeuralNetMultiRunner nnmr;

	private List<NeuralNet> nns;

	private List<ParameterUpdater> pus;

	private AtomicInteger range_cnt;

	private int[][] ranges;

	private double reg_lambda;

	private Timer timer1 = Timer.newTimer();

	private int total_iters = 0;

	private ThreadPoolExecutor tpe;

	private TaskType tt;

	private DenseMatrix W;

	private double weight_decay;

	private List<Worker> ws;

	private DenseTensor X;

	private DenseMatrix Y;

	public NeuralNetTrainer(NeuralNet nn, NeuralNetParams nnp) throws Exception {
		prepare(nn, nnp.getThreadSize(), nnp.getBatchSize(), nnp.getLearnRate(), nnp.getRegLambda(),
				nnp.getGradientClipCutoff(), nnp.getOptimizerType(), nnp.isFullSequenceBatch(), nnp.isRandomBatch(),
				nnp.getGradientAccumulatorResetSize(), nnp.getWeightDecayL2(), nnp.getLearnRateDecay(),
				nnp.getLearnRateDecaySize());
	}

	public Performance evaluate(DenseTensor X, DenseMatrix Y) throws Exception {
		for (NeuralNet nn : nns) {
			nn.setIsTesting(true);
		}

		NeuralNet n = nns.get(0);
		DenseMatrix Yh = nnmr.classify(X);

		IntegerMatrix _Y = new IntegerMatrix(Y.rowSize());
		IntegerMatrix _Yh = new IntegerMatrix(Y.rowSize());

		for (int i = 0; i < Y.rowSize(); i++) {
			DenseVector y = Y.row(i);
			DenseVector yh = Yh.row(i);

			IntegerArray anss = new IntegerArray(y.size());
			IntegerArray preds = new IntegerArray(y.size());

			for (int j = 0; j < y.size(); j++) {
				anss.add((int) y.value(j));
				preds.add((int) yh.value(j));
			}

			_Y.add(anss);
			_Yh.add(preds);
		}

		Performance p = null;

		if (Y.colSize() > 1) {
			p = eval.evaluteSequences(_Y, _Yh, n.getLabelIndexer());
		} else {
			p = eval.evalute(_Y, _Yh, n.getLabelIndexer());
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
	}

	private void prepare(NeuralNet nn, int thread_size, int batch_size, double learn_rate, double reg_lambda,
			double grad_clip_cutoff, OptimizerType ot, boolean is_full_seq_batch, boolean is_random_batch,
			int grad_acc_reset_size, double weight_decay, double learn_rate_decay, int learn_rate_decay_size)
			throws Exception {
		this.learn_rate = learn_rate;
		this.reg_lambda = reg_lambda;
		this.batch_size = batch_size;
		this.is_full_seq_batch = is_full_seq_batch;
		this.is_random_batch = is_random_batch;
		this.grad_acc_reset_size = grad_acc_reset_size;
		this.weight_decay = weight_decay;
		this.learn_rate_decay = learn_rate_decay;
		this.learn_rate_decay_size = learn_rate_decay_size;

		this.tt = nn.getTaskType();

		nns = Generics.newArrayList(thread_size);
		nns.add(nn);

		for (int i = 0; i < thread_size - 1; i++) {
			NeuralNet n = copy(nn);
			n.prepare();

			nns.add(n);
		}

		pus = Generics.newArrayList(thread_size);

		for (NeuralNet n : nns) {
			ParameterUpdater pu = new ParameterUpdater(n);
			pu.setLearningRate(learn_rate);
			// pu.setWeightDecay(reg_lambda, learn_rate, tmp_data_size);
			pu.setOptimizerType(ot);
			pu.setGradientClipCutoff(grad_clip_cutoff);
			pu.setWeightDecay(weight_decay);
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

		nnmr = new NeuralNetMultiRunner(nns);
	}

	public void train(DenseTensor X, DenseMatrix Y, DenseTensor Xt, DenseMatrix Yt, int max_iters) throws Exception {

		if (Y.colSize() > 0) {

			if (is_full_seq_batch) {
				data_locs = ArrayUtils.range(Y.sizeOfEntries());
				ranges = BatchUtils.getBatchRanges(Y.rowSize(), batch_size);
			} else {
				List<DenseMatrix> _X = Generics.newLinkedList();
				List<DenseVector> _Y = Generics.newLinkedList();

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

		double best_acc = 0;
		double best_cost = 0;
		int best_cor_cnt = 0;

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
				cost += CrossEntropyCostFunction.getL2RegularizationTerm(reg_lambda, W, data_size);
			}

			double acc = 1d * cor_cnt / data_size;

			if (best_cost < cost) {
				best_cost = cost;
				best_cor_cnt = cor_cnt;
				best_acc = acc;
			}

			double norm = ArrayMath.normL2(W.values());

			System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s (%s), norm: %f, learn-rate: %f\n", i + 1, cost,
					acc, cor_cnt, data_size, timer2.stop(), timer1.stop(), norm, learn_rate);

			if (Xt != null && Yt != null) {
				Performance p = evaluate(Xt, Yt);
				System.out.println(p.toString());
			}

			if (iters % grad_acc_reset_size == 0) {
				for (ParameterUpdater pu : pus) {
					pu.resetGradientAccumulators();
				}
			}

			if (iters % learn_rate_decay_size == 0) {
				learn_rate = learn_rate * learn_rate_decay;

				for (ParameterUpdater pu : pus) {
					pu.setLearningRate(learn_rate);
				}
			}

		}
	}

}