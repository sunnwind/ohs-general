package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.ml.eval.Performance;
import ohs.ml.eval.PerformanceEvaluator;
import ohs.ml.neuralnet.cost.CrossEntropyCostFunction;
import ohs.ml.neuralnet.layer.BatchNormalizationLayer;
import ohs.ml.neuralnet.layer.BidirectionalRecurrentLayer;
import ohs.ml.neuralnet.layer.DropoutLayer;
import ohs.ml.neuralnet.layer.EmbeddingLayer;
import ohs.ml.neuralnet.layer.FullyConnectedLayer;
import ohs.ml.neuralnet.layer.Layer;
import ohs.ml.neuralnet.layer.LstmLayer;
import ohs.ml.neuralnet.layer.NonlinearityLayer;
import ohs.ml.neuralnet.layer.RnnLayer;
import ohs.ml.neuralnet.layer.SoftmaxLayer;
import ohs.ml.neuralnet.layer.WindowLayer;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
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

		private NeuralNet nn;

		private ParameterUpdater pu;

		private CrossEntropyCostFunction cf;

		public Worker(NeuralNet nn) {
			this.nn = nn;
			pu = new ParameterUpdater(W, nn.getGradients(), rW1, rW2, param.getBatchSize());
			pu.setLearningRate(param.getLearnRate());
			pu.setWeightDecay(param.getRegLambda(), param.getLearnRate(), data_size);
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

					DenseMatrix Xm = X_.rowsAsMatrix(locs);
					IntegerArray Ym = Y_.get(locs);

					DenseMatrix Yh = (DenseMatrix) nn.forward(Xm);
					DenseMatrix D = cf.evaluate(Yh, Ym);

					cost += cf.getCost();
					correct_cnt += cf.getCorrectCnt();

					nn.backward(D);

					pu.update();
				}
			} else {
				int data_loc = 0;
				int batch_size = param.getBatchSize();
				IntegerArrayMatrix Xm = (IntegerArrayMatrix) X;
				IntegerArrayMatrix Ym = (IntegerArrayMatrix) Y;

				IntegerArray xm = new IntegerArray();
				IntegerArray ym = new IntegerArray();

				while ((data_loc = range_cnt.getAndIncrement()) < data_locs.length) {
					int loc = data_locs[data_loc];
					IntegerArray x = Xm.get(loc);
					IntegerArray y = Ym.get(loc);

					int[][] ranges = BatchUtils.getBatchRanges(x.size(), batch_size);

					for (int j = 0; j < ranges.length; j++) {
						xm.clear();
						ym.clear();

						int[] range = ranges[j];

						for (int k = range[0]; k < range[1]; k++) {
							xm.add(x.get(k));
							ym.add(y.get(k));
						}

						DenseMatrix Yh = (DenseMatrix) nn.forward(xm);
						DenseMatrix D = cf.evaluate(Yh, ym);

						cost += cf.getCost();
						correct_cnt += cf.getCorrectCnt();

						nn.backward(D);

						pu.update();

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
				}
			}
		}
	}

	public static NeuralNet copy(NeuralNet nn) {
		NeuralNet ret = new NeuralNet();
		for (Layer l : nn) {
			if (l instanceof FullyConnectedLayer) {
				FullyConnectedLayer n = (FullyConnectedLayer) l;
				ret.add(new FullyConnectedLayer(n.getW(), n.getB().row(0)));
			} else if (l instanceof SoftmaxLayer) {
				SoftmaxLayer n = (SoftmaxLayer) l;
				ret.add(new SoftmaxLayer(n.getOutputSize()));
			} else if (l instanceof DropoutLayer) {
				DropoutLayer n = (DropoutLayer) l;
				ret.add(new DropoutLayer(n.getOutputSize(), n.getP()));
			} else if (l instanceof BatchNormalizationLayer) {
				BatchNormalizationLayer n = (BatchNormalizationLayer) l;
				ret.add(new BatchNormalizationLayer(n.getRunMeans(), n.getRunVars(), n.getGamma(), n.getBeta()));
			} else if (l instanceof NonlinearityLayer) {
				NonlinearityLayer n = (NonlinearityLayer) l;
				ret.add(new NonlinearityLayer(n.getOutputSize(), n.getNonlinearity()));
			} else if (l instanceof EmbeddingLayer) {
				EmbeddingLayer n = (EmbeddingLayer) l;
				ret.add(new EmbeddingLayer(n.getW(), n.isLearnEmbedding()));
			} else if (l instanceof WindowLayer) {
				WindowLayer n = (WindowLayer) l;
				ret.add(new WindowLayer(n.getWindowSize(), n.getEmbeddingSize()));
			} else if (l instanceof RnnLayer) {
				RnnLayer n = (RnnLayer) l;
				ret.add(new RnnLayer(n.getWxh(), n.getWhh(), n.getB().row(0), n.getBpttSize(), n.getNonlinearity()));
			} else if (l instanceof LstmLayer) {
				LstmLayer n = (LstmLayer) l;
				ret.add(new LstmLayer(n.getWxh(), n.getWhh(), n.getB().row(0), n.getNonlinearity()));
			} else if (l instanceof BidirectionalRecurrentLayer) {
				BidirectionalRecurrentLayer n = (BidirectionalRecurrentLayer) l;
				Layer fwd2 = null;
				Layer bwd2 = null;

				if (n.getForwardLayer() instanceof RnnLayer) {
					RnnLayer fwd1 = (RnnLayer) n.getForwardLayer();
					RnnLayer bwd1 = (RnnLayer) n.getBackwardLayer();
					fwd2 = new RnnLayer(fwd1.getWxh(), fwd1.getWhh(), fwd1.getB().row(0), fwd1.getBpttSize(), fwd1.getNonlinearity());
					bwd2 = new RnnLayer(bwd1.getWxh(), bwd1.getWhh(), bwd1.getB().row(0), bwd1.getBpttSize(), bwd1.getNonlinearity());
				} else if (n.getForwardLayer() instanceof LstmLayer) {
					LstmLayer fwd1 = (LstmLayer) n.getForwardLayer();
					LstmLayer bwd1 = (LstmLayer) n.getBackwardLayer();
					fwd2 = new LstmLayer(fwd1.getWxh(), fwd1.getWhh(), fwd1.getB().row(0), fwd1.getNonlinearity());
					bwd2 = new LstmLayer(bwd1.getWxh(), bwd1.getWhh(), bwd1.getB().row(0), bwd1.getNonlinearity());
				}
				ret.add(new BidirectionalRecurrentLayer(fwd2, bwd2));
			} else {
				System.err.println("unknown layer");
				System.exit(0);
			}
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

	public int pad_label = -1;

	private NeuralNet nn;

	private NeuralNetParams param;

	private DenseMatrix W;

	private DenseMatrix rW1;

	private DenseMatrix rW2;

	private DenseMatrix W_best;

	private DenseMatrix W_no_bias;

	private AtomicInteger range_cnt;

	private int data_size;

	private int[] data_locs;

	private int[][] ranges;

	private Object X;

	private Object Y;

	private ThreadPoolExecutor tpe;

	private List<Worker> ws;

	private Indexer<String> labelIndexer;

	private PerformanceEvaluator eval = new PerformanceEvaluator();

	private Timer timer1 = Timer.newTimer();

	public NeuralNetTrainer(NeuralNet nn, NeuralNetParams param, int data_size, Indexer<String> labelIndexer) throws Exception {
		prepare(nn, param, data_size, labelIndexer);
	}

	public void finish() {
		tpe.shutdown();

		nn.setIsTesting(true);

		VectorUtils.copy(W_best, W);
	}

	public void prepare(NeuralNet nn, NeuralNetParams param, int data_size, Indexer<String> labelIndexer) throws Exception {
		this.nn = nn;
		this.param = param;
		this.data_size = data_size;

		int thread_size = param.getThreadSize();
		List<NeuralNet> nns = copy(nn, thread_size - 1);

		for (NeuralNet n : nns) {
			n.prepare();
		}

		nns.add(nn);

		W = nn.getParameters();
		rW1 = W.copy(true);
		rW2 = W.copy(true);

		W_best = W.copy(true);
		W_no_bias = nn.getW();

		tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		ws = Generics.newArrayList(thread_size);

		for (int i = 0; i < thread_size; i++) {
			ws.add(new Worker(nns.get(i)));
		}
	}

	// public void train(NeuralNetParams param, NeuralNet nn, Object X, Object Y, Object Xt, Object Yt, int max_iter,
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
	// ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);
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
	// cost += CrossEntropyCostFunction.getL2RegularizationTerm(param.getRegLambda(), W_no_bias, data_size);
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
	// System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s (%s), norm: %f, learn-rate: %f\n", iter, cost, acc, cor_cnt,
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
			IntegerArrayMatrix X_ = ((IntegerArrayMatrix) X);
			for (IntegerArray x : X_) {
				data_size += x.size();
			}
			data_locs = ArrayUtils.range(X_.size());
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
				cost += CrossEntropyCostFunction.getL2RegularizationTerm(param.getRegLambda(), W_no_bias, data_size);
			}

			double acc = 1f * cor_cnt / data_size;

			if (best_cost < cost) {
				best_cost = cost;
				best_cor_cnt = cor_cnt;
				best_acc = acc;

				VectorUtils.copy(W, W_best);
			}

			double norm = VectorMath.normL2(W_no_bias);
			System.out.printf("%dth, cost: %f, acc: %f (%d/%d), time: %s (%s), norm: %f, learn-rate: %f\n", i, cost, acc, cor_cnt,
					data_size, timer2.stop(), timer1.stop(), norm, param.getLearnRate());

			if (Xt != null && Yt != null) {
				nn.setIsTesting(true);

				IntegerArray yh = null;
				IntegerArray y = null;

				if (X instanceof DenseMatrix) {
					yh = nn.classify(Xt);
					y = (IntegerArray) Yt;
				} else {
					IntegerArrayMatrix Xm = (IntegerArrayMatrix) Xt;
					IntegerArrayMatrix Ym = (IntegerArrayMatrix) Yt;

					y = new IntegerArray(Xm.size());
					yh = new IntegerArray(Xm.size());

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

				Performance p = eval.evalute(y, yh, labelIndexer);
				perfs.add(p);

				System.out.println(p.toString());
				nn.setIsTesting(false);
			}
		}
	}

}
