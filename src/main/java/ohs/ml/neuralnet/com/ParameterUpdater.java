package ohs.ml.neuralnet.com;

import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;

/**
 * 
 * http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
 * 
 * http://cs231n.github.io/neural-networks-2/
 * 
 * http://cs231n.github.io/neural-networks-3/
 * 
 * @author ohs
 */
public class ParameterUpdater {

	public static enum OptimizerType {
		ADAGRAD, ADAM, RMSPROP, SIMPLE
	}

	private int batch_size;

	private double beta1 = 0.9;

	private double beta2 = 0.999;

	private double decay_rate = 0.9;

	private DenseTensor dWs;

	private double eps = 0.00000001;

	private double grad_clip_cutoff = 1000;

	private double learn_rate = 0.001;

	private NeuralNet nn;

	private OptimizerType ot = OptimizerType.ADAM;

	private DenseTensor rWs1;

	private DenseTensor rWs2;

	private boolean use_batch_size_scale = false;

	private double weight_decay = 0.99999;

	private DenseTensor Ws;

	/**
	 * @param W
	 * @param dW
	 * @param rW1
	 * @param rW2
	 * @param batch_size
	 */
	public ParameterUpdater(NeuralNet nn, int batch_size) {
		this.nn = nn;
		this.batch_size = batch_size;

		Ws = nn.getW(false);
		dWs = nn.getDW(false);
		rWs1 = new DenseTensor();
		rWs2 = new DenseTensor();

		rWs1.ensureCapacity(Ws.size());
		rWs2.ensureCapacity(Ws.size());

		for (int i = 0; i < Ws.size(); i++) {
			DenseMatrix W = Ws.get(i);
			DenseMatrix rW1 = W.copy(true);
			DenseMatrix rW2 = W.copy(true);

			rWs1.add(rW1);
			rWs2.add(rW2);
		}

		// locs = ArrayUtils.range(W.rowSize());

		// ArrayUtils.shuffle(locs);
	}

	public NeuralNet getNeuralNet() {
		return nn;
	}

	public void setGradientClippingCutoff(double grad_clip_cutoff) {
		this.grad_clip_cutoff = grad_clip_cutoff;
	}

	public void setLearningRate(double learn_rate) {
		this.learn_rate = learn_rate;
	}

	public void setOptimizerType(OptimizerType ot) {
		this.ot = ot;
	}

	public void setUseBatchSizeScale(boolean use_batch_size_scale) {
		this.use_batch_size_scale = use_batch_size_scale;
	}

	/**
	 * http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization
	 * 
	 * @param reg_lambda
	 * @param learn_rate
	 * @param data_size
	 */
	public void setWeightDecay(double reg_lambda, double learn_rate, long data_size) {
		weight_decay = (1 - reg_lambda * learn_rate / data_size);
	}

	public void update() {
		double batch_size_scale = use_batch_size_scale ? 1d / batch_size : 1;

		for (int i = 0; i < Ws.size(); i++) {
			DenseMatrix W = Ws.get(i);
			DenseMatrix dW = dWs.get(i);
			DenseMatrix rW1 = rWs1.get(i);
			DenseMatrix rW2 = rWs2.get(i);

			if (grad_clip_cutoff != Double.MAX_VALUE) {
				double norm = VectorMath.normL2(dW);
				if (norm > grad_clip_cutoff) {
					dW.multiply(grad_clip_cutoff / norm);
				}
			}

			double sum = 0;
			double x = 0;
			double dx = 0;
			double rv1 = 0;
			double rv2 = 0;

			for (int j = 0; j < W.rowSize(); j++) {
				DenseVector dw = dW.row(j);
				DenseVector w = W.row(j);
				DenseVector r1 = rW1.row(j);
				DenseVector r2 = rW2.row(j);

				// if (g.sum() != 0) {
				synchronized (w) {
					sum = 0;
					if (ot == OptimizerType.SIMPLE) {
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k) * batch_size_scale;
							x = w.value(k) * weight_decay;
							x -= learn_rate * dx;
							w.set(k, x);
							sum += x;
						}
					} else if (ot == OptimizerType.ADAGRAD) {
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k) * batch_size_scale;
							r1.add(k, Math.pow(dx, 2));
							x = w.value(k) * weight_decay;
							x -= learn_rate / Math.sqrt(r1.value(k) + eps) * dx;
							w.set(k, x);
							sum += x;
						}
					} else if (ot == OptimizerType.RMSPROP) {
						sum = 0;
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k) * batch_size_scale;
							rv1 = ArrayMath.addAfterMultiply(r1.value(k), decay_rate, Math.pow(dx, 2));
							r1.set(k, rv1);

							x = w.value(k) * weight_decay;
							x -= -learn_rate / Math.sqrt(rv1 + eps) * dx;
							w.set(k, x);
							sum += x;
						}
					} else if (ot == OptimizerType.ADAM) {
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k) * batch_size_scale;
							rv1 = ArrayMath.addAfterMultiply(r1.value(k), beta1, dx);
							rv2 = ArrayMath.addAfterMultiply(r2.value(k), beta2, Math.pow(dx, 2));
							r1.set(k, rv1);
							r2.set(k, rv2);

							rv1 = rv1 / (1 - beta1);
							rv2 = rv2 / (1 - beta2);

							x = w.value(k) * weight_decay;
							x -= learn_rate / Math.sqrt(rv2 + eps) * rv1;
							w.set(k, x);
							sum += x;

							// m.add(j, -learn_rate / Math.sqrt(rrv + eps) * rv);
						}
					}

					w.setSum(sum);
					// }
					dw.setAll(0);
				}
			}
		}
	}

}
