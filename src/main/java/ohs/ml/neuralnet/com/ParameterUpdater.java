package ohs.ml.neuralnet.com;

import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
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

	public static enum Type {
		SIMPLE, ADAGRAD, RMSPROP, ADAM
	}

	/**
	 * Model
	 */
	private DenseMatrix W;

	/**
	 * Accumulator
	 */
	private DenseMatrix rW1;

	private DenseMatrix rW2;

	/**
	 * Gradients
	 */
	private DenseMatrix dW;

	private double learn_rate = 0.001;

	private double eps = 0.00000001;

	private double grad_clip_threshold = 1;

	private int batch_size;

	private Type type = Type.ADAM;

	private double decay_rate = 0.9;

	private double beta1 = 0.9;

	private double beta2 = 0.999;

	private int[] locs;

	private boolean use_batch_size_scale = false;

	private double weight_decay = 0.99999;

	/**
	 * @param W
	 * @param dW
	 * @param rW1
	 * @param rW2
	 * @param batch_size
	 */
	public ParameterUpdater(DenseMatrix W, DenseMatrix dW, DenseMatrix rW1, DenseMatrix rW2, int batch_size) {
		this.W = W;
		this.dW = dW;
		this.rW1 = rW1;
		this.rW2 = rW2;
		this.batch_size = batch_size;

		locs = ArrayUtils.range(W.rowSize());

		ArrayUtils.shuffle(locs);
	}

	public void setGradientClippingThreshold(double grad_clip_threshold) {
		this.grad_clip_threshold = grad_clip_threshold;
	}

	public void setLearningRate(double learn_rate) {
		this.learn_rate = learn_rate;
	}

	public void setType(Type type) {
		this.type = type;
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

	private void doWeightNormalization(DenseVector dLw, DenseVector v, DenseVector dLg) {
		double norm_v = VectorMath.normL2(v);

		VectorMath.multiply(dLw, v, dLg);
		dLg.multiply(1f / norm_v);
	}

	private void doWeightNormalization2(DenseVector w, DenseVector v) {
		double g = VectorMath.normL2(w);
		double norm_v = VectorMath.normL2(v);

		w = v;
		w.multiply(g / norm_v);
	}

	public void update() {
		double norm = VectorMath.normL2(dW);

		if (norm > grad_clip_threshold) {
			dW.multiply(grad_clip_threshold / norm);
		}

		double sum = 0;
		double x = 0;
		double dx = 0;
		double rv1 = 0;
		double rv2 = 0;
		double batch_size_scale = use_batch_size_scale ? 1d / batch_size : 1;

		for (int i = 0; i < locs.length; i++) {
			int loc = locs[i];
			DenseVector dw = dW.row(loc);
			DenseVector w = W.row(loc);
			DenseVector r1 = rW1.row(loc);
			DenseVector r2 = rW2.row(loc);

			// if (g.sum() != 0) {
			synchronized (w) {
				sum = 0;
				if (type == Type.SIMPLE) {
					for (int j = 0; j < dw.size(); j++) {
						dx = dw.value(j) * batch_size_scale;
						x = w.value(j) * weight_decay;
						x -= learn_rate * dx;
						w.set(j, x);
						sum += x;
					}
				} else if (type == Type.ADAGRAD) {
					for (int j = 0; j < dw.size(); j++) {
						dx = dw.value(j) * batch_size_scale;
						r1.add(j, Math.pow(dx, 2));
						x = w.value(j) * weight_decay;
						x -= learn_rate / Math.sqrt(r1.value(j) + eps) * dx;
						w.set(j, x);
						sum += x;
					}
				} else if (type == Type.RMSPROP) {
					sum = 0;
					for (int j = 0; j < dw.size(); j++) {
						dx = dw.value(j) * batch_size_scale;
						rv1 = ArrayMath.addAfterMultiply(r1.value(j), decay_rate, Math.pow(dx, 2));
						r1.set(j, rv1);

						x = w.value(j) * weight_decay;
						x -= -learn_rate / Math.sqrt(rv1 + eps) * dx;
						w.set(j, x);
						sum += x;
					}
				} else if (type == Type.ADAM) {
					for (int j = 0; j < dw.size(); j++) {
						dx = dw.value(j) * batch_size_scale;
						rv1 = ArrayMath.addAfterMultiply(r1.value(j), beta1, dx);
						rv2 = ArrayMath.addAfterMultiply(r2.value(j), beta2, Math.pow(dx, 2));
						r1.set(j, rv1);
						r2.set(j, rv2);

						rv1 = rv1 / (1 - beta1);
						rv2 = rv2 / (1 - beta2);

						x = w.value(j) * weight_decay;
						x -= learn_rate / Math.sqrt(rv2 + eps) * rv1;
						w.set(j, x);
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
