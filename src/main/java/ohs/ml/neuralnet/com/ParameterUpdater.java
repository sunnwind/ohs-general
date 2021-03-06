package ohs.ml.neuralnet.com;

import java.util.List;

import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.utils.Generics;

/**
 * 
 * http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
 * 
 * http://ruder.io/optimizing-gradient-descent/index.html#adamax
 * 
 * http://cs231n.github.io/neural-networks-2/
 * 
 * http://cs231n.github.io/neural-networks-3/
 * 
 * An overview of gradient descent optimization algorithms
 * 
 * @author ohs
 */
public class ParameterUpdater {

	public static enum OptimizerType {
		SIMPLE, ADAGRAD, RMSPROP, ADAM, ADAMAX, NADAM
	}

	private static double t = 0;

	private OptimizerType ot = OptimizerType.ADAM;

	private double beta1 = 0.9;

	private double beta2 = 0.999;

	private double decay_rate = 0.9;

	private double eps = 0.00000001;

	/**
	 * Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C.
	 * (2016). Neural Architectures for Named Entity Recognition. Arxiv, 1��10.
	 * Retrieved from http://arxiv.org/abs/1603.01360
	 */
	private double grad_clip_cutoff = 5;

	private double learn_rate = 0.001;

	private double l2_weight_decay = 1;

	private double grad_decay = 1;

	private boolean use_hard_grad_clipping = false;

	private boolean use_avg_grad = false;

	private NeuralNet nn;

	private DenseTensor Ws;

	private DenseTensor dWs;

	private DenseTensor Rs1;

	private DenseTensor Rs2;

	public ParameterUpdater(DenseTensor Ws, DenseTensor dWs) {
		this.Ws = Ws;
		this.dWs = dWs;

		Rs1 = new DenseTensor();
		Rs2 = new DenseTensor();

		Rs1.ensureCapacity(Ws.size());
		Rs2.ensureCapacity(Ws.size());

		for (int i = 0; i < Ws.size(); i++) {
			DenseMatrix W = Ws.get(i);
			DenseMatrix rW1 = W.copy(true);
			DenseMatrix rW2 = W.copy(true);

			Rs1.add(rW1);
			Rs2.add(rW2);
		}
	}

	public ParameterUpdater(DenseTensor Ws, DenseTensor dWs, DenseTensor Rs1, DenseTensor Rs2) {
		this.Ws = Ws;
		this.dWs = dWs;
		this.Rs1 = Rs1;
		this.Rs2 = Rs2;
	}

	public ParameterUpdater(NeuralNet nn) {
		this.nn = nn;

		Ws = nn.getW(false);
		dWs = nn.getDW(false);
		Rs1 = new DenseTensor();
		Rs2 = new DenseTensor();

		Rs1.ensureCapacity(Ws.size());
		Rs2.ensureCapacity(Ws.size());

		for (int i = 0; i < Ws.size(); i++) {
			DenseMatrix W = Ws.get(i);
			DenseMatrix rW1 = W.copy(true);
			DenseMatrix rW2 = W.copy(true);

			Rs1.add(rW1);
			Rs2.add(rW2);
		}
	}

	public List<DenseTensor> getGradientAccumulators() {
		List<DenseTensor> ret = Generics.newArrayList(2);
		ret.add(Rs1);
		ret.add(Rs2);
		return ret;
	}

	public NeuralNet getNeuralNet() {
		return nn;
	}

	public void resetGradientAccumulators() {
		Rs1.setAll(0);
		Rs2.setAll(0);
	}

	public void setGradientClipCutoff(double grad_clip_cutoff) {
		this.grad_clip_cutoff = grad_clip_cutoff;
	}

	public void setGradientDecay(double grad_decay) {
		this.grad_decay = grad_decay;
	}

	public void setLearningRate(double learn_rate) {
		this.learn_rate = learn_rate;
	}

	public void setOptimizerType(OptimizerType ot) {
		this.ot = ot;
	}

	public void setUseAverageGradients(boolean use_avg_grad) {
		this.use_avg_grad = use_avg_grad;
	}

	public void setUseHardGradClipping(boolean use_hard_grad_clipping) {
		this.use_hard_grad_clipping = use_hard_grad_clipping;
	}

	public void setL2WeightDecay(double l2_weight_decay) {
		this.l2_weight_decay = l2_weight_decay;
	}

	/**
	 * http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization
	 * 
	 * @param reg_lambda
	 * @param learn_rate
	 * @param data_size
	 */
	public void setWeightDecay(double reg_lambda, double learn_rate, long data_size) {
		l2_weight_decay = (1 - reg_lambda * learn_rate / data_size);
	}

	public void update(int batch_size) {
		t += 0.01;

		for (int i = 0; i < Ws.size(); i++) {
			DenseMatrix W = Ws.get(i);
			DenseMatrix dW = dWs.get(i);
			DenseMatrix R1 = Rs1.get(i);
			DenseMatrix R2 = Rs2.get(i);

			if (use_avg_grad && batch_size > 0) {
				dW.multiply(1d / batch_size);
			}

			if (grad_clip_cutoff != Double.MAX_VALUE) {
				double norm = VectorMath.normL2(dW);

				if (norm > grad_clip_cutoff) {
					if (use_hard_grad_clipping) {
						VectorMath.clip(dW, -grad_clip_cutoff, grad_clip_cutoff, dW);
					} else {
						dW.multiply(grad_clip_cutoff / norm);
					}
				}
			}

			double sum = 0;
			double w = 0;
			double dw = 0;
			double dwa1 = 0;
			double dwa2 = 0;

			synchronized (W) {
				for (int j = 0; j < W.rowSize(); j++) {
					DenseVector dWm = dW.row(j);
					DenseVector Wm = W.row(j);
					DenseVector r1 = R1.row(j);
					DenseVector r2 = R2.row(j);

					// if (g.sum() != 0) {
					// synchronized (Wm) {
					sum = 0;
					if (ot == OptimizerType.SIMPLE) {
						for (int k = 0; k < dWm.size(); k++) {
							dw = dWm.value(k) * grad_decay;
							w = Wm.value(k) * l2_weight_decay;

							w -= learn_rate * dw;
							Wm.set(k, w);
							sum += w;
						}
					} else if (ot == OptimizerType.ADAGRAD) {
						for (int k = 0; k < dWm.size(); k++) {
							dw = dWm.value(k) * grad_decay;
							w = Wm.value(k) * l2_weight_decay;

							r1.add(k, Math.pow(dw, 2));

							w -= learn_rate / (Math.sqrt(dwa2) + eps) * dw;
							Wm.set(k, w);
							sum += w;
						}
					} else if (ot == OptimizerType.RMSPROP) {
						sum = 0;
						for (int k = 0; k < dWm.size(); k++) {
							dw = dWm.value(k) * grad_decay;
							w = Wm.value(k) * l2_weight_decay;

							dwa1 = ArrayMath.addAfterMultiply(r1.value(k), decay_rate, Math.pow(dw, 2), 1 - decay_rate);
							r1.set(k, dwa1);

							w -= -learn_rate / (Math.sqrt(dwa2) + eps) * dw;
							Wm.set(k, w);
							sum += w;
						}
					} else if (ot == OptimizerType.ADAM) {
						for (int k = 0; k < dWm.size(); k++) {
							dw = dWm.value(k) * grad_decay;
							w = Wm.value(k) * l2_weight_decay;

							dwa1 = ArrayMath.addAfterMultiply(r1.value(k), beta1, dw);
							dwa2 = ArrayMath.addAfterMultiply(r2.value(k), beta2, Math.pow(dw, 2));

							r1.set(k, dwa1);
							r2.set(k, dwa2);

							// dxa1 = dxa1 / (1 - beta1);
							// dxa2 = dxa2 / (1 - beta2);

							dwa1 = dwa1 / (1 - Math.pow(beta1, t));
							dwa2 = dwa2 / (1 - Math.pow(beta2, t));

							w -= learn_rate / (Math.sqrt(dwa2) + eps) * dwa1;
							Wm.set(k, w);
							sum += w;
						}
					} else if (ot == OptimizerType.ADAMAX) {
						for (int k = 0; k < dWm.size(); k++) {
							dw = dWm.value(k) * grad_decay;
							w = Wm.value(k) * l2_weight_decay;

							dwa1 = ArrayMath.addAfterMultiply(r1.value(k), beta1, dw);
							dwa2 = Math.max(beta2 * r2.value(k), Math.abs(dw));

							if (dwa2 == 0) {
								dwa2 = eps;
							}

							r1.set(k, dwa1);
							r2.set(k, dwa2);

							w -= learn_rate / dwa2 * dwa1;
							Wm.set(k, w);
							sum += w;
						}
					} else if (ot == OptimizerType.NADAM) {
						for (int k = 0; k < dWm.size(); k++) {
							dw = dWm.value(k) * grad_decay;
							w = Wm.value(k) * l2_weight_decay;

							dwa1 = ArrayMath.addAfterMultiply(r1.value(k), beta1, dw);
							dwa2 = ArrayMath.addAfterMultiply(r2.value(k), beta2, Math.pow(dw, 2));

							r1.set(k, dwa1);
							r2.set(k, dwa2);

							dwa1 = dwa1 / (1 - Math.pow(beta1, t));
							dwa2 = dwa2 / (1 - Math.pow(beta2, t));

							w -= learn_rate / (Math.sqrt(dwa2) + eps) *

									(beta1 * dwa1 + (1 - beta1) / (1 - Math.pow(beta1, t)) * dw);
							Wm.set(k, w);
							sum += w;
						}
					}

					Wm.setSum(sum);
					// }
					dWm.setAll(0);
					// }
				}
			}
		}
	}

}
