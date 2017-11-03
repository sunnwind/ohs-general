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

	private double beta1 = 0.9;

	private double beta2 = 0.999;

	private double decay_rate = 0.9;

	private DenseTensor dWs;

	private double eps = 0.00000001;

	/**
	 * Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C.
	 * (2016). Neural Architectures for Named Entity Recognition. Arxiv, 1–10.
	 * Retrieved from http://arxiv.org/abs/1603.01360
	 */
	private double grad_clip_cutoff = 5;

	private double learn_rate = 0.001;

	private NeuralNet nn;

	private OptimizerType ot = OptimizerType.ADAM;

	private DenseTensor Rs1;

	private DenseTensor Rs2;

	private double weight_decay = 1;

	private DenseTensor Ws;

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

	public void setLearningRate(double learn_rate) {
		this.learn_rate = learn_rate;
	}

	public void setOptimizerType(OptimizerType ot) {
		this.ot = ot;
	}

	public void setWeightDecay(double weight_decay) {
		this.weight_decay = weight_decay;
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

		for (int i = 0; i < Ws.size(); i++) {
			DenseMatrix W = Ws.get(i);
			DenseMatrix dW = dWs.get(i);
			DenseMatrix R1 = Rs1.get(i);
			DenseMatrix R2 = Rs2.get(i);

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

			// synchronized (W) {
			for (int j = 0; j < W.rowSize(); j++) {
				DenseVector dw = dW.row(j);
				DenseVector w = W.row(j);
				DenseVector r1 = R1.row(j);
				DenseVector r2 = R2.row(j);

				// if (g.sum() != 0) {
				synchronized (w) {
					sum = 0;
					if (ot == OptimizerType.SIMPLE) {
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k);
							x = w.value(k) * weight_decay;
							x -= learn_rate * dx;
							w.set(k, x);
							sum += x;
						}
					} else if (ot == OptimizerType.ADAGRAD) {
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k);
							r1.add(k, Math.pow(dx, 2));

							x = w.value(k) * weight_decay;
							x -= learn_rate / Math.sqrt(r1.value(k) + eps) * dx;
							w.set(k, x);
							sum += x;
						}
					} else if (ot == OptimizerType.RMSPROP) {
						sum = 0;
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k);
							rv1 = ArrayMath.addAfterMultiply(r1.value(k), decay_rate, Math.pow(dx, 2), 1 - decay_rate);
							r1.set(k, rv1);

							x = w.value(k) * weight_decay;
							x -= -learn_rate / Math.sqrt(rv1 + eps) * dx;
							w.set(k, x);
							sum += x;
						}
					} else if (ot == OptimizerType.ADAM) {
						for (int k = 0; k < dw.size(); k++) {
							dx = dw.value(k);
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
				}
				dw.setAll(0);
				// }
			}
			// }
		}
	}

}
