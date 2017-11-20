package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.commons.math.stat.descriptive.moment.Mean;

import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;

/**
 * http://www.breloff.com/layernorm/
 * 
 * https://theneuralperspective.com/2016/10/27/gradient-topics/
 * 
 * @author ohs
 */
public class LayerNormalizationLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3632625740730865189L;

	private DenseVector beta;

	private DenseVector dbeta;

	private DenseVector dgamma;

	private double eps = 0.00000001;

	private DenseVector gamma;

	private double momentum = 0.9;

	private DenseMatrix tmp_dZ = new DenseMatrix(0);

	private DenseMatrix tmp_Z = new DenseMatrix(0);

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private DenseTensor A;

	private DenseTensor Y;

	private DenseMatrix T = new DenseMatrix(0);

	private DenseMatrix Mt = new DenseMatrix();

	private DenseMatrix Vt = new DenseMatrix();

	private DenseMatrix M = new DenseMatrix();

	private DenseTensor Z;

	private double run_mu = 1;

	private double run_var = 0;

	public LayerNormalizationLayer(double run_mu, double run_var, double momentum, DenseVector gamma, DenseVector beta) {
		this.run_mu = run_mu;
		this.run_var = run_var;
		this.momentum = momentum;
		this.gamma = gamma;
		this.beta = beta;
	}

	public LayerNormalizationLayer(int output_size) {
		this(1, 0, 0.9, new DenseVector(output_size, 1d), new DenseVector(output_size));
	}

	public LayerNormalizationLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor dA = new DenseTensor();
		dA.ensureCapacity(dY.size());

		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), dY.row(0).colSize());
		VectorUtils.enlarge(tmp_dZ, dY.sizeOfInnerVectors(), dY.row(0).colSize());

		if (T.size() == 0) {
			T = new DenseMatrix(dY.row(0).colSize(), dY.row(0).colSize());
		}

		int start = 0;

		for (int i = 0; i < dY.size(); i++) {
			DenseMatrix dYm = dY.get(i);
			DenseMatrix Zm = Z.get(i);

			DenseMatrix dAm = tmp_dX.subMatrix(start, dYm.rowSize());
			DenseMatrix dZm = tmp_dZ.subMatrix(start, dYm.rowSize());

			dAm.setAll(0);
			dZm.setAll(0);

			start += dYm.rowSize();

			DenseVector mus = M.row(i);
			DenseVector run_mus = Mt.row(i);
			DenseVector run_vars = Vt.row(i);

			for (int t = 0; t < dYm.rowSize(); t++) {
				DenseVector dym = dYm.row(t);
				DenseVector dam = dAm.row(t);
				DenseVector zm = Zm.row(t);
				DenseVector dzm = dZm.row(t);

				double mu = mus.value(t);
				double var_t = run_vars.value(t);
				double mu_t = run_mus.value(t);
				double diff = (mu_t - mu) / var_t;

				double D = dym.size();

				for (int o = 0; o < zm.size(); o++) {
					for (int p = 0; p < zm.size(); p++) {
						double v1 = o == p ? 1 : 0;
						double v2 = momentum
								* ((1d / D) + zm.value(o) * zm.value(p) / (D - 1) - momentum * zm.value(0) * diff);
						double v = 1d / var_t * (v1 - v2);
						T.set(o, p, v);
					}
				}

				for (int p = 0; p < zm.size(); p++) {
					double sum = 0;
					for (int o = 0; o < zm.size(); o++) {
						sum += dym.value(o) * gamma.value(o) * T.value(o, p);
					}
					dam.add(p, sum);
				}

				for (int o = 0; o < zm.size(); o++) {
					dgamma.add(o, dym.value(o) * zm.value(0));
					dbeta.add(o, dym.value(o));
				}
			}

			dA.add(dAm);
		}

		return dA;
	}

	@Override
	public Layer copy() {
		return new LayerNormalizationLayer(run_mu, run_var, momentum, gamma, beta);
	}

	@Override
	public Object forward(Object I) {
		DenseTensor A = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		DenseTensor Z = new DenseTensor();

		Y.ensureCapacity(A.size());
		Z.ensureCapacity(A.size());

		VectorUtils.enlarge(tmp_Y, A.sizeOfInnerVectors(), A.row(0).colSize());
		VectorUtils.enlarge(tmp_Z, A.sizeOfInnerVectors(), A.row(0).colSize());

		DenseMatrix M = new DenseMatrix();
		DenseMatrix Mrun = new DenseMatrix();
		DenseMatrix Vrun = new DenseMatrix();

		int start = 0;

		for (DenseMatrix Am : A) {
			if (is_testing) {
				DenseMatrix Ym = tmp_Y.subMatrix(start, Am.rowSize());
				Ym.setAll(0);

				start += Am.rowSize();

				for (int t = 0; t < Am.rowSize(); t++) {
					DenseVector am = Am.row(t);
					DenseVector ym = Ym.row(t);

					double mu = VectorMath.mean(am);
					double var = VectorMath.variance(am, mu);

					VectorMath.zTransform(am, mu, var, eps, ym);
					VectorMath.multiply(gamma, ym, ym);
					VectorMath.add(beta, ym);
				}
				Y.add(Ym);
			} else {
				DenseMatrix Ym = tmp_Y.subMatrix(start, Am.rowSize());
				DenseMatrix Zm = tmp_Z.subMatrix(start, Am.rowSize());

				Ym.setAll(0);
				Zm.setAll(0);

				start += Am.rowSize();

				DenseVector mus = new DenseVector(Am.rowSize());
				DenseVector run_mus = new DenseVector(Am.rowSize());
				DenseVector run_vars = new DenseVector(Am.rowSize());

				for (int t = 0; t < Am.rowSize(); t++) {
					DenseVector am = Am.row(t);
					DenseVector zm = Zm.row(t);
					DenseVector ym = Ym.row(t);

					double mu = VectorMath.mean(am);
					double var = VectorMath.variance(am, mu);

					mus.add(t, mu);

					run_mu = ArrayMath.addAfterMultiply(mu, momentum, run_mu, 1 - momentum);
					run_var = ArrayMath.addAfterMultiply(var, momentum, run_var, 1 - momentum);

					run_mus.add(t, run_mu);
					run_vars.add(t, run_var);

					VectorMath.zTransform(am, mu, var, eps, zm);
					VectorMath.multiply(gamma, zm, ym);
					VectorMath.add(beta, ym);
				}

				Y.add(Ym);
				Z.add(Zm);

				M.add(mus);
				Mrun.add(run_mus);
				Vrun.add(run_vars);
			}
		}

		this.A = A;
		this.Y = Y;
		this.Z = Z;
		this.M = M;
		this.Mt = Mrun;
		this.Vt = Vrun;

		return Y;
	}

	public DenseVector getBeta() {
		return beta;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		ret.add(dgamma.toDenseMatrix());
		ret.add(dbeta.toDenseMatrix());
		return ret;
	}

	public DenseVector getGamma() {
		return gamma;
	}

	@Override
	public int getInputSize() {
		return 0;
	}

	@Override
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.add(gamma.toDenseMatrix());
		ret.add(beta.toDenseMatrix());
		return ret;
	}

	@Override
	public void initWeights() {
		gamma.setAll(1);
	}

	@Override
	public void prepareTraining() {
		dgamma = gamma.copy(true);
		dbeta = gamma.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		run_mu = ois.readDouble();
		run_var = ois.readDouble();
		momentum = ois.readDouble();
		gamma = new DenseVector(ois);
		beta = new DenseVector(ois);
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeDouble(run_mu);
		oos.writeDouble(run_var);
		oos.writeDouble(momentum);
		gamma.writeObject(oos);
		beta.writeObject(oos);
	}

}
