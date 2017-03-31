package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;

/**
 * https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
 * 
 * http://cthorey.github.io./backpropagation/
 * 
 * https://github.com/cthorey/CS231/blob/master/assignment3/cs231n/layers.py
 * 
 * @author ohs
 */
public class BatchNormalizationLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3632625740730865189L;

	private DenseVector runMeans;

	private DenseVector runVars;

	private double momentum = 0.9;

	private double eps = 0.00000001;

	private DenseVector mu;

	private DenseMatrix xc;

	private DenseVector var;

	private DenseMatrix xn;

	private DenseVector gamma;

	private DenseVector beta;

	private DenseVector dgamma;

	private DenseVector dbeta;

	private DenseVector std;

	private DenseMatrix tmpT;

	private DenseMatrix tmp_Y;

	private DenseMatrix tmp_dX;

	private DenseMatrix tmp_dxn;

	private DenseMatrix tmp_dxc;

	private DenseMatrix tmp_xc;

	private DenseMatrix tmp_xn;

	public BatchNormalizationLayer(DenseVector runMeans, DenseVector runVars, DenseVector gamma, DenseVector beta) {
		this.runMeans = runMeans;
		this.runVars = runVars;
		this.gamma = gamma;
		this.beta = beta;
	}

	public BatchNormalizationLayer(int output_size) {
		this(new DenseVector(output_size), new DenseVector(output_size), new DenseVector(output_size), new DenseVector(output_size));
	}

	public BatchNormalizationLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;

		if (I == null) {
			return null;
		}

		int data_size = dY.rowSize();

		if (tmp_dxn == null || tmp_dxn.rowSize() < data_size) {
			tmp_dxc = dY.copy(true);
			tmp_dxn = dY.copy(true);
			tmp_dX = dY.copy(true);
		}

		VectorMath.sumColumns(dY, dbeta);

		DenseMatrix T = tmpT.rowsAsMatrix(data_size);
		VectorMath.multiply(xn, dY, T);
		VectorMath.sumColumns(T, dgamma);

		DenseMatrix dxn = tmp_dxn.rowsAsMatrix(data_size);
		VectorMath.multiply(dY, gamma, dxn);

		DenseMatrix dxc = tmp_dxc.rowsAsMatrix(data_size);
		VectorMath.divide(dxn, std, dxc);

		VectorMath.multiply(dxn, xc, T);
		DenseVector std_sq = std.copy();
		VectorMath.pow(std_sq, 2, std_sq);
		VectorMath.divide(T, std_sq, T);
		DenseVector dstd = gamma.copy(true);
		VectorMath.sumColumns(T, dstd);
		dstd.multiply(-1);

		DenseVector dvar = gamma.copy(true);
		VectorMath.divide(dstd, std, dvar);
		dvar.multiply(0.5);

		T.setAll(2f / data_size);
		VectorMath.multiply(T, xc, T);
		VectorMath.multiply(T, dvar, T);
		VectorMath.add(dxc, T, dxc);

		DenseVector dmu = gamma.copy(true);
		VectorMath.meanColumns(dxc, dmu);

		DenseMatrix dX = tmp_dX.rowsAsMatrix(data_size);
		VectorMath.subtract(dxc, dmu, dX);

		return dX;
	}

	@Override
	public Object forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;

		if (tmp_Y == null || tmp_Y.rowSize() < X.rowSize()) {
			tmp_Y = X.copy(true);
		}

		DenseMatrix Y = tmp_Y.rowsAsMatrix(X.rowSize());

		if (is_testing) {
			DenseVector std = runVars.copy();
			std.add(eps);

			VectorMath.sqrt(std, std);
			VectorMath.subtract(X, runMeans, Y);
			VectorMath.divide(Y, std, Y);
			VectorMath.multiply(Y, gamma, Y);
			VectorMath.add(Y, beta, Y);
		} else {
			if (tmpT == null || tmpT.rowSize() < X.rowSize()) {
				tmpT = X.copy(true);
				tmp_xc = X.copy(true);
				tmp_xn = X.copy(true);
			}

			if (mu == null) {
				mu = gamma.copy(true);
				var = gamma.copy(true);
				std = gamma.copy(true);
			}

			int size = X.rowSize();
			DenseMatrix T = tmpT.rowsAsMatrix(size);
			xc = tmp_xc.rowsAsMatrix(size);
			xn = tmp_xn.rowsAsMatrix(size);

			/*
			 * step 1: calculate mean
			 */
			VectorMath.meanColumns(X, mu);

			// step 2: subtract mean vector of every trainings example

			VectorMath.subtract(X, mu, xc);

			// step 3: following the lower branch - calculation DENOMINATOR

			VectorMath.pow(xc, 2, T);

			// step 4: calculate variance

			VectorMath.meanColumns(T, var);

			// step 5: add eps for numerical stability, then sqrt

			VectorUtils.copy(var, std);
			std.add(eps);
			VectorMath.sqrt(std, std);
			VectorMath.divide(xc, std, xn);

			VectorMath.multiply(xn, gamma, Y);
			VectorMath.add(Y, beta, Y);

			VectorMath.addAfterMultiply(runMeans, momentum, mu, (1 - momentum), runMeans);
			VectorMath.addAfterMultiply(runVars, momentum, var, (1 - momentum), runVars);
		}

		return Y;
	}

	public DenseVector getBeta() {
		return beta;
	}

	@Override
	public DenseMatrix getDW() {
		return new DenseMatrix(new DenseVector[] { dgamma, dbeta });
	}

	public DenseVector getGamma() {
		return gamma;
	}

	@Override
	public int getInputSize() {
		return 0;
	}

	@Override
	public int getOutputSize() {
		return runMeans.size();
	}

	public DenseVector getRunMeans() {
		return runMeans;
	}

	public DenseVector getRunVars() {
		return runVars;
	}

	@Override
	public DenseMatrix getW() {
		return new DenseMatrix(new DenseVector[] { gamma, beta });
	}

	@Override
	public void init() {
		gamma.setAll(1);
		runVars.setAll(1);
	}

	@Override
	public void prepareTraining() {
		dgamma = gamma.copy(true);
		dbeta = gamma.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		runMeans = new DenseVector(ois);
		runVars = new DenseVector(ois);
		gamma = new DenseVector(ois);
		beta = new DenseVector(ois);
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		runMeans.writeObject(oos);
		runVars.writeObject(oos);
		gamma.writeObject(oos);
		beta.writeObject(oos);
	}

}
