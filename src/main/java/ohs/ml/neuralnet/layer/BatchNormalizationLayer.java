package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;

/**
 * https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
 * 
 * http://cthorey.github.io./backpropagation/
 * 
 * https://github.com/cthorey/CS231/blob/master/assignment3/cs231n/layers.py
 * 
 * https://eleg5491.github.io/initialization-and-normalization#weightnorm
 * 
 * @author ohs
 */
public class BatchNormalizationLayer extends Layer {

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

	private DenseVector mu = new DenseVector(0);

	private DenseVector runMeans;

	private DenseVector runVars;

	private DenseVector std;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_dXC = new DenseMatrix(0);

	private DenseMatrix tmp_dXN = new DenseMatrix(0);

	private DenseMatrix tmp_T = new DenseMatrix(0);

	private DenseMatrix tmp_XC = new DenseMatrix(0);

	private DenseMatrix tmp_XN = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private DenseVector var = new DenseVector(0);

	private DenseTensor X;

	private DenseTensor XC;

	private DenseTensor XN;

	private DenseTensor Y;

	public BatchNormalizationLayer(DenseVector runMeans, DenseVector runVars, DenseVector gamma, DenseVector beta) {
		this.runMeans = runMeans;
		this.runVars = runVars;
		this.gamma = gamma;
		this.beta = beta;
	}

	public BatchNormalizationLayer(int output_size) {
		this(new DenseVector(output_size), new DenseVector(output_size), new DenseVector(output_size),
				new DenseVector(output_size));
	}

	public BatchNormalizationLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dY.size());

		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), dY.get(0).colSize());
		VectorUtils.enlarge(tmp_dXC, dY.sizeOfInnerVectors(), dY.get(0).colSize());
		VectorUtils.enlarge(tmp_dXN, dY.sizeOfInnerVectors(), dY.get(0).colSize());
		VectorUtils.enlarge(tmp_T, dY.sizeOfInnerVectors(), dY.get(0).colSize());

		int start = 0;

		for (int i = 0; i < dY.size(); i++) {
			DenseMatrix dYm = dY.get(i);
			DenseMatrix XNm = XN.get(i);
			DenseMatrix XCm = XC.get(i);
			DenseMatrix Tm = tmp_T.subMatrix(start, dYm.rowSize());
			DenseMatrix dXm = tmp_dX.subMatrix(start, dYm.rowSize());
			DenseMatrix dXCm = tmp_dXC.subMatrix(start, dYm.rowSize());
			DenseMatrix dXNm = tmp_dXN.subMatrix(start, dYm.rowSize());

			Tm.setAll(0);
			dXm.setAll(0);
			dXCm.setAll(0);
			dXNm.setAll(0);

			start += dYm.rowSize();

			VectorMath.sumColumns(dYm, dbeta);

			VectorMath.multiply(XNm, dYm, Tm);
			VectorMath.sumColumns(Tm, dgamma);

			VectorMath.multiply(dYm, gamma, dXNm);

			VectorMath.divide(dXNm, std, dXCm);

			VectorMath.multiply(dXNm, XCm, Tm);
			DenseVector std_sq = std.copy();
			VectorMath.pow(std_sq, 2, std_sq);
			VectorMath.divide(Tm, std_sq, Tm);
			DenseVector dstd = gamma.copy(true);
			VectorMath.sumColumns(Tm, dstd);
			dstd.multiply(-1);

			DenseVector dvar = gamma.copy(true);
			VectorMath.divide(dstd, std, dvar);
			dvar.multiply(0.5);

			Tm.setAll(2f / dYm.rowSize());

			VectorMath.multiply(Tm, XCm, Tm);
			VectorMath.multiply(Tm, dvar, Tm);
			VectorMath.add(dXCm, Tm, dXCm);

			DenseVector dmu = gamma.copy(true);
			VectorMath.mean(dXCm, dmu, false);

			VectorMath.subtract(dXCm, dmu, dXm);

			dX.add(dXm);
		}

		return dX;
	}

	@Override
	public Layer copy() {
		return new BatchNormalizationLayer(runMeans, runVars, gamma, beta);
	}

	@Override
	public Object forward(Object I) {
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		DenseTensor XN = new DenseTensor();
		DenseTensor XC = new DenseTensor();

		Y.ensureCapacity(X.size());
		XN.ensureCapacity(X.size());
		XC.ensureCapacity(X.size());

		this.X = X;

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), X.get(0).colSize());

		if (is_training) {
			int size = X.sizeOfInnerVectors();
			VectorUtils.enlarge(tmp_T, size, X.get(0).colSize());
			VectorUtils.enlarge(tmp_XC, size, X.get(0).colSize());
			VectorUtils.enlarge(tmp_XN, size, X.get(0).colSize());
		}

		if (mu.size() == 0) {
			mu = gamma.copy(true);
			var = gamma.copy(true);
			std = gamma.copy(true);
		}

		int start = 0;

		for (DenseMatrix Xm : X) {
			if (is_training) {
				DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
				DenseMatrix Tm = tmp_T.subMatrix(start, Xm.rowSize());
				DenseMatrix XCm = tmp_XC.subMatrix(start, Xm.rowSize());
				DenseMatrix XNm = tmp_XN.subMatrix(start, Xm.rowSize());

				Ym.setAll(0);
				Tm.setAll(0);
				XCm.setAll(0);
				XNm.setAll(0);

				start += Xm.rowSize();

				/*
				 * step 1: calculate mean
				 */
				VectorMath.mean(Xm, mu, false);

				// step 2: subtract mean vector of every training example

				VectorMath.subtract(Xm, mu, XCm);

				// step 3: following the lower branch - calculation DENOMINATOR

				VectorMath.pow(XCm, 2, Tm);

				// step 4: calculate variance

				VectorMath.mean(Tm, var, false);

				// step 5: add eps for numerical stability, then sqrt

				VectorUtils.copy(var, std);
				std.add(eps);
				VectorMath.sqrt(std, std);
				VectorMath.divide(XCm, std, XNm);

				VectorMath.multiply(XNm, gamma, Ym);
				VectorMath.add(Ym, beta, Ym);

				VectorMath.addAfterMultiply(runMeans, momentum, mu, (1 - momentum), runMeans);
				VectorMath.addAfterMultiply(runVars, momentum, var, (1 - momentum), runVars);

				Y.add(Ym);
				XN.add(XNm);
				XC.add(XCm);
			} else {
				DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
				Ym.setAll(0);

				start += Xm.rowSize();

				DenseVector std = runVars.copy();
				std.add(eps);

				VectorMath.sqrt(std, std);
				VectorMath.subtract(Xm, runMeans, Ym);
				VectorMath.divide(Ym, std, Ym);
				VectorMath.multiply(Ym, gamma, Ym);
				VectorMath.add(Ym, beta, Ym);

				Y.add(Ym);
			}
		}

		this.Y = Y;
		this.XN = XN;
		this.XC = XC;

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
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.add(gamma.toDenseMatrix());
		ret.add(beta.toDenseMatrix());
		return ret;
	}

	@Override
	public void initWeights(ParameterInitializer pi) {
		gamma.setAll(1);
		runVars.setAll(1);
	}

	@Override
	public void createGradientHolders() {
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
