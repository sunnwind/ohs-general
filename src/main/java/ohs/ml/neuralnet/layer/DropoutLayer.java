package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;

/**
 * http://cs231n.github.io/neural-networks-2/
 * 
 * @author ohs
 */
public class DropoutLayer extends Layer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4640822254772512028L;

	private double p = 0.5;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_M = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	public DropoutLayer() {
		this(0.5);
	}

	public DropoutLayer(double p) {
		this.p = p;
	}

	public DropoutLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public DenseMatrix backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;
		VectorUtils.enlarge(tmp_dX, dY.rowSize(), dY.colSize());
		DenseMatrix dX = tmp_dX.rows(dY.rowSize());
		VectorUtils.copy(dY, dX);
		return dX;
	}

	@Override
	public Layer copy() {
		return new DropoutLayer(p);
	}

	@Override
	public DenseMatrix forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;
		int data_size = X.rowSize();

		VectorUtils.enlarge(tmp_Y, data_size, X.colSize());

		DenseMatrix Y = tmp_Y.rows(data_size);

		if (is_testing) {
			VectorUtils.copy(X, Y);
		} else {
			VectorUtils.enlarge(tmp_M, data_size, X.colSize());

			DenseMatrix M = tmp_M.rows(data_size);
			VectorMath.random(0, 1, M);
			VectorMath.mask(M, p);
			M.multiply(1f / p); // inverted drop-out

			VectorMath.multiply(X, M, Y);
		}
		return Y;
	}

	public double getP() {
		return p;
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		p = ois.readDouble();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeDouble(p);
	}

}
