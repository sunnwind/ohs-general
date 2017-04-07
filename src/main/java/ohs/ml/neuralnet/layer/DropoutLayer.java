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

	private DenseMatrix M;

	private DenseMatrix tmp_M;

	private DenseMatrix tmp_Y;

	private DenseMatrix tmp_dX;

	private double p = 0.5;

	private int input_size;

	private int output_size;

	public DropoutLayer(int output_size) {
		this(output_size, 0.5);
	}

	public DropoutLayer(int output_size, double p) {
		this.input_size = output_size;
		this.output_size = output_size;
		this.p = p;
	}

	public DropoutLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public DenseMatrix backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;
		if (tmp_dX == null || tmp_dX.rowSize() < dY.rowSize()) {
			tmp_dX = dY.copy(true);
		}
		DenseMatrix dX = tmp_dX.rowsAsMatrix(dY.rowSize());
		VectorUtils.copy(dY, dX);
		return dX;
	}

	@Override
	public DenseMatrix forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;
		int data_size = X.rowSize();

		if (tmp_Y == null || tmp_Y.rowSize() < data_size) {
			tmp_Y = X.copy(true);
		}

		DenseMatrix Y = tmp_Y.rowsAsMatrix(X.rowSize());

		if (is_testing) {
			VectorUtils.copy(X, Y);
		} else {
			if (tmp_M == null || tmp_M.rowSize() < X.rowSize()) {
				tmp_M = X.copy(true);
				tmp_Y = X.copy(true);
			}

			M = tmp_M.rowsAsMatrix(X.rowSize());

			VectorMath.random(0, 1, M);
			VectorMath.mask(M, p);
			M.multiply(1f / p); // inverted drop-out

			VectorMath.multiply(X, M, Y);
		}
		return Y;
	}

	@Override
	public int getOutputSize() {
		return output_size;
	}

	public double getP() {
		return p;
	}

	@Override
	public DenseMatrix getW() {
		return null;
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		output_size = ois.readInt();
		p = ois.readDouble();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(output_size);
		oos.writeDouble(p);
	}

}
