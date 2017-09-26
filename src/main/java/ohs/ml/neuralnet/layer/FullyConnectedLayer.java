package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.types.number.IntegerArray;

/**
 * 
 * http://cs231n.github.io/neural-networks-2/#init
 * 
 * @author ohs
 */
public class FullyConnectedLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3286065471918528100L;

	private DenseMatrix W;

	private DenseVector b;

	private DenseMatrix dW;

	private DenseVector db;

	/**
	 * input of this layer
	 */
	private Object fwd_I;

	private DenseMatrix tmp_Y;

	private DenseMatrix tmp_dX;

	private int input_size;

	private int output_size;

	public FullyConnectedLayer(DenseMatrix W, DenseVector b) {
		this.W = W;
		this.b = b;

		input_size = W.rowSize();
		output_size = W.colSize();
	}

	public FullyConnectedLayer(int input_size, int output_size) {
		this(new DenseMatrix(input_size, output_size), new DenseVector(output_size));
	}

	public FullyConnectedLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public DenseMatrix backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;
		DenseMatrix dX = null;
		int data_size = dY.rowSize();

		if (fwd_I instanceof DenseMatrix) {
			DenseMatrix X = (DenseMatrix) fwd_I;
			for (int i = 0; i < X.rowSize(); i++) {

				VectorMath.outerProduct(X.row(i), dY.row(i), dW, true);
				VectorMath.add(dY.row(i), db);
			}
			if (tmp_dX == null || tmp_dX.rowSize() < dY.rowSize()) {
				tmp_dX = new DenseMatrix(data_size, input_size);
			}
			dX = tmp_dX.rows(data_size);
			VectorMath.productRows(dY, W, dX, false);
		} else {
			IntegerArray X = (IntegerArray) fwd_I;
			for (int i = 0; i < X.size(); i++) {
				int idx = X.get(i);
				DenseVector dw = dW.row(idx);
				VectorMath.add(dw, dY.row(i), dw);
			}
		}
		return dX;
	}

	public void computeDB() {
		// VectorMath.sumColumns(dW, db);
	}

	@Override
	public DenseMatrix forward(Object I) {
		this.fwd_I = I;

		int data_size = I instanceof DenseMatrix ? ((DenseMatrix) I).rowSize() : ((IntegerArray) I).size();

		if (tmp_Y == null || tmp_Y.rowSize() < data_size) {
			tmp_Y = new DenseMatrix(data_size, output_size);
		}

		DenseMatrix Y = tmp_Y.rows(data_size);

		if (I instanceof DenseMatrix) {
			DenseMatrix X = (DenseMatrix) I;
			VectorMath.product(X, W, Y, false);
			VectorMath.add(Y, b, Y);
		} else {
			IntegerArray X = (IntegerArray) I;
			for (int i = 0; i < X.size(); i++) {
				int idx = X.get(i);
				VectorUtils.copy(W.row(idx), Y.row(i));
			}
			VectorMath.add(Y, b, Y);
		}
		return Y;
	}

	@Override
	public DenseMatrix getB() {
		return b.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDB() {
		return db.toDenseMatrix();
	}

	@Override
	public DenseMatrix getDW() {
		return dW;
	}

	@Override
	public int getInputSize() {
		return W.rowSize();
	}

	@Override
	public int getOutputSize() {
		return W.colSize();
	}

	@Override
	public DenseMatrix getW() {
		return W;
	}

	@Override
	public void init() {
		ParameterInitializer.init2(W);
	}

	@Override
	public void prepare() {
		dW = W.copy(true);
		db = b.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		W = new DenseMatrix(ois);
		b = new DenseVector(ois);

		input_size = W.rowSize();
		output_size = W.colSize();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		W.writeObject(oos);
		b.writeObject(oos);
	}

}
