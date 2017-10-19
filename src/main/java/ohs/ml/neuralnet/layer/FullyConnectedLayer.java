package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;

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

	private DenseVector b;

	private DenseVector db;

	private DenseMatrix dW;

	private int input_size;

	private int output_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private DenseMatrix W;

	/**
	 * input of this layer
	 */
	private Object X;

	private DenseMatrix Y;

	public FullyConnectedLayer() {

	}

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

		if (X instanceof DenseMatrix) {
			int data_size = dY.rowSize();
			DenseMatrix X = (DenseMatrix) this.X;

			for (int i = 0; i < X.rowSize(); i++) {
				VectorMath.outerProduct(X.row(i), dY.row(i), dW, true);
				VectorMath.add(dY.row(i), db);
			}

			VectorUtils.enlarge(tmp_dX, data_size, input_size);

			dX = tmp_dX.rows(data_size);
			VectorMath.productRows(dY, W, dX, false);

		} else if (X instanceof IntegerArray) {
			IntegerArray X = (IntegerArray) this.X;

			for (int i = 0; i < X.size(); i++) {
				int idx = X.get(i);
				DenseVector dw = dW.row(idx);
				VectorMath.add(dw, dY.row(i), dw);
			}
		}
		return dX;
	}

	@Override
	public Layer copy() {
		return new FullyConnectedLayer(W, b);
	}

	@Override
	public DenseMatrix forward(Object I) {
		this.X = I;

		int data_size = I instanceof DenseMatrix ? ((DenseMatrix) I).rowSize() : ((IntegerArray) I).size();

		VectorUtils.enlarge(tmp_Y, data_size, output_size);

		DenseMatrix Y = tmp_Y.rows(data_size);
		Y.setAll(0);

		if (I instanceof DenseMatrix) {
			DenseMatrix X = (DenseMatrix) I;
			VectorMath.product(X, W, Y, false);
			VectorMath.add(Y, b, Y);
		} else if (I instanceof IntegerArray) {
			IntegerArray X = (IntegerArray) I;
			for (int i = 0; i < X.size(); i++) {
				int idx = X.get(i);
				VectorUtils.copy(W.row(idx), Y.row(i));
			}
			VectorMath.add(Y, b, Y);
		}

		this.Y = Y;

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
