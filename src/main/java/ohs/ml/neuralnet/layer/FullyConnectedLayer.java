package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import javax.swing.colorchooser.DefaultColorSelectionModel;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
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
	private DenseTensor X;

	private DenseTensor Y;

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
	public DenseTensor backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dY.size());

		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), input_size);
		int start = 0;

		for (int i = 0; i < dY.size(); i++) {
			DenseMatrix dYm = dY.get(i);
			DenseMatrix Xm = X.get(i);
			DenseMatrix dXm = tmp_dX.subMatrix(start, dYm.rowSize());
			dXm.setAll(0);

			start += Xm.rowSize();

			for (int j = 0; j < Xm.rowSize(); j++) {
				DenseVector x = Xm.row(j);
				DenseVector dy = dYm.row(j);

				VectorMath.outerProduct(x, dy, dW, true);
				VectorMath.add(dy, db);
			}

			VectorMath.productRows(dYm, W, dXm, false);
			dX.add(dXm);
		}

		return dX;
	}

	@Override
	public Layer copy() {
		return new FullyConnectedLayer(W, b);
	}

	@Override
	public DenseTensor forward(Object I) {
		this.X = (DenseTensor) I;

		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), output_size);
		int start = 0;

		for (DenseMatrix Xm : X) {
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			VectorMath.product(Xm, W, Ym, false);
			VectorMath.add(Ym, b, Ym);
			Y.add(Ym);
		}

		this.Y = Y;
		return Y;
	}

	@Override
	public DenseTensor getB() {
		DenseTensor ret = new DenseTensor();
		ret.add(b.toDenseMatrix());
		return ret;
	}

	@Override
	public DenseTensor getDB() {
		DenseTensor ret = new DenseTensor();
		ret.add(db.toDenseMatrix());
		return ret;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		ret.add(dW);
		return ret;
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
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.add(W);
		return ret;
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
