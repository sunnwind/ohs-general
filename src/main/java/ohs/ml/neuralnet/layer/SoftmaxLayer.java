package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;

public class SoftmaxLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private int output_size;

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	public SoftmaxLayer() {

	}

	public SoftmaxLayer(int output_size) {
		this.output_size = output_size;
	}

	public SoftmaxLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		return I;
	}

	@Override
	public Layer copy() {
		return new SoftmaxLayer(output_size);
	}

	@Override
	public Object forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;
		int data_size = X.rowSize();

		VectorUtils.enlarge(tmp_Y, data_size, X.colSize());

		DenseMatrix Y = tmp_Y.rows(data_size);
		VectorMath.softmax(X, Y);
		return Y;
	}

	@Override
	public int getOutputSize() {
		return output_size;
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		output_size = ois.readInt();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(output_size);
	}

}
