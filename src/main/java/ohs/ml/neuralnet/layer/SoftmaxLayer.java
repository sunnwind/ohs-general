package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;

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
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		X.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), X.get(0).colSize());
		int start = 0;

		for (DenseMatrix Xm : X) {
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);
			start += Ym.size();

			VectorMath.softmax(Xm, Ym);
			Y.add(Ym);
		}
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
