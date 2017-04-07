package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.ml.neuralnet.nonlinearity.LeakReLU;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.ml.neuralnet.nonlinearity.ReLU;
import ohs.ml.neuralnet.nonlinearity.Sigmoid;
import ohs.ml.neuralnet.nonlinearity.Tanh;

public class NonlinearityLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -902683936157386041L;

	private Nonlinearity non;

	private DenseMatrix Z;

	private DenseMatrix tmp_Z;

	private DenseMatrix tmp_dX;

	private int output_size;

	public NonlinearityLayer(int output_size, Nonlinearity non) {
		this.output_size = output_size;
		this.non = non;
	}

	public NonlinearityLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public DenseMatrix backward(Object I) {
		DenseMatrix dY = (DenseMatrix) I;
		int data_size = dY.rowSize();

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = dY.copy(true);
		}

		non.backward(Z, Z);

		DenseMatrix dX = tmp_dX.rowsAsMatrix(data_size);
		VectorMath.multiply(dY, Z, dX);
		return dX;
	}

	@Override
	public DenseMatrix forward(Object I) {
		DenseMatrix X = (DenseMatrix) I;
		int data_size = X.rowSize();
		if (tmp_Z == null || tmp_Z.rowSize() < data_size) {
			tmp_Z = X.copy(true);
		}

		Z = tmp_Z.rowsAsMatrix(data_size);
		non.forward(X, Z);
		return Z;
	}

	public Nonlinearity getNonlinearity() {
		return non;
	}

	@Override
	public int getOutputSize() {
		return output_size;
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		output_size = ois.readInt();
		String name = ois.readUTF();
		/*
		 * https://sites.google.com/site/hmhandks/android/classgaegcheclassmyeong-eulogaegcheleulsaengseonghaja
		 */

		Class c = Class.forName(name);
		non = (Nonlinearity) c.newInstance();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(output_size);
		oos.writeUTF(non.getClass().getName());
	}

}
