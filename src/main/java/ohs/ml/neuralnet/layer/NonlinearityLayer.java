package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.sql.Date;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;

public class NonlinearityLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -902683936157386041L;

	private Nonlinearity non;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private Object X;

	private Object Y;

	public NonlinearityLayer(Nonlinearity non) {
		this.non = non;
	}

	public NonlinearityLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		Object ret = null;

		if (I instanceof DenseMatrix) {
			DenseMatrix dY = (DenseMatrix) I;
			int data_size = dY.rowSize();

			VectorUtils.enlarge(tmp_dX, dY.rowSize(), dY.colSize());

			DenseMatrix Y = (DenseMatrix) this.Y;

			non.backward(Y, Y);

			DenseMatrix dX = tmp_dX.rows(data_size);
			VectorMath.multiply(dY, Y, dX);
			ret = dX;

		} else if (I instanceof DenseTensor) {
			DenseTensor dY = (DenseTensor) I;

			VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), dY.get(0).colSize());

			DenseTensor Y = (DenseTensor) this.Y;
			DenseTensor dX = new DenseTensor();
			dX.ensureCapacity(Y.rowSize());

			for (int i = 0, start = 0; i < Y.rowSize(); i++) {
				DenseMatrix dYm = dY.get(i);
				DenseMatrix Ym = Y.get(i);
				non.backward(Ym, Ym);

				DenseMatrix dXm = tmp_dX.rows(start, Ym.rowSize());
				VectorMath.multiply(dYm, Ym, dXm);
				dX.add(dXm);

				start += Ym.rowSize();
			}
			ret = dX;
		}

		return ret;
	}

	@Override
	public Layer copy() {
		return new NonlinearityLayer(non);
	}

	@Override
	public Object forward(Object I) {
		this.X = I;

		if (I instanceof DenseMatrix) {
			DenseMatrix X = (DenseMatrix) I;
			int data_size = X.rowSize();

			if (tmp_Y.rowSize() < data_size) {
				tmp_Y = X.copy(true);
			}

			DenseMatrix Y = tmp_Y.rows(data_size);
			non.forward(X, Y);
			this.Y = Y;
		} else if (I instanceof DenseTensor) {
			DenseTensor X = (DenseTensor) I;

			if (tmp_Y.rowSize() < X.sizeOfInnerVectors()) {
				tmp_Y = new DenseMatrix(X.sizeOfInnerVectors(), X.get(0).colSize());
			}

			DenseTensor Y = new DenseTensor();
			Y.ensureCapacity(X.size());

			for (int i = 0, start = 0; i < X.size(); i++) {
				DenseMatrix Xm = X.get(i);
				DenseMatrix Ym = tmp_Y.rows(start, Xm.rowSize());
				non.forward(Xm, Ym);
				Y.add(Ym);
				start += Xm.rowSize();
			}
			this.Y = Y;
		}
		return Y;
	}

	public Nonlinearity getNonlinearity() {
		return non;
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		String name = ois.readUTF();
		/*
		 * https://sites.google.com/site/hmhandks/android/classgaegcheclassmyeong-
		 * eulogaegcheleulsaengseonghaja
		 */

		Class c = Class.forName(name);
		non = (Nonlinearity) c.newInstance();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeUTF(non.getClass().getName());
	}

}
