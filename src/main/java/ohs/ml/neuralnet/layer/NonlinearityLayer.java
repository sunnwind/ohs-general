package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

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

	private DenseTensor X;

	private DenseTensor Y;

	public NonlinearityLayer(Nonlinearity non) {
		this.non = non;
	}

	public NonlinearityLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dY.size());

		DenseTensor Y = this.Y;

		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), dY.get(0).colSize());

		int start = 0;

		for (int i = 0; i < dY.size(); i++) {
			DenseMatrix dYm = dY.get(i);
			DenseMatrix Ym = Y.get(i);
			non.backward(Ym, Ym);

			DenseMatrix dXm = tmp_dX.subMatrix(start, Ym.rowSize());
			dXm.setAll(0);
			start += Ym.rowSize();

			VectorMath.multiply(dYm, Ym, dXm);
			dX.add(dXm);
		}

		return dX;
	}

	@Override
	public NonlinearityLayer copy() {
		return new NonlinearityLayer(non);
	}

	@Override
	public Object forward(Object I) {
		this.X = (DenseTensor) I;

		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), X.get(0).colSize());
		int start = 0;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			non.forward(Xm, Ym);
			Y.add(Ym);
		}

		this.Y = Y;
		return Y;
	}

	public NonlinearityLayer() {

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
