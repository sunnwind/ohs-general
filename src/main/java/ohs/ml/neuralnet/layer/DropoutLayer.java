package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;

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
	public DenseTensor backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), dY.get(0).colSize());

		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dY.size());

		int start = 0;

		for (DenseMatrix dYm : dY) {
			DenseMatrix dXm = tmp_dX.subMatrix(start, dYm.rowSize());
			start += dYm.rowSize();

			VectorUtils.copy(dYm, dXm);

			dX.add(dXm);
		}

		return dX;
	}

	@Override
	public Layer copy() {
		return new DropoutLayer(p);
	}

	@Override
	public Object forward(Object I) {
		DenseTensor X = (DenseTensor) I;

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), X.get(0).colSize());

		DenseTensor Y = new DenseTensor();
		Y.ensureCapacity(X.size());

		int start = 0;
		for (DenseMatrix Xm : X) {
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			if (is_training) {
				VectorUtils.enlarge(tmp_M, Xm.rowSize(), Xm.colSize());

				DenseMatrix M = tmp_M.subMatrix(Xm.rowSize());
				M.setAll(0);

				VectorMath.random(0, 1, M);
				VectorMath.mask(M, p);
				M.multiply(1f / p); // inverted drop-out
				VectorMath.multiply(Xm, M, Ym);
			} else {
				VectorUtils.copy(Xm, Ym);
			}

			Y.add(Ym);
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
