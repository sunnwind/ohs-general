package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.ArrayChecker;
import ohs.math.ArrayMath;
import ohs.math.VectorChecker;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import scala.Predef.ArrayCharSequence;

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

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	/**
	 * input of this layer
	 */
	private DenseTensor X;

	private DenseTensor Y;

	private DenseMatrix W;

	private DenseVector b;

	private DenseMatrix dW;

	private DenseVector db;

	private int input_size;

	private int output_size;

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

		// DenseTensor N = this.N;
		// DenseVector Nw = this.Nw;

		VectorUtils.enlarge(tmp_dX, dY.sizeOfInnerVectors(), input_size);
		int start = 0;

		for (int i = 0; i < dY.size(); i++) {
			DenseMatrix dYm = dY.get(i);
			DenseMatrix Xm = X.get(i);
			DenseMatrix Zm = Z.get(i);
			// DenseMatrix Nxm = N.get(i);
			DenseMatrix dXm = tmp_dX.subMatrix(start, dYm.rowSize());
			dXm.setAll(0);

			start += Xm.rowSize();

			// if (use_cosine_norm) {
			// for (int j = 0; j < Xm.rowSize(); j++) {
			// DenseVector xm = Xm.row(j);
			// DenseVector dym = dYm.row(j);
			// DenseVector nxm = Nxm.row(j);
			// DenseVector zm = Zm.row(j);
			//
			// for (int k = 0; k < xm.size(); k++) {
			// double v = xm.value(k);
			// double nx = nxm.value(k);
			// double nw = Nw.value(k);
			//
			// double v2 = v / (nx * nw);
			// }
			//
			// VectorMath.outerProduct(xm, dym, dW, true);
			// VectorMath.add(dym, db);
			// }
			//
			// VectorMath.productRows(dYm, W, dXm, false);
			// } else {
			for (int j = 0; j < Xm.rowSize(); j++) {
				DenseVector xm = Xm.row(j);
				DenseVector dym = dYm.row(j);

				VectorMath.outerProduct(xm, dym, dW, true);
				VectorMath.add(dym, db);
			}
			VectorMath.productRows(dYm, W, dXm, false);
			// }
			dX.add(dXm);
		}

		return dX;
	}

	@Override
	public Layer copy() {
		return new FullyConnectedLayer(W, b);
	}

	private boolean use_cosine_norm = true;

	private DenseVector Nw = new DenseVector(0);

	private DenseMatrix Nx = new DenseMatrix(0);

	private DenseMatrix tmp_Nx = new DenseMatrix(0);

	private DenseTensor N;

	private DenseMatrix tmp_Z = new DenseMatrix(0);

	private DenseTensor Z;

	private DenseVector T;

	@Override
	public DenseTensor forward(Object I) {
		this.X = (DenseTensor) I;

		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		DenseTensor Z = new DenseTensor();
		DenseTensor N = new DenseTensor();

		Y.ensureCapacity(X.size());
		Z.ensureCapacity(X.size());
		N.ensureCapacity(X.size());

		if (Nw.size() == 0) {
			Nw = new DenseVector(W.colSize());
			T = new DenseVector(W.colSize());
		}

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), output_size);
		VectorUtils.enlarge(tmp_Z, X.sizeOfInnerVectors(), output_size);

		int start = 0;

		// DenseVector Nw = this.Nw;
		// VectorMath.normL2(W, Nw, false);

		for (DenseMatrix Xm : X) {
			DenseMatrix Zm = tmp_Z.subMatrix(start, Xm.rowSize());
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());

			Zm.setAll(0);
			Ym.setAll(0);

			start += Xm.rowSize();

			VectorMath.product(Xm, W, Zm, false);

			// if (use_cosine_norm) {
			// DenseVector Nxm = VectorMath.normL2(Xm, true);
			//
			// for (int i = 0; i < Zm.rowSize(); i++) {
			// DenseVector zm = Zm.row(i);
			// double nx = Nxm.value(i);
			//
			// VectorMath.multiply(Nw, Nxm, T);
			// }
			//
			// N.add(Nxm.toDenseMatrix());
			// }

			VectorMath.add(Zm, b, Ym);

			Z.add(Zm);
			Y.add(Ym);
		}

		this.N = N;
		this.Z = Z;
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
	public void initWeights() {
		ParameterInitializer.init2(W);
	}

	@Override
	public void prepareTraining() {
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
