package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

/**
 * 
 * Graves, A. (2012). Supervised Sequence Labelling with Recurrent Neural
 * Networks. http://doi.org/10.1007/978-3-642-24797-2
 * 
 * 
 * https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
 * 
 * https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
 * 
 * @author ohs
 */
public class BidirectionalRecurrentLayer extends RecurrentLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8764635726766071893L;

	private RecurrentLayer bwd;

	private RecurrentLayer fwd;

	private int input_size;

	private int output_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_rX = new DenseMatrix(0);

	private DenseMatrix tmp_rY = new DenseMatrix(0);

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private DenseTensor Y;

	public BidirectionalRecurrentLayer() {

	}

	public BidirectionalRecurrentLayer(RecurrentLayer fwd, RecurrentLayer bwd) {
		super();
		this.fwd = fwd;
		this.bwd = bwd;
		this.input_size = fwd.getInputSize();
		this.output_size = fwd.getOutputSize();
	}

	public BidirectionalRecurrentLayer(Type type, int input_size, int hidden_size, int shift_size, int window_size,
			Nonlinearity non) {
		if (type == Type.RNN) {
			fwd = new RnnLayer(input_size, hidden_size, shift_size, window_size, non);
			bwd = new RnnLayer(input_size, hidden_size, shift_size, window_size, non);
		} else if (type == Type.LSTM) {
			fwd = new LstmLayer(input_size, hidden_size, shift_size, window_size);
			bwd = new LstmLayer(input_size, hidden_size, shift_size, window_size);
		}

		this.input_size = fwd.getInputSize();
		this.output_size = fwd.getOutputSize();
	}

	private DenseMatrix tmp_dZ = new DenseMatrix(0);

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;
		DenseTensor dYf = new DenseTensor();
		DenseTensor dYb = new DenseTensor();

		dYf.ensureCapacity(dY.size());
		dYb.ensureCapacity(dY.size());

		VectorUtils.enlarge(tmp_dZ, dY.sizeOfInnerVectors() * 2, output_size);

		{
			int start = 0;
			for (int i = 0; i < dY.size(); i++) {
				DenseMatrix dYm = dY.get(i);

				int size = dYm.rowSize();

				DenseMatrix dYfm = tmp_dZ.subMatrix(start, size);
				start += size;

				DenseMatrix dYbm = tmp_dZ.subMatrix(start, size);
				start += size;

				dYfm.setAll(0);
				dYbm.setAll(0);

				for (int j = 0; j < dYm.rowSize(); j++) {
					DenseVector dym = dYm.row(j);
					DenseVector dyfm = dYfm.row(j);
					DenseVector dybm = dYbm.row(j);

					VectorUtils.copy(dym, 0, dyfm, 0, output_size);
					VectorUtils.copy(dym, output_size, dybm, 0, output_size);
				}

				dYf.add(dYfm);
				dYb.add(dYfm);
			}
		}

		dYb = reverse(dYb);

		DenseTensor dXf = (DenseTensor) fwd.backward(dYf);
		DenseTensor dXb = (DenseTensor) bwd.backward(dYb);

		dXb = reverse(dXb);

		VectorUtils.enlarge(tmp_dX, dYf.sizeOfInnerVectors(), dXf.get(0).colSize());

		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dYf.size());

		int start = 0;

		for (int i = 0; i < dYf.size(); i++) {
			DenseMatrix dXfm = dXf.get(i);
			DenseMatrix dXbm = dXb.get(i);
			DenseMatrix dXm = tmp_dX.subMatrix(start, dXfm.rowSize());
			dXm.setAll(0);

			start += dXfm.rowSize();

			VectorMath.add(dXfm, dXbm, dXm);
			dX.add(dXm);
		}

		return dX;
	}

	@Override
	public RecurrentLayer copy() {
		return new BidirectionalRecurrentLayer((RecurrentLayer) fwd.copy(), (RecurrentLayer) bwd.copy());
	}

	@Override
	public Object forward(Object I) {
		DenseTensor Xf = (DenseTensor) I;
		DenseTensor Xb = reverse(Xf);

		DenseTensor Yf = (DenseTensor) fwd.forward(Xf);
		DenseTensor Yb = (DenseTensor) bwd.forward(Xb);

		Yb = reverse(Yb);

		DenseTensor Y = new DenseTensor();
		Y.ensureCapacity(Xf.size());

		VectorUtils.enlarge(tmp_Y, Xf.sizeOfInnerVectors(), output_size * 2);

		int start = 0;

		for (int i = 0; i < Yf.size(); i++) {
			DenseMatrix Yfm = Yf.get(i);
			DenseMatrix Ybm = Yb.get(i);
			DenseMatrix Ym = tmp_Y.subMatrix(start, Yfm.rowSize());
			Ym.setAll(0);

			start += Yfm.rowSize();

			for (int j = 0; j < Ym.rowSize(); j++) {
				DenseVector yfm = Yfm.row(j);
				DenseVector ybm = Ybm.row(j);
				DenseVector ym = Ym.row(j);
				VectorUtils.copyRows(new DenseMatrix(new DenseVector[] { yfm, ybm }), ym);
			}

			Y.add(Ym);
		}

		this.Y = Y;

		return Y;
	}

	private void addReversely(DenseMatrix Xf, DenseMatrix Xb, DenseMatrix X) {
		for (int i = 0; i < Xf.rowSize(); i++) {
			DenseVector xf = Xf.row(i);
			DenseVector xb = Xb.row(Xf.rowSize() - i - 1);
			DenseVector x = X.row(i);
			VectorMath.add(xf, xb, x);
		}
	}

	@Override
	public DenseTensor getB() {
		DenseTensor ret = new DenseTensor();
		ret.addAll(fwd.getB());
		ret.addAll(bwd.getB());
		return ret;
	}

	public RecurrentLayer getBackwardLayer() {
		return bwd;
	}

	@Override
	public DenseTensor getDB() {
		DenseTensor ret = new DenseTensor();
		ret.addAll(fwd.getDB());
		ret.addAll(bwd.getDB());
		return ret;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		ret.addAll(fwd.getDW());
		ret.addAll(bwd.getDW());
		return ret;
	}

	public RecurrentLayer getForwardLayer() {
		return fwd;
	}

	@Override
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.addAll(fwd.getW());
		ret.addAll(bwd.getW());
		return ret;
	}

	@Override
	public void initWeights(ParameterInitializer pi) {
		fwd.initWeights(pi);
		bwd.initWeights(pi);
	}

	@Override
	public void createGradientHolders() {
		fwd.createGradientHolders();
		bwd.createGradientHolders();
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		String name = ois.readUTF();
		Class c = Class.forName(name);

		if (c.getName().toLowerCase().contains("rnnlayer")) {
			fwd = (RnnLayer) c.newInstance();
			bwd = (RnnLayer) c.newInstance();
		} else if (c.getName().toLowerCase().contains("lstmlayer")) {
			fwd = (LstmLayer) c.newInstance();
			bwd = (LstmLayer) c.newInstance();
		}

		fwd.readObject(ois);
		bwd.readObject(ois);
		input_size = fwd.getInputSize();
		output_size = fwd.getOutputSize();
	}

	public void resetH0() {
		fwd.resetH0();
		bwd.resetH0();
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeUTF(fwd.getClass().getName());
		fwd.writeObject(oos);
		bwd.writeObject(oos);
	}

	private DenseTensor reverse(DenseTensor X, DenseMatrix T) {
		VectorUtils.enlarge(T, X.sizeOfInnerVectors(), X.get(0).colSize());

		DenseTensor R = new DenseTensor();
		R.ensureCapacity(X.size());

		int start = 0;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Rm = T.subMatrix(start, Xm.rowSize());
			start += Xm.rowSize();

			VectorUtils.copy(Xm, Rm);

			int mid = Xm.rowSize() / 2;

			for (int j = 0; j < mid; j++) {
				Rm.swapRows(j, Xm.rowSize() - 1 - j);
			}

			R.add(Rm);
		}

		return R;
	}

	private DenseTensor reverse(DenseTensor X) {
		DenseTensor R = new DenseTensor();
		R.ensureCapacity(X.size());

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Rm = new DenseMatrix();
			Rm.addAll(Xm);

			int mid = Xm.rowSize() / 2;

			for (int j = 0; j < mid; j++) {
				Rm.swapRows(j, Xm.rowSize() - 1 - j);
			}

			R.add(Rm);
		}
		return R;
	}

}
