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
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

/**
 * 
 * Graves, A. (2012). Supervised Sequence Labelling with Recurrent Neural
 * Networks. http://doi.org/10.1007/978-3-642-24797-2
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

	public BidirectionalRecurrentLayer(Type type, int input_size, int hidden_size, int bptt_size, Nonlinearity non) {
		if (type == Type.RNN) {
			fwd = new RnnLayer(input_size, hidden_size, bptt_size, non);
			bwd = new RnnLayer(input_size, hidden_size, bptt_size, non);
		} else if (type == Type.LSTM) {
			fwd = new LstmLayer(input_size, hidden_size);
			bwd = new LstmLayer(input_size, hidden_size);
		}

		this.input_size = fwd.getInputSize();
		this.output_size = fwd.getOutputSize();
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dYf = (DenseTensor) I;
		DenseTensor dYb = reverse(dYf, tmp_rY);
		DenseTensor dXf = (DenseTensor) fwd.backward(dYf);
		DenseTensor dXb = (DenseTensor) bwd.backward(dYb);

		VectorUtils.enlarge(tmp_dX, dYf.sizeOfInnerVectors(), dXf.get(0).colSize());

		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dYf.size());

		int start = 0;

		for (int i = 0; i < dYf.size(); i++) {
			DenseMatrix dXfm = dXf.row(i);
			DenseMatrix dXbm = dXb.row(i);
			DenseMatrix dXm = tmp_dX.subMatrix(start, dXfm.rowSize());
			dXm.setAll(0);

			start += dXfm.rowSize();

			addReversely(dXfm, dXbm, dXm);

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
		DenseTensor Xb = reverse(Xf, tmp_rX);

		DenseTensor Yf = (DenseTensor) fwd.forward(Xf);
		DenseTensor Yb = (DenseTensor) bwd.forward(Xb);

		DenseTensor Y = new DenseTensor();
		Y.ensureCapacity(Xf.size());

		VectorUtils.enlarge(tmp_Y, Xf.sizeOfInnerVectors(), output_size);

		int start = 0;

		for (int i = 0; i < Yf.size(); i++) {
			DenseMatrix Yfm = Yf.row(i);
			DenseMatrix Ybm = Yb.row(i);
			DenseMatrix Ym = tmp_Y.subMatrix(start, Yfm.rowSize());
			Ym.setAll(0);

			start += Yfm.rowSize();

			addReversely(Yfm, Ybm, Ym);

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
	public void init() {
		fwd.init();
		bwd.init();
	}

	@Override
	public void prepare() {
		fwd.prepare();
		bwd.prepare();
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

}
