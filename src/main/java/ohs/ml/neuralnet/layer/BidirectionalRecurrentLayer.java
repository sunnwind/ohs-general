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

	private DenseMatrix H;

	private int input_size;

	private int output_size;

	private DenseMatrix tmp_dX = new DenseMatrix(0);

	private DenseMatrix tmp_H = new DenseMatrix(0);

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
		DenseMatrix dH = (DenseMatrix) I;
		int data_size = dH.rowSize();

		VectorUtils.enlarge(tmp_dX, data_size, input_size);

		Object I2 = reverse(I);
		DenseMatrix dX1 = (DenseMatrix) fwd.backward(I);
		DenseMatrix dX2 = (DenseMatrix) bwd.backward(I2);

		DenseMatrix dX = tmp_dX.rows(data_size);
		dX.setAll(0);

		for (int i = 0; i < data_size; i++) {
			DenseVector o1 = dX1.row(i);
			DenseVector o2 = dX2.row(data_size - i - 1);
			DenseVector o3 = dX.row(i);
			VectorMath.add(o1, o2, o3);
		}

		return dX;
	}

	@Override
	public RecurrentLayer copy() {
		return new BidirectionalRecurrentLayer((RecurrentLayer) fwd.copy(), (RecurrentLayer) bwd.copy());
	}

	@Override
	public Object forward(Object I) {
		int data_size = I instanceof IntegerArray ? ((IntegerArray) I).size() : ((DenseMatrix) I).rowSize();
		Object I2 = reverse(I);

		DenseMatrix O1 = (DenseMatrix) fwd.forward(I);
		DenseMatrix O2 = (DenseMatrix) bwd.forward(I2);

		VectorUtils.enlarge(tmp_H, data_size, output_size);

		H = tmp_H.rows(data_size);
		H.setAll(0);

		for (int i = 0; i < data_size; i++) {
			DenseVector o1 = O1.row(i);
			DenseVector o2 = O2.row(data_size - i - 1);
			DenseVector o3 = H.row(i);
			VectorMath.add(o1, o2, o3);
		}

		return H;
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

	private Object reverse(Object I) {
		Object ret = null;

		if (I instanceof IntegerArray) {
			IntegerArray x = (IntegerArray) I;
			int[] vs = ArrayUtils.copy(x.values());
			ArrayUtils.reverse(vs);
			ret = new IntegerArray(vs);
		} else if (I instanceof DenseMatrix) {
			DenseMatrix X = (DenseMatrix) I;
			int data_size = X.rowSize();
			List<DenseVector> R = Generics.newArrayList(data_size);
			for (int i = 0; i < X.rowSize(); i++) {
				R.add(X.row(data_size - i - 1));
			}
			ret = new DenseMatrix(R);
		}
		return ret;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeUTF(fwd.getClass().getName());
		fwd.writeObject(oos);
		bwd.writeObject(oos);
	}

}
