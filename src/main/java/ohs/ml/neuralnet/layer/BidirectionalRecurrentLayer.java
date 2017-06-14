package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

/**
 * 
 * Graves, A. (2012). Supervised Sequence Labelling with Recurrent Neural Networks. http://doi.org/10.1007/978-3-642-24797-2
 * 
 * @author ohs
 */
public class BidirectionalRecurrentLayer extends Layer {

	public static enum Type {
		RNN, LSTM
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -8764635726766071893L;

	private Layer fwd;

	private Layer bwd;

	private DenseMatrix tmp_H;

	private DenseMatrix tmp_dX;

	private DenseMatrix H;

	private int input_size;

	private int output_size;

	public BidirectionalRecurrentLayer(Layer fwd, Layer bwd) {
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
			fwd = new LstmLayer(input_size, hidden_size, non);
			bwd = new LstmLayer(input_size, hidden_size, non);
		}

		this.input_size = fwd.getInputSize();
		this.output_size = fwd.getOutputSize();
	}

	@Override
	public Object backward(Object I) {
		DenseMatrix dH = (DenseMatrix) I;
		int data_size = dH.rowSize();

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = new DenseMatrix(data_size, input_size);
		}

		Object I2 = reverse(I);
		DenseMatrix dX1 = (DenseMatrix) fwd.backward(I);
		DenseMatrix dX2 = (DenseMatrix) bwd.backward(I2);

		DenseMatrix dX = tmp_dX.rowsAsMatrix(data_size);
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
	public Object forward(Object I) {
		int data_size = I instanceof IntegerArray ? ((IntegerArray) I).size() : ((DenseMatrix) I).rowSize();
		Object I2 = reverse(I);

		DenseMatrix O1 = (DenseMatrix) fwd.forward(I);
		DenseMatrix O2 = (DenseMatrix) bwd.forward(I2);

		if (tmp_H == null || tmp_H.rowSize() < data_size) {
			tmp_H = new DenseMatrix(data_size, output_size);
		}

		H = tmp_H.rowsAsMatrix(data_size);
		H.setAll(0);

		for (int i = 0; i < data_size; i++) {
			DenseVector o1 = O1.row(i);
			DenseVector o2 = O2.row(data_size - i - 1);
			DenseVector o3 = H.row(i);
			
			try {
				VectorMath.add(o1, o2, o3);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		if (data_size != H.size()) {
			System.out.println();
		}

		return H;
	}

	public Layer getBackwardLayer() {
		return bwd;
	}

	public Layer getForwardLayer() {
		return fwd;
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
		fwd = (Layer) c.newInstance();
		bwd = (Layer) c.newInstance();

		fwd.readObject(ois);
		bwd.readObject(ois);
		input_size = fwd.getInputSize();
		output_size = fwd.getOutputSize();
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
