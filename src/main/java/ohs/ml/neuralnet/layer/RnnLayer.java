package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

/**
 *
 * 
 * Sutskever, I. (2013). Training Recurrent neural Networks. PhD thesis. University of Toronto.
 * 
 * https://jramapuram.github.io/ramblings/rnn-backrpop/
 * 
 * http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 * 
 * https://gist.github.com/karpathy/d4dee566867f8291f086
 * 
 * http://r2rt.com/styles-of-truncated-backpropagation.html
 * 
 * http://stats.stackexchange.com/questions/219914/rnns-when-to-apply-bptt-and-or-update-weights
 * 
 * http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
 * 
 * @author ohs
 */
public class RnnLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2848535801786544922L;

	/**
	 * input to hidden
	 */
	private DenseMatrix Wxh;

	/**
	 * hidden to hidden
	 */
	private DenseMatrix Whh;

	private DenseMatrix dWxh;

	private DenseMatrix dWhh;

	private Object fwd_I;

	private int bptt_size = 5;

	private Nonlinearity non;

	private DenseVector b;

	private DenseVector db;

	private DenseVector dh_raw;

	private DenseVector dh_prev;

	private DenseMatrix tmp_H;

	private DenseMatrix tmp_dX;

	private int input_size;

	private int hidden_size;

	private DenseVector h0;

	private DenseMatrix H;

	private DenseVector h0_prev;

	private DenseMatrix tmp_dA;

	private DenseVector tmp;

	public RnnLayer(DenseMatrix Wxh, DenseMatrix Whh, DenseVector bh, int bptt_size, Nonlinearity non) {
		super();
		this.Wxh = Wxh;
		this.Whh = Whh;
		this.b = bh;
		this.bptt_size = bptt_size;
		this.non = non;

		this.input_size = Wxh.rowSize();
		this.hidden_size = Wxh.colSize();
	}

	public RnnLayer(int input_size, int hidden_size, int bptt_size, Nonlinearity non) {
		this(new DenseMatrix(input_size, hidden_size), new DenseMatrix(hidden_size, hidden_size), new DenseVector(hidden_size), bptt_size,
				non);
	}

	@Override
	public Object backward(Object I) {
		return backward2(I);
	}

	/**
	 * Sutskever, I. (2013). Training Recurrent neural Networks. PhD thesis. University of Toronto.
	 * 
	 * 
	 * http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
	 * 
	 * 
	 * 
	 * @param I
	 * @return
	 */
	public Object backward2(Object I) {
		DenseMatrix dH = (DenseMatrix) I;
		int data_size = dH.rowSize();

		if (dh_raw == null) {
			dh_raw = new DenseVector(hidden_size);
			dh_prev = dh_raw.copy(true);
			tmp = dh_raw.copy(true);
		}

		dh_prev.setAll(0);

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = new DenseMatrix(data_size, input_size);
			tmp_dA = new DenseMatrix(data_size, hidden_size);
		}

		DenseMatrix dX = tmp_dX.rows(data_size);
		DenseMatrix dA = tmp_dA.rows(data_size);

		non.backward(H, dA);

		for (int t = data_size - 1; t >= 0; t--) {
			DenseVector dh = dH.row(t);

			VectorMath.add(dh_prev, dh);

			VectorUtils.copy(dA.row(t), dh_raw);

			VectorMath.multiply(dh, dh_raw, dh_raw);

			VectorMath.productRows(dh_raw.toDenseMatrix(), Whh, dh_prev.toDenseMatrix(), false);

			VectorUtils.copy(dh_raw, tmp);

			int size = Math.max(0, t - bptt_size);

			for (int tt = t; tt >= size; tt--) {
				if (fwd_I instanceof IntegerArray) {
					IntegerArray X = ((IntegerArray) fwd_I);
					int idx = X.get(tt);
					DenseVector dwx = dWxh.row(idx);
					VectorMath.add(tmp, dwx);
				} else {
					DenseMatrix X = ((DenseMatrix) fwd_I);
					DenseVector x = X.row(tt);
					VectorMath.outerProduct(x, tmp, dWxh, true);

					DenseVector dx = dX.row(tt);
					VectorMath.productRows(tmp.toDenseMatrix(), Wxh, dx.toDenseMatrix(), false);
				}

				DenseVector h_prev = tt == 0 ? h0_prev : H.row(tt - 1);

				VectorMath.outerProduct(h_prev, tmp, dWhh, true);

				VectorMath.add(tmp, db);

				VectorMath.productRows(tmp.toDenseMatrix(), Whh, tmp.toDenseMatrix(), false);

				DenseVector da = tt == 0 ? dh_prev : dA.row(tt - 1);

				VectorMath.multiply(da, tmp, tmp);
			}
		}
		return dX;
	}

	@Override
	public DenseMatrix forward(Object I) {
		int data_size = I instanceof IntegerArray ? ((IntegerArray) I).size() : ((DenseMatrix) I).rowSize();

		if (tmp_H == null || tmp_H.rowSize() < data_size) {
			tmp_H = new DenseMatrix(data_size, hidden_size);
		}

		if (h0 == null) {
			h0 = new DenseVector(hidden_size);
			h0_prev = h0.copy(true);
		}

		H = tmp_H.rows(data_size);
		H.setAll(0);

		for (int t = 0; t < data_size; t++) {
			DenseVector h = H.row(t);
			DenseVector h_prev = t == 0 ? h0 : H.row(t - 1);

			VectorMath.product(h_prev, Whh, h, false);

			if (I instanceof IntegerArray) {
				int w = ((IntegerArray) I).get(t);
				DenseVector x = Wxh.row(w);
				VectorMath.add(x, h);
			} else {
				DenseVector x = ((DenseMatrix) I).row(t);
				VectorMath.product(x, Wxh, h, true);
			}

			VectorMath.add(b, h);

			non.forward(h.toDenseMatrix(), h.toDenseMatrix());
		}

		VectorUtils.copy(h0, h0_prev);
		VectorUtils.copy(H.row(data_size - 1), h0);

		fwd_I = I;

		return H;
	}

	@Override
	public DenseMatrix getB() {
		return new DenseMatrix(new DenseVector[] { b });
	}

	public DenseVector getBh() {
		return b;
	}

	public int getBpttSize() {
		return bptt_size;
	}

	@Override
	public DenseMatrix getDB() {
		return new DenseMatrix(new DenseVector[] { db });
	}

	@Override
	public DenseMatrix getDW() {
		int size = dWxh.rowSize() + dWhh.rowSize();
		List<DenseVector> ws = Generics.newArrayList(size);

		for (DenseVector w : dWxh) {
			ws.add(w);
		}

		for (DenseVector w : dWhh) {
			ws.add(w);
		}
		return new DenseMatrix(ws);
	}

	public DenseVector getH0() {
		return h0;
	}

	@Override
	public int getInputSize() {
		return Wxh.rowSize();
	}

	public Nonlinearity getNonlinearity() {
		return non;
	}

	@Override
	public int getOutputSize() {
		return hidden_size;
	}

	@Override
	public DenseMatrix getW() {
		int size = Wxh.rowSize() + Whh.rowSize();
		List<DenseVector> ws = Generics.newArrayList(size);

		for (DenseVector w : Wxh) {
			ws.add(w);
		}

		for (DenseVector w : Whh) {
			ws.add(w);
		}

		return new DenseMatrix(ws);
	}

	public DenseMatrix getWhh() {
		return Whh;
	}

	public DenseMatrix getWxh() {
		return Wxh;
	}

	@Override
	public void init() {
		ParameterInitializer.init2(Wxh);
		VectorMath.identity(Whh, 1);
	}

	@Override
	public void prepare() {
		dWxh = Wxh.copy(true);
		dWhh = Whh.copy(true);
		db = b.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		Wxh = new DenseMatrix(ois);
		Whh = new DenseMatrix(ois);
		b = new DenseVector(ois);
	}

	public void resetH0() {
		h0.setAll(0);
	}

	public void setBpptSize(int bppt_size) {
		this.bptt_size = bppt_size;
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		Wxh.writeObject(oos);
		Whh.writeObject(oos);
		b.writeObject(oos);
	}

}
