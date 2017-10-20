package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.nonlinearity.Nonlinearity;
import ohs.utils.Generics;

/**
 *
 * 
 * Sutskever, I. (2013). Training Recurrent neural Networks. PhD thesis.
 * University of Toronto.
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
 * https://github.com/cthorey/CS231/blob/master/assignment3/cs231n/rnn_layers.py
 * 
 * https://github.com/bagavi/CS231N/blob/master/assignment3/cs231n/rnn_layers.py
 *
 * https://gist.github.com/karpathy/587454dc0146a6ae21fc
 * 
 * @author ohs
 */
public class LstmLayer extends RecurrentLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2848535801786544922L;

	private DenseVector a;
	private DenseVector bh;
	private DenseMatrix C;

	private DenseVector c0;
	private DenseVector c0_prev;
	private DenseVector da;

	private DenseVector dbh;

	private DenseVector dc_prev;
	private DenseVector dc0;

	private DenseVector df;
	private DenseVector dg;
	private DenseVector dh_prev;
	private DenseVector di;

	private DenseVector doo;

	private DenseMatrix F;
	private DenseMatrix G;
	private DenseMatrix H;
	private DenseVector h0;
	private DenseVector h0_prev;

	private int hidden_size;
	private DenseMatrix I;
	private int input_size;
	private Nonlinearity non;
	private DenseMatrix O;
	private DenseVector tmp;
	private DenseMatrix tmp_C;
	private DenseMatrix tmp_dC;

	private DenseMatrix tmp_dX;
	private DenseMatrix tmp_F;
	private DenseMatrix tmp_G;
	private DenseMatrix tmp_H;
	private DenseMatrix tmp_I;

	private DenseMatrix tmp_O;
	/**
	 * input to hidden
	 */
	private DenseMatrix X;

	public LstmLayer() {

	}

	public LstmLayer(DenseMatrix Wxh, DenseMatrix Whh, DenseVector bh, Nonlinearity non) {
		super();
		this.Wxh = Wxh;
		this.Whh = Whh;
		this.bh = bh;
		this.non = non;

		input_size = Wxh.rowSize();
		hidden_size = Wxh.colSize() / 4;
	}

	public LstmLayer(int input_size, int hidden_size, Nonlinearity non) {
		this(new DenseMatrix(input_size, hidden_size * 4), new DenseMatrix(hidden_size, hidden_size * 4),
				new DenseVector(hidden_size * 4), non);
	}

	@Override
	public Object backward(Object I) {
		return backward2(I);
	}

	public Object backward2(Object IN) {
		DenseMatrix dH = (DenseMatrix) IN;
		int data_size = dH.rowSize();

		if (tmp_dX == null || tmp_dX.rowSize() < data_size) {
			tmp_dX = new DenseMatrix(data_size, input_size);
			tmp_dC = dH.copy(true);
		}

		if (dh_prev == null) {
			dh_prev = h0.copy(true);
			dc_prev = h0.copy(true);
			dc0 = h0.copy(true);

			di = h0.copy(true);
			df = h0.copy(true);
			doo = h0.copy(true);
			dg = h0.copy(true);
		}

		dh_prev.setAll(0);
		dc_prev.setAll(0);
		dc0.setAll(0);

		DenseMatrix dX = tmp_dX.rows(data_size);
		DenseMatrix dC = tmp_dC.rows(data_size);

		dC.setAll(0);

		for (int t = data_size - 1; t >= 0; t--) {
			DenseVector dh = dH.row(t);
			VectorMath.add(dh, dh_prev, dh);

			DenseVector c = C.row(t);

			VectorMath.multiply(c, dh, doo);

			DenseVector dc = dC.row(t);
			DenseVector o = O.row(t);
			VectorMath.tanhGradient(c, tmp);
			VectorMath.multiply(tmp, o, tmp);
			VectorMath.multiply(tmp, dh, tmp);
			VectorMath.add(tmp, dc);

			DenseVector c_prev = t == 0 ? c0_prev : C.row(t - 1);
			VectorMath.multiply(c_prev, dc, df);

			DenseVector dc_prev = t == 0 ? dc0 : dC.row(t - 1);
			DenseVector f = F.row(t);
			VectorMath.multiply(f, dc, tmp);
			VectorMath.add(tmp, dc_prev);

			DenseVector g = G.row(t);
			VectorMath.multiply(g, dc, di);

			DenseVector i = I.row(t);
			VectorMath.multiply(i, dc, dg);

			VectorMath.sigmoidGradient(i, tmp);
			VectorMath.multiply(tmp, di, di);

			VectorMath.sigmoidGradient(f, tmp);
			VectorMath.multiply(tmp, df, df);

			VectorMath.sigmoidGradient(o, tmp);
			VectorMath.multiply(tmp, doo, doo);

			VectorMath.tanhGradient(g, tmp);
			VectorMath.multiply(tmp, dg, dg);

			VectorUtils.copyRows(new DenseMatrix(new DenseVector[] { di, df, doo, dg }), da);

			DenseVector dx = dX.row(t);
			VectorMath.productRows(da.toDenseMatrix(), Wxh, dx.toDenseMatrix(), false);

			DenseVector x = X.row(t);
			VectorMath.outerProduct(x, da, dWxh, true);

			VectorMath.productRows(da.toDenseMatrix(), Whh, dh_prev.toDenseMatrix(), false);

			// VectorMath.productColumns(Whh, da, dh_prev);

			DenseVector h_prev = t == 0 ? h0_prev : H.row(t - 1);

			VectorMath.outerProduct(h_prev, da, dWhh, true);

			VectorMath.add(da, dbh);
		}
		return dX;
	}

	@Override
	public Layer copy() {
		return new LstmLayer(Wxh, Whh, bh, non);
	}

	@Override
	public DenseMatrix forward(Object IN) {
		X = (DenseMatrix) IN;
		int data_size = X.rowSize();

		if (tmp_H == null || tmp_H.rowSize() < data_size) {
			tmp_H = new DenseMatrix(data_size, hidden_size);
			tmp_I = tmp_H.copy(true);
			tmp_F = tmp_H.copy(true);
			tmp_O = tmp_H.copy(true);
			tmp_G = tmp_H.copy(true);
			tmp_C = tmp_H.copy(true);
		}

		if (h0 == null) {
			h0 = new DenseVector(hidden_size);
			h0_prev = h0.copy(true);

			c0 = h0.copy(true);
			c0_prev = h0.copy(true);
			tmp = h0.copy(true);

			a = new DenseVector(hidden_size * 4);
			da = a.copy(true);
		}

		// if (is_testing) {
		// resetH0();
		// }

		H = tmp_H.rows(data_size);
		C = tmp_C.rows(data_size);

		I = tmp_I.rows(data_size);
		F = tmp_F.rows(data_size);
		O = tmp_O.rows(data_size);
		G = tmp_G.rows(data_size);

		for (int t = 0; t < data_size; t++) {
			DenseVector x = X.row(t);
			DenseVector h_prev = t == 0 ? h0 : H.row(t - 1);

			VectorMath.product(x, Wxh, a, false);
			VectorMath.product(h_prev, Whh, a, true);
			VectorMath.add(bh, a);

			DenseVector i = I.row(t);
			DenseVector f = F.row(t);
			DenseVector o = O.row(t);
			DenseVector g = G.row(t);

			for (int j = 0; j < hidden_size; j++) {
				i.set(j, a.value(j));
				f.set(j, a.value(hidden_size + j));
				o.set(j, a.value(hidden_size * 2 + j));
				g.set(j, a.value(hidden_size * 3 + j));
			}

			VectorMath.sigmoid(i, i);
			VectorMath.sigmoid(f, f);
			VectorMath.sigmoid(o, o);
			VectorMath.tanh(g, g);

			DenseVector c = C.row(t);
			VectorMath.multiply(i, g, tmp);
			VectorUtils.copy(tmp, c);

			DenseVector c_prev = t == 0 ? c0 : C.row(t - 1);
			VectorMath.multiply(f, c_prev, tmp);
			VectorMath.add(tmp, c);

			VectorMath.tanh(c, c);

			DenseVector h = H.row(t);
			VectorMath.multiply(o, c, h);
		}

		VectorUtils.copy(h0, h0_prev);
		VectorUtils.copy(H.row(data_size - 1), h0);

		VectorUtils.copy(c0, c0_prev);
		VectorUtils.copy(C.row(data_size - 1), c0);

		return H;
	}

	@Override
	public DenseTensor getB() {
		DenseTensor ret = new DenseTensor();
		ret.add(bh.toDenseMatrix());
		return ret;
	}

	@Override
	public DenseTensor getDB() {
		DenseTensor ret = new DenseTensor();
		ret.add(dbh.toDenseMatrix());
		return ret;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor ret = new DenseTensor();
		ret.add(dWxh);
		ret.add(dWhh);
		return ret;
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
	public DenseTensor getW() {
		DenseTensor ret = new DenseTensor();
		ret.add(Wxh);
		ret.add(Whh);
		return ret;
	}

	@Override
	public void init() {
		ParameterInitializer.init2(Wxh);
		// VectorMath.identity(Whh, 1);

		for (int i = 0; i < hidden_size; i++) {
			Whh.set(i, i, 1);
			Whh.set(i, hidden_size + i, 1);
			Whh.set(i, hidden_size * 2 + i, 1);
			Whh.set(i, hidden_size * 3 + i, 1);
		}
	}

	@Override
	public void prepare() {
		dWxh = Wxh.copy(true);
		dWhh = Whh.copy(true);
		dbh = bh.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		Wxh = new DenseMatrix(ois);
		Whh = new DenseMatrix(ois);
		bh = new DenseVector(ois);
	}

	public void resetH0() {
		h0.setAll(0);
		c0.setAll(0);
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		Wxh.writeObject(oos);
		Whh.writeObject(oos);
		bh.writeObject(oos);
	}

}
