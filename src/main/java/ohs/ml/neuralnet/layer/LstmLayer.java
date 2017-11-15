package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;

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

	private DenseVector tmp;
	private DenseMatrix tmp_dC = new DenseMatrix(0);
	private DenseMatrix tmp_dX = new DenseMatrix(0);
	private DenseMatrix tmp_H = new DenseMatrix(0);
	private DenseMatrix tmp_C = new DenseMatrix(0);
	private DenseMatrix tmp_I = new DenseMatrix(0);
	private DenseMatrix tmp_F = new DenseMatrix(0);
	private DenseMatrix tmp_O = new DenseMatrix(0);
	private DenseMatrix tmp_G = new DenseMatrix(0);

	private DenseTensor X;
	private DenseTensor H;
	private DenseTensor C;
	private DenseTensor I;
	private DenseTensor F;
	private DenseTensor O;
	private DenseTensor G;

	private DenseVector z;
	private DenseVector h0;
	private DenseVector c0;

	private DenseVector dh_raw;
	private DenseVector dh_raw2;
	private DenseVector dh_next;
	private DenseVector dh_next2;

	private DenseVector dc0;
	private DenseVector df;
	private DenseVector dg;

	private DenseVector di;
	private DenseVector doo;

	private int hidden_size;
	private int input_size;

	public LstmLayer() {

	}

	public LstmLayer(DenseMatrix Wxh, DenseMatrix Whh, DenseVector bh, int shift_size, int window_size) {
		super();
		this.Wxh = Wxh;
		this.Whh = Whh;
		this.b = bh;
		this.shift_size = shift_size;
		this.window_size = window_size;

		input_size = Wxh.rowSize();
		hidden_size = Wxh.colSize() / 4;
	}

	public LstmLayer(int input_size, int hidden_size, int shift_size, int window_size) {
		this(new DenseMatrix(input_size, hidden_size * 4), new DenseMatrix(hidden_size, hidden_size * 4),
				new DenseVector(hidden_size * 4), shift_size, window_size);
	}

	@Override
	public Object backward(Object I) {
		return backward1(I);
	}

	public Object backward1(Object IN) {
		DenseTensor dH = (DenseTensor) IN;
		DenseTensor dX = new DenseTensor();
		dX.ensureCapacity(dH.size());

		VectorUtils.enlarge(tmp_dX, dH.sizeOfInnerVectors(), input_size);
		VectorUtils.enlarge(tmp_dC, dH.sizeOfInnerVectors(), hidden_size);

		if (dh_next == null) {
			dh_next = h0.copy(true);
			dh_raw = z.copy(true);

			dh_next2 = h0.copy(true);
			dh_raw2 = z.copy(true);

			dc0 = h0.copy(true);

			di = h0.copy(true);
			df = h0.copy(true);
			doo = h0.copy(true);
			dg = h0.copy(true);
		}

		int start = 0;

		for (int u1 = 0; u1 < dH.size(); u1++) {
			DenseMatrix dHm = dH.get(u1);
			DenseMatrix Xm = X.get(u1);
			DenseMatrix Hm = H.get(u1);
			DenseMatrix Cm = C.get(u1);
			DenseMatrix Im = I.get(u1);
			DenseMatrix Fm = F.get(u1);
			DenseMatrix Om = O.get(u1);
			DenseMatrix Gm = G.get(u1);

			DenseMatrix dXm = tmp_dX.subMatrix(start, dHm.rowSize());
			DenseMatrix dCm = tmp_dC.subMatrix(start, dHm.rowSize());
			start += dHm.rowSize();

			dXm.setAll(0);
			dCm.setAll(0);

			dh_next.setAll(0);
			dc0.setAll(0);

			tmp.setAll(0);

			for (int u2 = dHm.rowSize() - 1; u2 >= 0;) {
				int end = Math.max(u2 - window_size, 0);

				dh_next.setAll(0);
				dc0.setAll(0);

				for (int u3 = u2; u3 > end; u3--) {
					{
						DenseVector dh = dHm.row(u3);
						VectorMath.add(dh_next, dh);

						DenseVector c = Cm.row(u3);
						VectorMath.multiply(c, dh, doo);

						DenseVector dc = dCm.row(u3);
						DenseVector o = Om.row(u3);

						VectorMath.tanhGradient(c, tmp);
						VectorMath.multiply(tmp, o, tmp);
						VectorMath.multiply(tmp, dh, tmp);
						VectorMath.add(tmp, dc);

						DenseVector c_prev = u3 == 0 ? c0 : Cm.row(u3 - 1);
						VectorMath.multiply(c_prev, dc, df);

						DenseVector dc_prev = u3 == 0 ? dc0 : dCm.row(u3 - 1);
						DenseVector f = Fm.row(u3);
						VectorMath.multiply(f, dc, tmp);
						VectorMath.add(tmp, dc_prev);

						DenseVector g = Gm.row(u3);
						VectorMath.multiply(g, dc, di);

						DenseVector i = Im.row(u3);
						VectorMath.multiply(i, dc, dg);

						VectorMath.sigmoidGradient(i, tmp);
						VectorMath.multiply(tmp, di, di);

						VectorMath.sigmoidGradient(f, tmp);
						VectorMath.multiply(tmp, df, df);

						VectorMath.sigmoidGradient(o, tmp);
						VectorMath.multiply(tmp, doo, doo);

						VectorMath.tanhGradient(g, tmp);
						VectorMath.multiply(tmp, dg, dg);

						VectorUtils.copyRows(new DenseMatrix(new DenseVector[] { di, df, doo, dg }), dh_raw);

						VectorMath.add(dh_raw, db);

						VectorMath.productRows(dh_raw.toDenseMatrix(), Whh, dh_next.toDenseMatrix(), false);
					}

					DenseVector h_prev = u3 == 0 ? h0 : Hm.row(u3 - 1);
					VectorMath.outerProduct(h_prev, dh_raw, dWhh, true);

					DenseVector x = Xm.row(u3);
					VectorMath.outerProduct(x, dh_raw, dWxh, true);

					DenseVector dx = dXm.row(u3);
					VectorMath.productRows(dh_raw.toDenseMatrix(), Wxh, dx.toDenseMatrix(), true);
				}

				if (u2 == 0) {
					break;
				}

				u2 = Math.max(u2 - shift_size, 0);
			}

			dX.add(dXm);
		}

		return dX;
	}

	@Override
	public Layer copy() {
		return new LstmLayer(Wxh, Whh, b, shift_size, window_size);
	}

	@Override
	public DenseTensor forward(Object IN) {
		this.X = (DenseTensor) IN;

		DenseTensor X = (DenseTensor) IN;
		DenseTensor H = new DenseTensor();
		DenseTensor C = new DenseTensor();
		DenseTensor I = new DenseTensor();
		DenseTensor F = new DenseTensor();
		DenseTensor O = new DenseTensor();
		DenseTensor G = new DenseTensor();

		H.ensureCapacity(X.size());
		C.ensureCapacity(X.size());
		I.ensureCapacity(X.size());
		F.ensureCapacity(X.size());
		O.ensureCapacity(X.size());
		G.ensureCapacity(X.size());

		{
			int size = X.sizeOfInnerVectors();
			VectorUtils.enlarge(tmp_H, size, hidden_size);
			VectorUtils.enlarge(tmp_C, size, hidden_size);
			VectorUtils.enlarge(tmp_I, size, hidden_size);
			VectorUtils.enlarge(tmp_F, size, hidden_size);
			VectorUtils.enlarge(tmp_O, size, hidden_size);
			VectorUtils.enlarge(tmp_G, size, hidden_size);
		}

		if (h0 == null) {
			h0 = new DenseVector(hidden_size);
			c0 = h0.copy(true);
			tmp = h0.copy(true);

			z = new DenseVector(hidden_size * 4);
		}

		int start = 0;

		for (int u = 0; u < X.size(); u++) {
			DenseMatrix Xm = X.get(u);

			DenseMatrix Hm = tmp_H.subMatrix(start, Xm.rowSize());
			DenseMatrix Cm = tmp_C.subMatrix(start, Xm.rowSize());
			DenseMatrix Im = tmp_I.subMatrix(start, Xm.rowSize());
			DenseMatrix Fm = tmp_F.subMatrix(start, Xm.rowSize());
			DenseMatrix Om = tmp_O.subMatrix(start, Xm.rowSize());
			DenseMatrix Gm = tmp_G.subMatrix(start, Xm.rowSize());

			Hm.setAll(0);
			Cm.setAll(0);
			Im.setAll(0);
			Fm.setAll(0);
			Om.setAll(0);
			Gm.setAll(0);

			start += Xm.rowSize();

			for (int t = 0; t < Xm.rowSize(); t++) {
				DenseVector x = Xm.row(t);
				DenseVector h_prev = t == 0 ? h0 : Hm.row(t - 1);

				VectorMath.product(x, Wxh, z, false);
				VectorMath.product(h_prev, Whh, z, true);
				VectorMath.add(b, z);

				DenseVector i = Im.row(t);
				DenseVector f = Fm.row(t);
				DenseVector o = Om.row(t);
				DenseVector g = Gm.row(t);

				for (int j = 0; j < hidden_size; j++) {
					i.set(j, z.value(j));
					f.set(j, z.value(hidden_size + j));
					o.set(j, z.value(hidden_size * 2 + j));
					g.set(j, z.value(hidden_size * 3 + j));
				}

				VectorMath.sigmoid(i, i);
				VectorMath.sigmoid(f, f);
				VectorMath.sigmoid(o, o);
				VectorMath.tanh(g, g);

				DenseVector c = Cm.row(t);
				VectorMath.multiply(i, g, tmp);
				VectorUtils.copy(tmp, c);

				DenseVector c_prev = t == 0 ? c0 : Cm.row(t - 1);
				VectorMath.multiply(f, c_prev, tmp);
				VectorMath.add(tmp, c);

				VectorMath.tanh(c, c);

				DenseVector h = Hm.row(t);
				VectorMath.multiply(o, c, h);
			}

			// VectorUtils.copy(h0, h0_prev);
			// VectorUtils.copy(Hm.row(Xm.rowSize() - 1), h0);
			//
			// VectorUtils.copy(c0, c0_prev);
			// VectorUtils.copy(Cm.row(Xm.rowSize() - 1), c0);

			H.add(Hm);
			C.add(Cm);
			I.add(Im);
			F.add(Fm);
			O.add(Om);
			G.add(Gm);
		}

		this.H = H;
		this.C = C;
		this.I = I;
		this.F = F;
		this.O = O;
		this.G = G;

		return H;
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
		db = b.copy(true);
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		Wxh = new DenseMatrix(ois);
		Whh = new DenseMatrix(ois);
		b = new DenseVector(ois);
		window_size = ois.readInt();

		input_size = Wxh.rowSize();
		hidden_size = Wxh.colSize() / 4;
	}

	public void resetH0() {
		h0.setAll(0);
		c0.setAll(0);
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		Wxh.writeObject(oos);
		Whh.writeObject(oos);
		b.writeObject(oos);
		oos.writeInt(window_size);
	}

}
