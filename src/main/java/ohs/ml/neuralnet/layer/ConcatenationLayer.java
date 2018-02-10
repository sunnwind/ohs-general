package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.utils.Generics;

public class ConcatenationLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseTensor X;

	private DenseTensor Y;

	private DenseTensor A;

	private List<DenseTensor> Zs;

	private List<EmbeddingLayer> L;

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private List<DenseMatrix> tmp_dZs = Generics.newArrayList();

	private int output_size;

	public ConcatenationLayer() {

	}

	public ConcatenationLayer(List<EmbeddingLayer> L) {
		this.L = L;
		for (EmbeddingLayer l : L) {
			output_size += l.getOutputSize();
		}
	}

	public ConcatenationLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;

		if (tmp_dZs.size() == 0) {
			tmp_dZs = Generics.newArrayList(L.size());
			for (int i = 0; i < L.size(); i++) {
				tmp_dZs.add(new DenseMatrix(0));
			}
		}

		{
			int row_size = dY.sizeOfInnerVectors();

			for (int i = 0; i < L.size(); i++) {
				DenseMatrix tmp_dZ = tmp_dZs.get(i);
				Layer l = L.get(i);

				if (l == null) {

				} else {
					VectorUtils.enlarge(tmp_dZ, row_size, l.getOutputSize());
				}
			}
		}

		for (int i = 0, s = 0; i < L.size(); i++) {
			Layer l = L.get(i);

			if (l == null) {
				s++;
			} else {
				int e = s + l.getOutputSize();
				int start = 0;

				DenseTensor dZ = new DenseTensor();
				dZ.ensureCapacity(dY.size());

				DenseMatrix tmp_dZ = tmp_dZs.get(i);

				for (DenseMatrix dYm : dY) {
					DenseMatrix dZm = tmp_dZ.subMatrix(start, dYm.rowSize());
					dZm.setAll(0);

					start += dYm.rowSize();

					for (int j = 0; j < dYm.rowSize(); j++) {
						DenseVector dym = dYm.row(j);
						DenseVector dzm = dZm.row(j);

						for (int k = s, m = 0; k < e; k++, m++) {
							dzm.add(m, dym.value(k));
						}
					}
					dZ.add(dZm);
				}
				s = e;
				l.backward(dZ);
			}
		}

		return null;
	}

	public ConcatenationLayer copy() {
		List<EmbeddingLayer> C = Generics.newArrayList(L.size());
		for (EmbeddingLayer l : L) {
			C.add(l.copy());
		}

		ConcatenationLayer l = new ConcatenationLayer(C);
		return l;
	}

	@Override
	public void createGradientHolders() {
		for (Layer l : L) {
			l.createGradientHolders();
		}
	}

	@Override
	public Object forward(Object I) {
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		List<DenseTensor> Zs = Generics.newArrayList(L.size());

		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), output_size);

		for (int i = 0; i < L.size(); i++) {
			Layer l = L.get(i);
			Zs.add((DenseTensor) l.forward(X));
		}

		int start = 0;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			for (int j = 0; j < Ym.rowSize(); j++) {
				DenseVector ym = Ym.row(j);
				DenseVector xm = Xm.row(j);

				for (int k = 0, m = 0; k < Zs.size(); k++) {
					DenseTensor Z = Zs.get(k);

					if (Z == null) {
						ym.add(m++, xm.value(k));
					} else {
						DenseVector zm = Z.get(i).row(j);
						for (int l = 0; l < zm.size(); l++) {
							ym.add(m++, zm.value(l));
						}
					}
				}
			}

			Ym.unwrapValues();
			Y.add(Ym);
		}

		this.X = X;
		this.Y = Y;
		this.Zs = Zs;

		return Y;

	}

	@Override
	public DenseTensor getB() {
		DenseTensor B = new DenseTensor();
		for (Layer l : L) {
			if (l.getB() != null) {
				B.addAll(l.getB());
			}
		}

		return B;
	}

	@Override
	public DenseTensor getDB() {
		DenseTensor dB = new DenseTensor();
		for (Layer l : L) {
			if (l.getDB() != null) {
				dB.addAll(l.getDB());
			}
		}

		return dB;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor dW = new DenseTensor();
		for (Layer l : L) {
			if (l.getDW() != null) {
				dW.addAll(l.getDW());
			}
		}

		return dW;
	}

	public List<EmbeddingLayer> getInnerLayers() {
		return L;
	}

	@Override
	public int getOutputSize() {
		return output_size;
	}

	@Override
	public DenseTensor getW() {
		DenseTensor W = new DenseTensor();
		for (Layer l : L) {
			if (l.getW() != null) {
				W.addAll(l.getW());
			}
		}

		return W;
	}

	@Override
	public void initWeights(ParameterInitializer pi) {
		for (Layer l : L) {
			l.initWeights(pi);
		}
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		L = Generics.newArrayList(size);

		output_size = 0;

		for (int i = 0; i < size; i++) {
			String name = ois.readUTF();
			Class c = Class.forName(name);
			EmbeddingLayer l = (EmbeddingLayer) c.getDeclaredConstructor().newInstance();
			l.readObject(ois);
			L.add(l);

			output_size += l.getOutputSize();
		}

	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(L.size());
		for (EmbeddingLayer l : L) {
			oos.writeUTF(l.getClass().getName());
			l.writeObject(oos);
		}
	}

}
