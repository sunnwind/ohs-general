package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.utils.Generics;

public class ConcatenationLayer extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseTensor X;

	private DenseTensor Y;

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private List<DenseMatrix> tmp_dZs = Generics.newArrayList();

	private List<DenseTensor> Zs;

	private Map<Integer, Layer> L;

	private int output_size;

	private DenseTensor A;

	private int feat_size;

	public ConcatenationLayer(int feat_size, Map<Integer, Layer> L) {
		this.feat_size = feat_size;
		this.L = L;

		for (int i = 0; i < feat_size; i++) {
			Layer l = L.get(i);

			if (l == null) {
				output_size++;
			} else {
				output_size += l.getOutputSize();
			}
		}
	}

	public ConcatenationLayer(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	@Override
	public Object backward(Object I) {
		DenseTensor dY = (DenseTensor) I;

		if (tmp_dZs.size() == 0) {
			tmp_dZs = Generics.newArrayList(feat_size);
			for (int i = 0; i < feat_size; i++) {
				tmp_dZs.add(new DenseMatrix(0));
			}
		}

		{
			int row_size = dY.sizeOfInnerVectors();

			for (int i = 0; i < feat_size; i++) {
				DenseMatrix tmp_dZ = tmp_dZs.get(i);
				Layer l = L.get(i);

				if (l == null) {

				} else {
					VectorUtils.enlarge(tmp_dZ, row_size, l.getOutputSize());
				}
			}
		}

		for (int i = 0, s = 0; i < feat_size; i++) {
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
		Map<Integer, Layer> C = Generics.newHashMap(L.size());
		for (Entry<Integer, Layer> e : L.entrySet()) {
			int feat_idx = e.getKey();
			Layer l = e.getValue();
			C.put(feat_idx, l.copy());
		}

		ConcatenationLayer l = new ConcatenationLayer(feat_size, C);
		return l;
	}

	@Override
	public Object forward(Object I) {
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		List<DenseTensor> Zs = Generics.newArrayList(L.size());

		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), output_size);

		for (int i = 0; i < feat_size; i++) {
			Layer l = L.get(i);

			if (l == null) {
				Zs.add(null);
			} else {
				Zs.add((DenseTensor) l.forward(X));
			}
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
		for (Layer l : L.values()) {
			if (l.getB() != null) {
				B.addAll(l.getB());
			}
		}

		return B;
	}

	@Override
	public DenseTensor getDB() {
		DenseTensor dB = new DenseTensor();
		for (Layer l : L.values()) {
			if (l.getDB() != null) {
				dB.addAll(l.getDB());
			}
		}

		return dB;
	}

	@Override
	public DenseTensor getDW() {
		DenseTensor dW = new DenseTensor();
		for (Layer l : L.values()) {
			if (l.getDW() != null) {
				dW.addAll(l.getDW());
			}
		}

		return dW;
	}

	@Override
	public int getOutputSize() {
		return output_size;
	}

	@Override
	public DenseTensor getW() {
		DenseTensor W = new DenseTensor();
		for (Layer l : L.values()) {
			if (l.getW() != null) {
				W.addAll(l.getW());
			}
		}

		return W;
	}

	@Override
	public void initWeights() {
		for (Layer l : L.values()) {
			l.initWeights();
		}
	}

	@Override
	public void prepareTraining() {
		for (Layer l : L.values()) {
			l.prepareTraining();
		}
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		feat_size = ois.readInt();
		int size = ois.readInt();

		L = Generics.newHashMap(size);

		for (int i = 0; i < size; i++) {
			int feat_idx = ois.readInt();
			String name = ois.readUTF();

			Class c = Class.forName(name);
			Layer l = (Layer) c.getDeclaredConstructor().newInstance();
			l.readObject(ois);
			L.put(feat_idx, l);
		}
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(feat_size);
		oos.writeInt(L.size());
		for (Entry<Integer, Layer> e : L.entrySet()) {
			int feat_idx = e.getKey();
			Layer l = e.getValue();
			oos.writeInt(feat_idx);
			oos.writeUTF(l.getClass().getName());
			l.writeObject(oos);
		}
	}

}
