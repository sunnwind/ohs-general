package ohs.ml.neuralnet.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import org.apache.lucene.util.ArrayUtil;

import ohs.io.FileUtils;
import ohs.math.ArrayUtils;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseTensor;
import ohs.matrix.DenseVector;
import ohs.ml.neuralnet.com.NeuralNet;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.neuralnet.com.TaskType;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class ConcatenationLayerCharNN extends Layer {

	/**
	 * 
	 */
	private static final long serialVersionUID = -428617482088657354L;

	private DenseTensor X;

	private DenseTensor Y;

	private DenseMatrix tmp_Y = new DenseMatrix(0);

	private List<DenseMatrix> tmp_dZs = Generics.newArrayList();

	private List<DenseTensor> Zs;

	private List<Layer> L;

	private NeuralNet nn;

	private int output_size;

	private DenseTensor A;

	public ConcatenationLayerCharNN(List<Layer> L, NeuralNet nn) {
		this.L = L;
		this.nn = nn;
		for (Layer l : L) {
			output_size += l.getOutputSize();
		}

		if (nn != null) {
			output_size += nn.getOutputSize();
		}
	}

	public ConcatenationLayerCharNN(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	private DenseMatrix tmp_dT = new DenseMatrix(0);

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

			for (int i = 0; i < tmp_dZs.size(); i++) {
				DenseMatrix tmp_dZ = tmp_dZs.get(i);
				VectorUtils.enlarge(tmp_dZ, row_size, L.get(i).getOutputSize());
			}

			if (nn != null) {
				VectorUtils.enlarge(tmp_dT, row_size, nn.getOutputSize());
			}
		}

		for (int i = 0, s = 0; i < L.size(); i++) {
			Layer l = L.get(i);
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

		if (nn != null) {
			int s = output_size - nn.getOutputSize();

			DenseTensor T = this.T;
			DenseTensor A = this.A;
			DenseTensor X = this.X;
			DenseTensor dA = new DenseTensor();
			dA.ensureCapacity(dY.sizeOfInnerVectors());

			int num_chrs = 0;

			for (int i = 0; i < X.size(); i++) {
				DenseMatrix Xm = X.get(i);
				DenseMatrix dYm = dY.get(i);
				DenseMatrix Tm = T.get(i);

				for (int j = 0; j < Xm.rowSize(); j++) {
					DenseVector xm = Xm.row(j);
					DenseVector dym = dYm.row(j);
					dym = dym.subVector(s, output_size - s);

					int start = L.size();
					int end = xm.size();
					int size = end - start;
					num_chrs += size;

					dA.add(dym.toDenseMatrix());
				}
			}
			nn.backward(dA);
		}

		return null;
	}

	public ConcatenationLayerCharNN copy() {
		List<Layer> C = Generics.newArrayList(L.size());
		for (Layer l : L) {
			C.add(l.copy());
		}
		ConcatenationLayerCharNN l = new ConcatenationLayerCharNN(C, nn == null ? null : nn.copy());
		return l;
	}

	private DenseTensor T;

	@Override
	public Object forward(Object I) {
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		List<DenseTensor> Zs = Generics.newArrayList(nn == null ? L.size() : L.size() + 1);

		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), output_size);

		for (Layer l : L) {
			Zs.add((DenseTensor) l.forward(X));
		}

		if (nn != null) {
			DenseTensor A = new DenseTensor();
			A.ensureCapacity(X.sizeOfInnerVectors());

			for (DenseMatrix Xm : X) {
				for (DenseVector xm : Xm) {
					int start = L.size();
					int end = xm.size();
					int size = end - start;
					DenseMatrix Am = new DenseMatrix();
					Am.ensureCapacity(size);

					for (int i = start; i < end; i++) {
						Am.add(new DenseVector(new double[] { xm.value(i) }));
					}
					A.add(Am);
				}
			}

			DenseTensor T = (DenseTensor) nn.forward(A);
			DenseTensor Z = new DenseTensor();
			Z.ensureCapacity(T.sizeOfInnerVectors());

			for (int i = 0, k = 0; i < X.size(); i++) {
				DenseMatrix Xm = X.get(i);
				DenseMatrix Zm = new DenseMatrix();
				Zm.ensureCapacity(Xm.rowSize());

				for (int j = 0; j < Xm.rowSize(); j++) {
					Zm.add(T.get(k++).toDenseVector());
				}

				Z.add(Zm);
			}

			Zs.add(Z);

			this.A = A;
			this.T = T;
		}

		int start = 0;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			for (int j = 0; j < Ym.rowSize(); j++) {
				DenseVector ym = Ym.row(j);

				for (int k = 0, m = 0; k < Zs.size(); k++) {
					DenseVector zm = Zs.get(k).get(i).row(j);
					for (int l = 0; l < zm.size(); l++) {
						ym.add(m++, zm.value(l));
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

		if (nn != null) {
			for (Layer l : nn) {
				if (l.getB() != null) {
					B.addAll(l.getB());
				}
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

		if (nn != null) {
			for (Layer l : nn) {
				if (l.getDB() != null) {
					dB.addAll(l.getDB());
				}
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

		if (nn != null) {
			for (Layer l : nn) {
				if (l.getDW() != null) {
					dW.addAll(l.getDW());
				}
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
		for (Layer l : L) {
			if (l.getW() != null) {
				W.addAll(l.getW());
			}
		}

		if (nn != null) {
			for (Layer l : nn) {
				if (l.getW() != null) {
					W.addAll(l.getW());
				}
			}
		}
		return W;
	}

	@Override
	public void initWeights(ParameterInitializer pi) {
		for (Layer l : L) {
			l.initWeights(pi);
		}

		if (nn != null) {
			nn.initWeights(new ParameterInitializer());
		}
	}

	@Override
	public void createGradientHolders() {
		for (Layer l : L) {
			l.createGradientHolders();
		}

		if (nn != null) {
			nn.createGradientHolders();
		}
	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();

		L = Generics.newArrayList(size);

		for (int i = 0; i < size; i++) {
			String name = ois.readUTF();
			Class c = Class.forName(name);
			Layer l = (Layer) c.getDeclaredConstructor().newInstance();
			l.readObject(ois);
			L.add(l);
		}
	}

	public void setOutputWordIndexes(boolean output_word_indexes) {
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(L.size());
		for (int i = 0; i < L.size(); i++) {
			Layer l = L.get(i);
			oos.writeUTF(l.getClass().getName());
			l.writeObject(oos);
		}
	}

}
