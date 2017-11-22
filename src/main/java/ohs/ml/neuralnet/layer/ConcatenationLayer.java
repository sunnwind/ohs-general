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
import ohs.ml.neuralnet.com.ParameterInitializer;
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

	private List<DenseTensor> Z;

	private List<Layer> L;

	private int output_size;

	public ConcatenationLayer(List<Layer> L) {
		this.L = L;
		for (Layer l : L) {
			output_size += l.getOutputSize();
		}
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
			int size = dY.sizeOfInnerVectors();
			for (int i = 0; i < tmp_dZs.size(); i++) {
				DenseMatrix T = tmp_dZs.get(i);
				VectorUtils.enlarge(T, size, L.get(i).getOutputSize());
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

		return null;
	}

	public ConcatenationLayer copy() {
		List<Layer> lss = Generics.newArrayList(L.size());

		for (Layer l : L) {
			lss.add(l.copy());

		}

		ConcatenationLayer l = new ConcatenationLayer(lss);
		return l;
	}

	@Override
	public Object forward(Object I) {
		DenseTensor X = (DenseTensor) I;
		DenseTensor Y = new DenseTensor();
		List<DenseTensor> Z = Generics.newArrayList(L.size());

		Y.ensureCapacity(X.size());

		VectorUtils.enlarge(tmp_Y, X.sizeOfInnerVectors(), output_size);

		for (Layer l : L) {
			Z.add((DenseTensor) l.forward(I));
		}

		int start = 0;

		for (int i = 0; i < X.size(); i++) {
			DenseMatrix Xm = X.get(i);
			DenseMatrix Ym = tmp_Y.subMatrix(start, Xm.rowSize());
			Ym.setAll(0);

			start += Xm.rowSize();

			for (int j = 0; j < Ym.rowSize(); j++) {
				DenseVector ym = Ym.row(j);

				for (int k = 0, m = 0; k < Z.size(); k++) {
					DenseVector zm = Z.get(k).get(i).row(j);
					for (int l = 0; l < zm.size(); l++) {
						ym.add(m++, zm.value(l));
					}
				}
			}

			Y.add(Ym);
		}

		this.X = X;
		this.Y = Y;
		this.Z = Z;

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
	public void initWeights() {
		for (Layer l : L) {
			l.initWeights();
		}
	}

	@Override
	public void prepareTraining() {
		for (Layer l : L) {
			l.prepareTraining();
		}

	}

	@Override
	public void readObject(ObjectInputStream ois) throws Exception {
	}

	public void setOutputWordIndexes(boolean output_word_indexes) {
	}

	@Override
	public void writeObject(ObjectOutputStream oos) throws Exception {
	}

}
