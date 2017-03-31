package ohs.ml.word2vec;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import ohs.io.FileUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

public class Word2VecModel {

	/**
	 * input to hidden
	 */
	private DenseMatrix Wxh;

	/**
	 * output to hidden
	 */
	private DenseMatrix Wyh;

	/**
	 * all weights
	 */
	private DenseMatrix W;

	public Word2VecModel(int vocab_size, int hidden_size) {
		Wxh = new DenseMatrix(vocab_size, hidden_size);
		Wyh = new DenseMatrix(vocab_size, hidden_size);

		List<DenseVector> rows = Generics.newArrayList(Wxh.rowSize() + Wyh.rowSize());

		for (DenseVector row : Wxh) {
			rows.add(row);
		}

		for (DenseVector row : Wyh) {
			rows.add(row);
		}

		W = new DenseMatrix(rows);
	}

	public ByteSize byteSize() {
		return new ByteSize(Double.BYTES * size());
	}

	public DenseMatrix getAveragedModel() {
		DenseMatrix A = Wxh.copy();
		VectorMath.addAfterMultiply(Wxh, 0.5, Wyh, 0.5, A);
		VectorMath.unitVector(A, A);
		return A;
	}

	/**
	 * all weights
	 * 
	 * @return
	 */
	public DenseMatrix getW() {
		return W;
	}

	/**
	 * input to hidden
	 * 
	 * @return
	 */
	public DenseMatrix getWxh() {
		return Wxh;
	}

	/**
	 * output to hidden
	 * 
	 * @return
	 */
	public DenseMatrix getWyh() {
		return Wyh;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ Word2Vec Model Info ]\n");
		sb.append(String.format("Wxh:\t(%d, %d)\n", Wxh.rowSize(), Wxh.colSize()));
		sb.append(String.format("Wyh:\t(%d, %d)\n", Wyh.rowSize(), Wyh.colSize()));
		sb.append(String.format("params:\t%d\n", size()));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void init() {
		double bound = Math.sqrt(6f / (Wxh.rowSize() + Wxh.colSize()));
		VectorMath.random(-bound, bound, Wxh);
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		Wxh = new DenseMatrix(ois);
		Wyh = new DenseMatrix(ois);
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void reset() {
		W.setAll(0);
	}

	public void setW(DenseMatrix w) {
		W = w;
	}

	public void setW1(DenseMatrix Wxh) {
		this.Wxh = Wxh;
	}

	public void setW2(DenseMatrix Wyh) {
		this.Wyh = Wyh;
	}

	public long size() {
		return W.sizeOfEntries();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		Wxh.writeObject(oos);
		Wyh.writeObject(oos);
	}

	public void writeObject(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
