package ohs.ml.glove;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.Vector;
import ohs.ml.neuralnet.com.ParameterInitializer;
import ohs.ml.word2vec.WordSearcher;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

public class GloveModel {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		String dir = MIRPath.TREC_CDS_2014_DIR;

		GloveModel M = new GloveModel();
		M.readObject(dir + "glove_model.ser.gz");

		DocumentCollection ldc = new DocumentCollection(dir + "col/dc/");

		Set<String> stopwords = FileUtils.readStringHashSetFromText(MIRPath.STOPWORD_INQUERY_FILE);

		WordSearcher.interact(new WordSearcher(ldc.getVocab(), M.getAveragedModel(), stopwords));
		System.out.println("process ends.");
	}

	/**
	 * center words
	 */
	private DenseMatrix W1;

	private DenseVector b1;

	/**
	 * context words
	 */
	private DenseMatrix W2;

	private DenseVector b2;

	/**
	 * all weights
	 */
	private DenseMatrix W;

	public GloveModel() {
		super();
	}

	public GloveModel(int vocab_size, int hidden_size) {
		W1 = new DenseMatrix(vocab_size, hidden_size);
		W2 = new DenseMatrix(vocab_size, hidden_size);
		b1 = new DenseVector(vocab_size);
		b2 = new DenseVector(vocab_size);
		createW();

	}

	public ByteSize byteSize() {
		return new ByteSize(Double.BYTES * W.sizeOfEntries());
	}

	private void createW() {
		List<DenseVector> rows = Generics.newArrayList();

		for (Vector row : W1) {
			rows.add((DenseVector) row);
		}

		for (Vector row : W2) {
			rows.add((DenseVector) row);
		}

		rows.add(b1);
		rows.add(b2);

		W = new DenseMatrix(rows);
	}

	public DenseMatrix getAveragedModel() {
		DenseMatrix A = W1.copy();
		VectorMath.addAfterMultiply(W1, 0.5, W2, 0.5, A);
		VectorMath.unitVector(A, A);
		return A;
	}

	public DenseVector getB1() {
		return b1;
	}

	public DenseVector getB2() {
		return b2;
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
	public DenseMatrix getW1() {
		return W1;
	}

	public DenseMatrix getW2() {
		return W2;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ Glove Model Info ]\n");
		sb.append(String.format("W1:\t[%d, %d]\n", W1.rowSize(), W1.colSize()));
		sb.append(String.format("W2:\t[%d, %d]\n", W2.rowSize(), W2.colSize()));
		sb.append(String.format("b1:\t[%d]\n", b1.size()));
		sb.append(String.format("b2:\t[%d]\n", b2.size()));
		sb.append(String.format("params:\t[%d]\n", W.sizeOfEntries()));
		sb.append(String.format("mem:\t%s\n", byteSize().toString()));
		return sb.toString();
	}

	public void init() {
		ParameterInitializer.init2(W1);
		ParameterInitializer.init2(W2);
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		W1 = new DenseMatrix(ois);
		W2 = new DenseMatrix(ois);
		b1 = new DenseVector(ois);
		b2 = new DenseVector(ois);
		createW();
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void setW(DenseMatrix w) {
		W = w;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		W1.writeObject(oos);
		W2.writeObject(oos);
		b1.writeObject(oos);
		b2.writeObject(oos);
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
