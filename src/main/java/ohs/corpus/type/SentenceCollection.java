package ohs.corpus.type;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import ohs.io.FileUtils;
import ohs.math.ArrayMath;
import ohs.math.ArrayUtils;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.ByteSize;

public class SentenceCollection extends IntegerMatrix {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7501469460888955780L;

	public static Vocab readVocabFromCollection(String fileName) throws Exception {
		System.out.printf("read [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		Vocab vocab = new Vocab(ois);
		ois.close();
		return vocab;
	}

	private Vocab vocab = new Vocab();

	private int[] samples = new int[0];

	private IntegerArray docToSentCnt;

	private int min_len;

	private int max_len;

	private double avg_len;

	private long tok_size;

	public SentenceCollection() {

	}

	public SentenceCollection(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public SentenceCollection(Vocab vocab, IntegerMatrix sents, IntegerArray docToSentCnt) {
		this.vocab = vocab;
		this.docToSentCnt = docToSentCnt;

		ensureCapacity(sents.size());

		for (IntegerArray sent : sents) {
			add(sent);
		}

		computeLengthStats();
	}

	public double avgLength() {
		return avg_len;
	}

	public ByteSize byteSize() {
		long bytes = vocab.byteSize().getBytes();
		bytes += Integer.BYTES * tok_size;
		bytes += Integer.BYTES * samples.length;
		bytes += Integer.BYTES * docToSentCnt.size();
		return new ByteSize(bytes);
	}

	private void computeLengthStats() {
		min_len = Integer.MAX_VALUE;
		max_len = 0;

		for (int i = 0; i < size(); i++) {
			tok_size += get(i).size();
			min_len = Math.min(min_len, get(i).size());
			max_len = Math.max(max_len, get(i).size());
		}

		avg_len = 1f * tok_size / size();
	}

	public IntegerArray getContextWords(int sloc, int wloc, int context_size) {
		// int sloc = ArrayMath.random(0, sents.length);
		// int wloc = ArrayMath.random(0, sents[sloc].length);

		IntegerArray sent = get(sloc);

		// List<String> words = Arrays.asList(vocab.getObjects(sent));
		IntegerArray context = new IntegerArray();

		int start = Math.max(0, wloc - context_size);
		for (int i = start; i < wloc; i++) {
			context.add(sent.get(i));
		}

		if (wloc + 1 < sent.size()) {
			int end = Math.min(sent.size(), wloc + context_size + 1);
			for (int i = wloc + 1; i < end; i++) {
				context.add(sent.get(i));
			}
		}
		return context;
	}

	public IntegerArray getDocToSentCnt() {
		return docToSentCnt;
	}

	public int getRandomSentLoc() {
		return ArrayMath.random(0, size());
	}

	public int[] getRandomWordLoc() {
		int sloc = ArrayMath.random(0, size());
		int wloc = ArrayMath.random(0, get(sloc).size());
		return new int[] { sloc, wloc };
	}

	public int getRandomWordLoc(int sloc) {
		return ArrayMath.random(0, get(sloc).size());
	}

	public Vocab getVocab() {
		return vocab;
	}

	public int getWord(int sloc, int wloc) {
		return get(sloc).get(wloc);
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ Collection Info ]\n");
		sb.append(String.format("sents:\t[%d]\n", size()));
		sb.append(String.format("toks:\t[%d]\n", tok_size));
		sb.append(String.format("min len:\t[%d]\n", min_len));
		sb.append(String.format("max len:\t[%d]\n", max_len));
		sb.append(String.format("avg len:\t[%f]\n", avg_len));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public long length() {
		return tok_size;
	}

	public void makeSampleTable(int table_size) {
		double[] probs = new double[vocab.size()];
		ArrayUtils.copy(vocab.getCounts().values(), probs);
		ArrayMath.pow(probs, 0.75, probs);
		ArrayMath.cumulateAfterNormalize(probs, probs);
		samples = ArrayMath.sampleTable(probs, table_size);
	}

	public int maxLength() {
		return max_len;
	}

	public int minLength() {
		return min_len;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		vocab = new Vocab(ois);
		docToSentCnt = new IntegerArray(ois);

		super.readObject(ois);

		// IntegerArrayMatrix sents = new IntegerArrayMatrix(ois);
		//
		// ensureCapacity(sents.size());
		//
		// for (int i = 0; i < sents.size(); i++) {
		// add(sents.get(i));
		// }

		computeLengthStats();

		System.out.println(info());
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public int sampleWord() {
		return ArrayMath.sample(samples);
	}

	public String toString() {
		return info();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		vocab.writeObject(oos);
		docToSentCnt.writeObject(oos);
		super.writeObject(oos);
	}

	public void writeObject(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
