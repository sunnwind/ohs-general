package ohs.types.generic;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import ohs.io.FileUtils;
import ohs.types.number.IntegerArray;
import ohs.utils.ByteSize;
import ohs.utils.Generics;

public class Vocab extends Indexer<String> {

	public static enum SYM {
		UNK, START, END;

		public static SYM parse(String text) {
			SYM ret = null;
			if (text.equals(UNK.getText())) {
				ret = UNK;
			} else if (text.equals(START.getText())) {
				ret = START;
			} else if (text.equals(END.getText())) {
				ret = END;
			}
			return ret;
		}

		public String getText() {
			String ret = "<unk>";

			if (this == START) {
				ret = "<s>";
			} else if (this == END) {
				ret = "</s>";
			}

			return ret;
		}
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 4937817092179320891L;

	private IntegerArray word_cnts = new IntegerArray(0);

	private IntegerArray doc_freqs = new IntegerArray(0);

	private int doc_cnt = 0;

	private long sent_cnt = 0;

	private long tok_cnt = 0;

	public Vocab() {
		super();
	}

	public Vocab(Indexer<String> wordIndexer) {
		super(wordIndexer);
	}

	public Vocab(int size) {
		super(size);
	}

	public Vocab(List<String> words) {
		super(words, true);
	}

	public Vocab(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public Vocab(String fileName) throws Exception {
		readObject(fileName);
	}

	public ByteSize byteSize() {
		long ret = 0;
		for (String word : getObjects()) {
			ret += 2 * word.length();
		}
		ret += Integer.BYTES * size();
		ret += Integer.BYTES * word_cnts.size();
		ret += Integer.BYTES * doc_freqs.size();
		return new ByteSize(ret);
	}

	private void countTokens() {
		tok_cnt = 0;
		for (int i = 0; i < word_cnts.size(); i++) {
			tok_cnt += word_cnts.get(i);
		}
	}

	public int getCount(int w) {
		return word_cnts.get(w);
	}

	public int getCount(String word) {
		int w = indexOf(word);
		int ret = 0;
		if (w > -1) {
			ret = word_cnts.get(w);
		}
		return ret;
	}

	public Counter<String> getCounter() {
		Counter<String> ret = Generics.newCounter();
		for (int w = 0; w < size(); w++) {
			ret.setCount(getObject(w), getCount(w));
		}
		return ret;
	}

	public IntegerArray getCounts() {
		return word_cnts;
	}

	public int getDocCnt() {
		return doc_cnt;
	}

	public int getDocFreq(int w) {
		return doc_freqs.get(w);
	}

	public int getDocFreq(String word) {
		int w = indexOf(word);
		int ret = 0;
		if (w > -1) {
			ret = doc_freqs.get(w);
		}
		return ret;
	}

	public IntegerArray getDocFreqs() {
		return doc_freqs;
	}

	public double getProb(int w) {
		return 1d * word_cnts.get(w) / tok_cnt;
	}

	public double getProb(String word) {
		double ret = 0;
		int w = indexOf(word);
		if (w > -1) {
			ret = getProb(w);
		}
		return ret;
	}

	public long getSentCnt() {
		return sent_cnt;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ Vocab Info ]\n");
		sb.append(String.format("size:\t[%d]\n", size()));
		sb.append(String.format("docs:\t[%d]\n", doc_cnt));
		sb.append(String.format("sents:\t[%d]\n", sent_cnt));
		sb.append(String.format("toks:\t[%d]\n", tok_cnt));
		sb.append(String.format("mem:\t%s", byteSize().toString()));
		return sb.toString();
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		for (int i = 0; i < size; i++) {
			add(ois.readUTF());
		}
		trimToSize();

		doc_cnt = ois.readInt();
		word_cnts = new IntegerArray(ois);
		doc_freqs = new IntegerArray(ois);

		countTokens();
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void setDocCnt(int doc_cnt) {
		this.doc_cnt = doc_cnt;
	}

	public void setDocFreqs(IntegerArray doc_freqs) {
		this.doc_freqs = doc_freqs;
	}

	public void setSentCnt(long sent_cnt) {
		this.sent_cnt = sent_cnt;
	}

	public void setWordCnts(IntegerArray word_cnts) {
		this.word_cnts = word_cnts;
		countTokens();
	}

	public long sizeOfTokens() {
		return tok_cnt;
	}

	public String toString() {
		return info();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringCollection(oos, this);
		oos.writeInt(doc_cnt);
		word_cnts.writeObject(oos);
		doc_freqs.writeObject(oos);
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s].\n", fileName);
		ObjectOutputStream ois = FileUtils.openObjectOutputStream(fileName);
		writeObject(ois);
		ois.close();
	}

}
