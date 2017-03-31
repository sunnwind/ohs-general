package ohs.corpus.type;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import ohs.io.FileUtils;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.ByteSize;
import ohs.utils.StrUtils;

public class SubDocumentCollection extends ArrayList<IntegerArrayMatrix> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6907816426608470290L;

	private Vocab vocab = new Vocab();

	private long len_min;

	private long len_max;

	private double len_avg;

	private long len;

	private IntegerArray docseqs;

	private List<String> docids;

	private List<String> items;

	public SubDocumentCollection() {

	}

	public SubDocumentCollection(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public SubDocumentCollection(Vocab vocab, List<String> docids, List<String> items, int[][][] docs, int start_doc_seq) {
		this.vocab = vocab;
		this.docids = docids;
		this.items = items;

		docseqs = new IntegerArray(docs.length);

		ensureCapacity(docs.length);

		for (int i = 0; i < docs.length; i++) {
			add(new IntegerArrayMatrix(docs[i]));
			docseqs.add(start_doc_seq + i);
		}

		computeLengths();
	}

	public SentenceCollection asSentenceCollection() {
		int sent_size = 0;

		for (IntegerArrayMatrix doc : this) {
			sent_size += doc.size();
		}

		IntegerArrayMatrix sents = new IntegerArrayMatrix(sent_size);
		IntegerArray docToSentCnt = new IntegerArray(size());

		for (int i = 0; i < size(); i++) {
			IntegerArrayMatrix doc = get(i);
			docToSentCnt.add(doc.size());

			for (int j = 0; j < doc.size(); j++) {
				sents.add(doc.get(j));
			}
		}

		return new SentenceCollection(vocab, sents, docToSentCnt);
	}

	private void computeLengths() {
		len_min = Integer.MAX_VALUE;
		len_max = 0;
		len = 0;

		for (int i = 0; i < size(); i++) {
			IntegerArrayMatrix doc = get(i);
			int len = doc.sizeOfEntries();
			len += len;
			len_min = Math.min(len_min, len);
			len_max = Math.max(len_max, len);
		}
		len_avg = 1f * len / size();
	}

	public double getAvgLength() {
		return len_avg;
	}

	public ByteSize getByteSize() {
		long bytes = vocab.byteSize().getBytes();
		bytes += Integer.BYTES * len;
		return new ByteSize(bytes);
	}

	public String getId(int i) {
		return docids.get(i);
	}

	public String getItem(int i) {
		return items.get(i);
	}

	public long getMaxLength() {
		return len_max;
	}

	public long getMinLength() {
		return len_min;
	}

	public int getSequence(int i) {
		return docseqs.get(i);
	}

	public String getText(int i) {
		IntegerArrayMatrix doc = get(i);
		StringBuffer sb = new StringBuffer();
		for (int j = 0; j < doc.size(); j++) {
			IntegerArray sent = doc.get(j);
			sb.append(StrUtils.join(" ", vocab.getObjects(sent.values())));

			if (j != doc.size() - 1) {
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	public String info() {
		StringBuffer sb = new StringBuffer();
		sb.append("[ Collection Info ]\n");
		sb.append(String.format("docs:\t[%d]\n", size()));
		sb.append(String.format("toks:\t[%d]\n", len));
		sb.append(String.format("min len:\t[%d]\n", len_min));
		sb.append(String.format("max len:\t[%d]\n", len_max));
		sb.append(String.format("avg len:\t[%f]\n", len_avg));
		sb.append(String.format("mem:\t%s", getByteSize().toString()));
		return sb.toString();
	}

	public long length() {
		return len;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		vocab = new Vocab(ois);
		docids = FileUtils.readStringList(ois);
		items = FileUtils.readStringList(ois);

		int size = ois.readInt();
		ensureCapacity(size);

		for (int i = 0; i < size(); i++) {
			add(new IntegerArrayMatrix(ois));
		}
		computeLengths();
		System.out.println(info());
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read [%s].\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public String toString() {
		return info();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		vocab.writeObject(oos);
		FileUtils.writeStringCollection(oos, docids);
		FileUtils.writeStringCollection(oos, items);

		oos.writeInt(size());
		for (IntegerArrayMatrix a : this) {
			a.writeObject(oos);
		}
	}

	public void writeObject(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
