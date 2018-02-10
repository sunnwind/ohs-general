package ohs.nlp.ling.types;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ohs.io.FileUtils;
import ohs.types.generic.Counter;
import ohs.utils.Generics;

public class LDocumentCollection extends ArrayList<LDocument> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2528539276233604652L;

	private Map<String, String> attrMap = Generics.newHashMap();

	public LDocumentCollection() {
		super();
	}

	public LDocumentCollection(int size) {
		super(size);
	}

	public LDocumentCollection(LDocumentCollection m) {
		super(m);
	}

	public LDocumentCollection(List<LDocument> m) {
		super(m);
	}

	@Override
	public LDocumentCollection clone() {
		LDocumentCollection ret = new LDocumentCollection(size());

		for (LDocument d : this) {
			ret.add(d.clone());
		}

		HashMap<String, String> am = (HashMap<String, String>) attrMap;
		ret.setAttrMap((Map<String, String>) am.clone());
		return ret;
	}

	public void doPadding() {
		for (LDocument d : this) {
			d.doPadding();
		}
	}

	public Map<String, String> getAttrMap() {
		return attrMap;
	}

	public Counter<String> getCounter(int idx) {
		Counter<String> ret = Generics.newCounter();
		for (String t : getTokenStrings(idx)) {
			ret.incrementCount(t, 1);
		}
		return ret;
	}

	public LDocument getSentences() {
		LDocument ret = new LDocument(sizeOfSentences());
		for (LDocument d : this) {
			ret.addAll(d);
		}
		return ret;
	}

	public LSentence getTokens() {
		LSentence ret = new LSentence(sizeOfTokens());
		for (LDocument d : this) {
			ret.addAll(d.getTokens());
		}
		return ret;
	}

	public List<String> getTokenStrings(int idx) {
		return getTokens().getTokenStrings(idx);
	}

	public List<String> getTokenStrings(int[] idxs, String glue) {
		return getTokens().getTokenStrings(idxs, glue);
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		LToken.INDEXER = FileUtils.readStringIndexer(ois);

		{
			int size = ois.readInt();
			for (int i = 0; i < size; i++) {
				LDocument t = new LDocument();
				t.readObject(ois);
				add(t);
			}
		}

		{
			int size = ois.readInt();

			for (int i = 0; i < size; i++) {
				attrMap.put(ois.readUTF(), ois.readUTF());
			}
		}

	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s]\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void setAttrMap(Map<String, String> attrMap) {
		this.attrMap = attrMap;
	}

	public int sizeOfSentences() {
		int size = 0;
		for (LDocument d : this) {
			size += d.size();
		}
		return size;
	}

	public int sizeOfTokens() {
		int size = 0;
		for (LDocument d : this) {
			size += d.sizeOfTokens();
		}
		return size;
	}

	public LDocumentCollection subCollection(int i, int j) {
		return new LDocumentCollection(subList(i, j));
	}

	public LDocumentCollection subCollection(int[] idxs) {
		LDocumentCollection ret = new LDocumentCollection(idxs.length);
		for (int idx : idxs) {
			ret.add(get(idx));
		}
		return ret;
	}

	public void writeObejct(String fileName) throws Exception {
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.flush();
		oos.close();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringIndexer(oos, LToken.INDEXER);

		{
			oos.writeInt(size());
			for (int i = 0; i < size(); i++) {
				get(i).writeObject(oos);
			}
		}

		{
			oos.writeInt(attrMap.size());
			for (Entry<String, String> e : attrMap.entrySet()) {
				oos.writeUTF(e.getKey());
				oos.writeUTF(e.getValue());
			}
		}
	}

}
