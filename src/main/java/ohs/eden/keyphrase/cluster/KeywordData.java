package ohs.eden.keyphrase.cluster;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Map;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.types.common.StrPair;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.Timer;

public class KeywordData {

	private Indexer<StrPair> kwdIndexer;

	private Indexer<String> docIndxer;

	private ListMap<Integer, Integer> kwdToDocs;

	private List<Integer> kwdids;

	private int[] kwd_freqs = new int[0];

	private SetMap<Integer, Integer> clstToKwds;

	private Map<Integer, Integer> clstToLabel;

	private Map<Integer, Integer> clstToParent;

	public KeywordData() {
		kwdIndexer = Generics.newIndexer();
		docIndxer = Generics.newIndexer();

		kwdids = Generics.newArrayList();
		kwdToDocs = Generics.newListMap();

		clstToKwds = Generics.newSetMap();
		clstToLabel = Generics.newHashMap();
		clstToParent = Generics.newHashMap();
	}

	public void add(String fileName) {
		System.out.printf("read [%s]\n", fileName);

		Counter<Integer> kwdFreqs = Generics.newCounter();

		if (kwd_freqs != null) {
			for (int i = 0; i < kwd_freqs.length; i++) {
				kwdFreqs.incrementCount(i, kwd_freqs[i]);
			}
		}

		TextFileReader reader = new TextFileReader(fileName);
		reader.setPrintSize(100000);

		while (reader.hasNext()) {
			reader.printProgress();
			String line = reader.next();

			if (line.startsWith(FileUtils.LINE_SIZE)) {
				continue;
			}

			String[] parts = line.split("\t");

			String kor = parts[0];
			String eng = parts[1];

			StrPair kwdp = new StrPair(kor, eng);
			int kwd_freq = Integer.parseInt(parts[2]);

			// kor = kor.substring(1, kor.length() - 2);
			// eng = eng.substring(1, eng.length() - 2);
			String kwdStr = kor + "\t" + eng;
			int kwdid = kwdIndexer.getIndex(kwdp);
			kwdids.add(kwdid);
			kwdFreqs.incrementCount(kwdid, kwd_freq);

			List<Integer> docids = Generics.newArrayList();

			for (int i = 3; i < parts.length; i++) {
				String cn = parts[i];
				docids.add(docIndxer.getIndex(cn));
			}
			kwdToDocs.put(kwdid, docids);
		}
		reader.printProgress();
		reader.close();

		kwd_freqs = new int[kwdFreqs.size()];

		for (int i = 0; i < kwdIndexer.size(); i++) {
			kwd_freqs[i] = (int) kwdFreqs.getCount(i);
		}
	}

	public SetMap<Integer, Integer> getClusterToKeywords() {
		return clstToKwds;
	}

	public Map<Integer, Integer> getClusterToLabel() {
		return clstToLabel;
	}

	public Indexer<String> getDocumentIndxer() {
		return docIndxer;
	}

	public int getKeywordFreq(int kwdid) {
		return kwd_freqs[kwdid];
	}

	public int[] getKeywordFreqs() {
		return kwd_freqs;
	}

	public Indexer<StrPair> getKeywordIndexer() {
		return kwdIndexer;
	}

	public List<Integer> getKeywords() {
		return kwdids;
	}

	public ListMap<Integer, Integer> getKeywordToDocs() {
		return kwdToDocs;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		int size = ois.readInt();
		kwdIndexer = Generics.newIndexer(size);

		for (int i = 0; i < size; i++) {
			StrPair kwdp = new StrPair();
			kwdp.read(ois);
			kwdIndexer.add(kwdp);
		}

		kwdids = FileUtils.readIntegerList(ois);
		kwd_freqs = FileUtils.readIntegers(ois);

		docIndxer = FileUtils.readStringIndexer(ois);
		kwdToDocs = FileUtils.readIntegerListMap(ois);

		clstToKwds = FileUtils.readIntegerSetMap(ois);
		clstToLabel = FileUtils.readIntegerMap(ois);
		clstToParent = FileUtils.readIntegerMap(ois);
	}

	public void readObject(String fileName) throws Exception {
		Timer timer = Timer.newTimer();
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
		System.out.printf("read at [%s] - [%s]\n", fileName, timer.stop());
	}

	public void setClusterLabel(Map<Integer, Integer> clusterToLabel) {
		this.clstToLabel = clusterToLabel;
	}

	public void setClusters(SetMap<Integer, Integer> clusters) {
		this.clstToKwds = clusters;
	}

	public void setClusterToParent(Map<Integer, Integer> clusterToParent) {
		this.clstToParent = clusterToParent;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		oos.writeInt(kwdIndexer.size());
		for (int i = 0; i < kwdIndexer.size(); i++) {
			kwdIndexer.getObject(i).write(oos);
		}

		FileUtils.writeIntegerCollection(oos, kwdids);
		FileUtils.writeIntegers(oos, kwd_freqs);

		FileUtils.writeStringIndexer(oos, docIndxer);
		FileUtils.writeIntegerListMap(oos, kwdToDocs);

		FileUtils.writeIntegerSetMap(oos, clstToKwds);
		FileUtils.writeIntegerMap(oos, clstToLabel);
		FileUtils.writeIntegerMap(oos, clstToParent);
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}
}
