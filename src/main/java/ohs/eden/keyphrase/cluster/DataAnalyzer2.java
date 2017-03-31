package ohs.eden.keyphrase.cluster;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.types.common.StrPair;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataAnalyzer2 {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DataAnalyzer2 da = new DataAnalyzer2();

		int a_type = 3;

		if (a_type == 0) {
			KeywordData kwdData = new KeywordData();
			kwdData.readObject(KPPath.KYP_DIR + "kwd_2p-wos_clusters.ser.gz");

			Map<Integer, String> docToType = Generics.newHashMap();

			{
				TextFileReader reader = new TextFileReader(KPPath.KYP_DIR + "cn.txt.gz");
				while (reader.hasNext()) {
					String[] parts = reader.next().split("\t");
					String cn = parts[0];
					String type = parts[1];

					if (type.equals("patent")) {
						break;
					}

					int docid = kwdData.getDocumentIndxer().indexOf(cn);

					if (docid < 0) {
						continue;
					}

					docToType.put(docid, type);
				}
				reader.close();

				for (int i = 0; i < kwdData.getDocumentIndxer().size(); i++) {
					String cn = kwdData.getDocumentIndxer().getObject(i);

					if (cn.startsWith("WOS")) {
						docToType.put(i, "wos");
					}
				}
			}
			da.analyze1(kwdData, docToType, KPPath.KYP_DIR + "kwd_2p-wos_types.txt");
		} else if (a_type == 1) {
			da.analyze2(KPPath.KYP_DIR + "kwd_2p-wos_types.txt", null);
		} else if (a_type == 2) {
			da.analyze3(KPPath.KYP_DIR + "kwd_2p.txt.gz", KPPath.KYP_DIR + "kwd_wos.txt.gz", null);
		} else if (a_type == 3) {
			da.analyze4();
		}

		System.out.println("process ends.");
	}

	public DataAnalyzer2() {
	}

	public void analyze4() throws Exception {
		KeywordData kwdData = new KeywordData();
		kwdData.readObject(KPPath.KYP_DIR + "keyword_data_clusters.ser.gz");

		System.out.println(kwdData.getClusterToKeywords().size());
		System.out.println(kwdData.getKeywords().size());
		System.out.println(kwdData.getKeywordIndexer().size());

		Counter<Integer> c = Generics.newCounter();

		for (int cid : kwdData.getClusterToKeywords().keySet()) {
			Set<Integer> kwdids = kwdData.getClusterToKeywords().get(cid);

			c.incrementCount(kwdids.size(), 1);
		}

		for (int key : Generics.newTreeSet(c.keySet())) {
			int cnt = (int) c.getCount(key);
			System.err.println(key + "\t" + cnt);
		}

		// System.out.println(c.toStringSortedByValues(true, true, c.size(), "\t"));
	}

	public void analyze3(String inFileName1, String inFileName2, String outFileName) {
		Counter<String> c1 = read(inFileName1);
		Counter<String> c2 = read(inFileName2);
		Counter<String> c3 = Generics.newCounter();

		int c1_size = c1.size();
		int c2_size = c2.size();
		int co_cnt = 0;

		for (String kwd : c1.keySet()) {
			double cnt1 = c1.getCount(kwd);
			if (c2.containsKey(kwd)) {
				double cnt2 = c2.getCount(kwd);
				c3.incrementCount(kwd, cnt1 + cnt2);
			}
		}

		// System.out.printf("(%d, %d, %d)\n", c1_size, c2_size, co_cnt);

		System.out.println(c3.toStringSortedByValues(true, true, 50, ""));

	}

	private Counter<String> read(String fileName) {
		System.out.printf("read [%s]\n", fileName);

		Counter<String> c = Generics.newCounter();

		TextFileReader reader = new TextFileReader(fileName);
		reader.setPrintSize(100000);

		while (reader.hasNext()) {
			reader.printProgress();
			String line = reader.next();

			if (line.startsWith(FileUtils.LINE_SIZE)) {
				continue;
			}

			String[] parts = line.split("\t");

			parts = StrUtils.unwrap(StrUtils.subArray(parts, 0, 2));

			String kor = parts[0];
			String eng = parts[1].trim();
			if (eng.length() > 0) {
				c.incrementCount(eng.toLowerCase(), 1);
			}
		}
		reader.printProgress();
		reader.close();

		return c;
	}

	public void analyze2(String inFileName, String outFileName) {
		TextFileReader reader = new TextFileReader(inFileName);

		Counter<String> c = Generics.newCounter();
		List<String> res = Generics.newArrayList();

		while (reader.hasNext()) {
			if (reader.getLineCnt() < 4) {
				continue;
			}

			String line = reader.next();

			String[] parts = line.split("\t");

			String kwdStr = parts[2];
			int kwd_cnt = Integer.parseInt(parts[3]);

			int doc_cnt_in_wos = 0;
			int doc_cnt_in_2p = 0;

			for (int i = 4; i < parts.length; i++) {
				if (i == parts.length - 1) {
					doc_cnt_in_wos += Integer.parseInt(parts[i]);
				} else {
					doc_cnt_in_2p += Integer.parseInt(parts[i]);
				}
			}

			if (doc_cnt_in_2p > 0 && doc_cnt_in_wos > 0) {
				c.incrementCount("BOTH", 1);
			} else {
				if (doc_cnt_in_2p > 0) {
					c.incrementCount("2P", 1);

				} else if (doc_cnt_in_wos > 0) {
					c.incrementCount("WOS", 1);
					if (res.size() < 20) {
						res.add(line);
					}
				}
			}
		}
		reader.close();

		// System.out.println(c.toString());
		System.out.println(StrUtils.join("\n", res));
	}

	public void analyze1(KeywordData kwdData, Map<Integer, String> docToType, String outFileName) {
		TextFileWriter writer = new TextFileWriter(outFileName);

		SetMap<Integer, Integer> clstToKwds = kwdData.getClusterToKeywords();
		Indexer<StrPair> kwdIdx = kwdData.getKeywordIndexer();
		Map<Integer, Integer> clstToLabel = kwdData.getClusterToLabel();

		writer.write(String.format("Clusters:\t%d", clstToKwds.size()));
		writer.write(String.format("\nKeywords:\t%d", clstToKwds.totalSize()));

		List<Integer> cids = Generics.newArrayList();

		boolean sort_alphabetically = false;

		if (sort_alphabetically) {
			List<String> keys = Generics.newArrayList();

			for (int cid : clstToKwds.keySet()) {
				keys.add(kwdIdx.getObject(cid).join("\t"));
			}

			Collections.sort(keys);

			cids = Generics.newArrayList();
			for (int i = 0; i < keys.size(); i++) {
				int kwdid = kwdIdx.indexOf(keys.get(i));
				cids.add(kwdid);
			}
		} else {
			Counter<Integer> c = Generics.newCounter();

			for (int cid : clstToKwds.keySet()) {
				c.incrementCount(cid, clstToKwds.get(cid).size());
			}
			cids = c.getSortedKeys();
		}

		writer.write("\nNo\tID\tLabel\tKWDS\tPapers\tReports\tWOS");

		String[] types = { "paper", "report", "wos" };

		for (int i = 0, n = 1; i < cids.size(); i++) {
			int cid = cids.get(i);
			int label = clstToLabel.get(cid);
			StrPair kwdp = kwdIdx.getObject(label);

			StringBuffer sb = new StringBuffer();
			String kwdStr = String.format("(%s)", kwdp.join(", "));
			sb.append(String.format("%d\t%d\t%s\t%d", n, cid, kwdStr, clstToKwds.get(cid).size()));

			Counter<Integer> c = Generics.newCounter();

			for (int kwdid : clstToKwds.get(cid)) {
				c.setCount(kwdid, kwdData.getKeywordFreqs()[kwdid]);
			}

			n++;

			List<Integer> kwdids = c.getSortedKeys();
			Counter<String> typeCnts = Generics.newCounter();
			for (int j = 0; j < kwdids.size(); j++) {
				int kwdid = kwdids.get(j);
				for (int docid : kwdData.getKeywordToDocs().get(kwdid)) {
					typeCnts.incrementCount(docToType.get(docid), 1);
				}
			}

			for (int j = 0; j < types.length; j++) {
				sb.append(String.format("\t%d", (int) typeCnts.getCount(types[j])));
			}
			writer.write("\n" + sb.toString());
		}
		writer.close();

		System.out.printf("write [%d] clusters at [%s]\n", clstToKwds.size(), outFileName);
	}

}
