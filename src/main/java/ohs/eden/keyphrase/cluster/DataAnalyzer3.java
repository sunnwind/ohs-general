package ohs.eden.keyphrase.cluster;

import java.util.List;

import ohs.io.FileUtils;
import ohs.types.common.StrPair;
import ohs.types.generic.Indexer;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataAnalyzer3 {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DataAnalyzer3 da = new DataAnalyzer3();
		da.analyze();

		System.out.println("process ends.");
	}

	public void analyze() throws Exception {
		KeywordData kwdData = new KeywordData();
		kwdData.readObject(KPPath.KP_DIR + "keyword_data_clusters.ser.gz");

		Indexer<String> docIndexer = kwdData.getDocumentIndxer();
		Indexer<StrPair> kwdIndexer = kwdData.getKeywordIndexer();
		List<String> lines = FileUtils.readLinesFromText(KPPath.KP_DIR + "topics.csv");
		Indexer<String> clstIndexer = Generics.newIndexer();
		SetMap<Integer, Integer> clstToKwds1 = Generics.newSetMap();

		for (int i = 1; i < lines.size(); i++) {
			String line = lines.get(i);
			String[] parts = line.split("\t");
			StrPair kwd = new StrPair(parts[2], parts[3]);

			if (!kwdIndexer.contains(kwd)) {
				continue;
			}

			int kid = kwdIndexer.getIndex(kwd);
			int cid = clstIndexer.getIndex(StrUtils.unwrap(parts[0]));

			String cn = StrUtils.unwrap(parts[1]);

			if (!docIndexer.contains(cn)) {
				continue;
			}

			int did = docIndexer.getIndex(parts[1]);

			clstToKwds1.put(cid, kid);
		}

		SetMap<Integer, Integer> kwdToKwds = Generics.newSetMap();

		for (int cid : clstToKwds1.keySet()) {
			List<Integer> kids = Generics.newArrayList(clstToKwds1.get(cid));
			for (int i = 0; i < kids.size(); i++) {
				int kid1 = kids.get(i);
				for (int j = i + 1; j < kids.size(); j++) {
					int kid2 = kids.get(j);
					kwdToKwds.put(kid1, kid2);
					kwdToKwds.put(kid2, kid1);
				}
			}
		}

		int total_pair_cnt = 0;
		int pair_cnt = 0;

		for (int cid : kwdData.getClusterToKeywords().keySet()) {
			List<Integer> kwds = Generics.newArrayList(kwdData.getClusterToKeywords().get(cid));

			for (int i = 0; i < kwds.size(); i++) {
				int kid1 = kwds.get(i);
				for (int j = i + 1; j < kwds.size(); j++) {
					int kid2 = kwds.get(j);

					total_pair_cnt++;

					if (kwdToKwds.contains(kid1, kid2) || kwdToKwds.contains(kid2, kid1)) {
						pair_cnt++;
					}
				}
			}
		}

		System.out.printf("total pair cnt:\t%d\n", total_pair_cnt);
		System.out.printf("pair cnt:\t%d\n", pair_cnt);

	}

}
