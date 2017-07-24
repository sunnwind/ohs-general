package ohs.ir.medical.query;

import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.types.generic.BidMap;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.utils.Generics;

public class RelevanceReader {

	public static CounterMap<String, String> filter(CounterMap<String, String> relData,
			BidMap<String, String> docIdMap) {
		CounterMap<String, String> ret = new CounterMap<String, String>();

		int num_pairs = 0;
		int num_result_pairs = 0;

		for (String queryId : relData.keySet()) {
			Counter<String> docRels = relData.getCounter(queryId);
			Counter<String> counter = new Counter<String>();

			num_pairs += docRels.size();

			for (String docId : docRels.keySet()) {
				double score = docRels.getCount(docId);
				String indexId = docIdMap.getKey(docId);
				if (indexId == null || score == 0) {
					continue;
				}
				counter.setCount(docId, score);
				num_result_pairs++;
			}

			if (counter.size() == 0) {
				continue;
			}

			ret.setCounter(queryId, counter);
		}
		System.out.printf("[%d] queries -> [%d] queries\n", relData.keySet().size(), ret.keySet().size());
		System.out.printf("[%d] pairs -> [%d] pairs\n", num_pairs, num_result_pairs);
		return ret;
	}

	/**
	 * d definitely relevant -> 2 p possibly relevant -> 1 n not relevant -> 0
	 * 
	 * @param judgement
	 * @return
	 */
	private static double getRelevance(String judgement) {
		double ret = 0;
		if (judgement.equals("d")) {
			ret = 2;
		} else if (judgement.equals("p")) {
			ret = 1;
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {

		{
			CounterMap<String, String> relData = readTrecGenomicsRelevances(
					MIRPath.TREC_GENO_2007_RELEVANCE_JUDGE_FILE);
			System.out.println(relData);
		}

	}

	public static CounterMap<String, String> readClefEHealthRelevances(String fileName) {
		CounterMap<String, String> ret = new CounterMap<String, String>();
		TextFileReader reader = new TextFileReader(fileName);
		while (reader.hasNext()) {
			String line = reader.next();
			// String[] parts = line.split("\\text");
			String[] parts = line.split(" ");

			// if(parts.length == 1){
			// parts = line.split(" ");
			// }

			String qid = parts[0];
			String docid = parts[2];
			double relevance = Double.parseDouble(parts[3]);

			if (fileName.contains("clef2015.test.graded")) {
				qid = qid.replace("qtest", "clef2015.test");
			}

			// if (relevance > 0) {
			ret.setCount(qid, docid, relevance);
			// }
		}
		reader.close();
		return ret;
	}

	public static CounterMap<String, String> readOhsumedRelevances(String fileName) {
		CounterMap<String, String> ret = Generics.newCounterMap();
		TextFileReader reader = new TextFileReader(fileName);
		while (reader.hasNext()) {
			String line = reader.next();
			String[] parts = line.split("\t");

			String qid = parts[0];
			String doc_ui = parts[1];
			String doc_i = parts[2];
			String judgement1 = parts[3];
			String judgement2 = parts.length > 4 ? parts[4] : "";
			String judgement3 = parts.length > 5 ? parts[5] : "";

			Counter<String> counter = new Counter<String>();

			if (judgement1.length() > 0) {
				counter.incrementCount(judgement1, 1);
			}

			if (judgement2.length() > 0) {
				counter.incrementCount(judgement2, 1);
			}

			if (judgement3.length() > 0) {
				counter.incrementCount(judgement3, 1);
			}

			if (counter.size() == 1) {
				double relevance = getRelevance(counter.argMax());
				// if (relevance > 0) {
				ret.setCount(qid, doc_ui, relevance);
				// }
			} else {

			}
		}
		reader.close();
		return ret;
	}

	public static CounterMap<String, String> readRelevances(String fileName) throws Exception {
		CounterMap<String, String> ret = new CounterMap<String, String>();
		if (fileName.contains("trec_cds")) {
			ret = readTrecCdsRelevances(fileName);
		} else if (fileName.contains("clef_ehealth")) {
			ret = readClefEHealthRelevances(fileName);
		} else if (fileName.contains("ohsumed")) {
			ret = readOhsumedRelevances(fileName);
		} else if (fileName.contains("trec_genomics")) {
			ret = readTrecGenomicsRelevances(fileName);
		} else if (fileName.contains("trec_pm")) {

		}
		System.out.printf("read [%d] at [%s]\n", ret.totalSize(), fileName);
		return ret;
	}

	public static CounterMap<String, String> readTrecCdsRelevances(String fileName) {
		CounterMap<String, String> ret = Generics.newCounterMap();
		TextFileReader reader = new TextFileReader(fileName);
		while (reader.hasNext()) {
			String line = reader.next();
			String[] parts = line.split("[\\s]+");

			String qid = parts[0];
			String docid = parts[2];
			double relevance = Double.parseDouble(parts[3]);
			// if (relevance > 0) {
			ret.setCount(qid, docid, relevance);
			// }
		}
		reader.close();
		return ret;
	}

	public static CounterMap<String, String> readTrecGenomicsRelevances(String fileName) {
		CounterMap<String, String> ret = Generics.newCounterMap();
		TextFileReader reader = new TextFileReader(fileName);
		while (reader.hasNext()) {
			String line = reader.next();
			if (line.startsWith("#")) {
				continue;
			}

			String[] parts = line.split("\t");

			String qid = parts[0];
			String docid = parts[1];
			double relevance = parts[4].equals("RELEVANT") ? 1 : 0;
			// if (relevance > 0) {
			ret.setCount(qid, docid, relevance);
			// }
		}
		reader.close();
		return ret;
	}
}
