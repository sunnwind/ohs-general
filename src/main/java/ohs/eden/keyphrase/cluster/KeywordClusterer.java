package ohs.eden.keyphrase.cluster;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.tartarus.snowball.ext.PorterStemmer;

import ohs.io.TextFileWriter;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.string.search.ppss.Gram;
import ohs.string.search.ppss.GramGenerator;
import ohs.types.common.StrPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;
import ohs.utils.UnicodeUtils;

public class KeywordClusterer {

	public static void main(String[] args) throws Exception {
		System.out.printf("[%s] begins.\n", KeywordClusterer.class.getName());

		KeywordData kwdData = new KeywordData();

		kwdData.add(KPPath.KYP_2P_FILE);
		// kwdData.add(KPPath.KYP_WOS_FILE);

		// kwdData.add(KPPath);
		// kwdData.writeObject(KPPath.KYP_DATA_SER_FILE);

		KeywordClusterer kc = new KeywordClusterer(kwdData);
		// kc.setTitleData(FileUtils.readStrCounterMap(KPPath.KYP_TITLE_DATA_FILE));
		kc.cluster();
		// kc.writeClusters(KPPath.KYP_DIR + "kwd_2p-wos_clusters.txt.gz");

		// kwdData.writeObject(KPPath.KYP_DIR + "kwd_2p-wos_clusters.ser.gz");

		System.out.println("process ends.");
	}

	public static String normalize(String s) {
		return s.replaceAll("[\\p{Punct}\\s]+", " ").toLowerCase().trim();
	}

	public static String normalizeEnglish(String s) {
		PorterStemmer stemmer = new PorterStemmer();
		StringBuffer sb = new StringBuffer();
		for (String word : StrUtils.splitPunctuations(s)) {
			stemmer.setCurrent(word.toLowerCase());
			stemmer.stem();
			sb.append(stemmer.getCurrent() + " ");
		}
		return sb.toString().trim();
	}

	private Map<Integer, SparseVector> kwdToWordCnts;

	private Indexer<String> wordIdx;

	private KeywordData kwdData;

	private Indexer<StrPair> kwdIdx;

	private SetMap<Integer, Integer> clstToKwds;

	private Map<Integer, Integer> clstToParent;

	private int[] kwdToClst;

	private Map<Integer, Integer> clstToLabel;

	private boolean write_temp_result = false;

	public KeywordClusterer(KeywordData kwdData) {
		this.kwdData = kwdData;

		kwdIdx = kwdData.getKeywordIndexer();
	}

	public void cluster() throws Exception {
		kwdToClst = new int[kwdIdx.size()];

		clstToParent = Generics.newHashMap();

		clstToKwds = Generics.newSetMap(kwdIdx.size());

		for (int i = 0; i < kwdIdx.size(); i++) {
			clstToKwds.put(i, i);
			kwdToClst[i] = i;
		}

		matchExactTwoLanguages();

		if (write_temp_result) {
			selectClusterLabels();
			writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + "temp-01.txt.gz");
		}

		matchExactKorean();

		if (write_temp_result) {
			selectClusterLabels();
			writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + "temp-02.txt.gz");
		}

		// matchExactEnglish();

		if (write_temp_result) {
			selectClusterLabels();
			writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + "temp-03.txt.gz");
		}

		matchKoreanGrams();

		if (write_temp_result) {
			selectClusterLabels();
			writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + "temp-04.txt.gz");
		}

		matchEnglishGrams();

		if (write_temp_result) {
			selectClusterLabels();
			writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + "temp-05.txt.gz");
		}

		if (kwdToWordCnts != null) {
			matchContexts();
		}

		if (write_temp_result) {
			writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + "temp-06.txt.gz");
		}

		selectClusterLabels();

		kwdData.setClusterLabel(clstToLabel);
		kwdData.setClusters(clstToKwds);
		kwdData.setClusterToParent(clstToParent);
	}

	private Counter<Integer> computeKeywordScores(Set<Integer> kwdids) {
		Map<Integer, SparseVector> kwdCents = Generics.newHashMap();

		Indexer<String> wordIndexer = Generics.newIndexer();

		for (int kwdid : kwdids) {
			StrPair kwdp = kwdIdx.getObject(kwdid);

			Counter<Integer> cc = Generics.newCounter();
			String[] two = kwdp.asArray();

			for (int i = 0; i < two.length; i++) {
				String key = two[i];

				Counter<String> c = Generics.newCounter();

				if (i == 0) {
					key = normalize(key);

					if (key.length() > 0) {
						c.incrementCount(key.charAt(0) + "", 1);
					}

					if (key.length() > 1) {
						for (int j = 1; j < key.length(); j++) {
							c.incrementCount(key.substring(j - 1, j), 1);
							c.incrementCount(key.charAt(j) + "", 1);
						}
					}
				} else {
					List<String> words = StrUtils.splitPunctuations(key.toLowerCase());

					if (words.size() > 0) {
						c.incrementCount(words.get(0) + "", 1);
					}

					if (words.size() > 1) {
						for (int j = 1; j < words.size(); j++) {
							c.incrementCount(StrUtils.join("_", words, j - 1, j + 1), 1);
							c.incrementCount(words.get(j), 1);
						}
					}
				}

				for (String word : c.keySet()) {
					cc.incrementCount(wordIndexer.getIndex(word), c.getCount(word));
				}
			}
			// cc.scale(kwd_freq);

			kwdCents.put(kwdid, VectorUtils.toSparseVector(cc));
		}

		TermWeighting.tfidf(kwdCents.values());

		SparseVector avgCent = VectorMath.average(kwdCents.values());

		Counter<Integer> ret = Generics.newCounter(kwdids.size());

		for (int kwdid : kwdCents.keySet()) {
			int kwd_freq = kwdData.getKeywordFreqs()[kwdid];
			ret.setCount(kwdid, kwd_freq * VectorMath.dotProduct(avgCent, kwdCents.get(kwdid)));
		}
		ret.normalize();
		return ret;
	}

	private double computeLetterCosine(int cid1, int cid2) {
		int[] cids = new int[] { cid1, cid2 };
		SparseVector[][] svss = new SparseVector[2][];

		for (int i = 0; i < cids.length; i++) {
			SparseVector[] svs = new SparseVector[2];

			Counter<Integer> c1 = new Counter<Integer>();
			Counter<Integer> c2 = new Counter<Integer>();

			for (int kwdid : clstToKwds.get(cids[i])) {
				StrPair kwdp = kwdIdx.getObject(kwdid);
				String korKey = normalize(kwdp.getFirst());
				String engKey = normalizeEnglish(kwdp.getSecond());

				int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

				korKey = UnicodeUtils.decomposeToJamoStr(korKey);
				engKey = normalizeEnglish(engKey).replace(" ", "");

				for (int j = 0; j < korKey.length(); j++) {
					c1.incrementCount((int) korKey.charAt(j), kwd_freq);
				}

				for (int j = 0; j < engKey.length(); j++) {
					c2.incrementCount((int) engKey.charAt(j), kwd_freq);
				}
			}

			svs[0] = VectorUtils.toSparseVector(c1);
			svs[1] = VectorUtils.toSparseVector(c2);

			for (SparseVector sv : svs) {
				VectorMath.unitVector(sv);
			}
			svss[i] = svs;
		}

		double kor_cosine = VectorMath.dotProduct(svss[0][0], svss[1][0]);
		double eng_cosine = VectorMath.dotProduct(svss[0][1], svss[1][1]);
		double cosine = ArrayMath.addAfterMultiply(kor_cosine, 0.5, eng_cosine);
		return cosine;
	}

	private void matchContexts() {
		System.out.println("match contexts.");

		int search_word_size = 10;
		int candidate_size = 100;

		for (int iter = 0; iter < 10; iter++) {

			int old_size = clstToKwds.size();

			Map<Integer, SparseVector> cents = Generics.newHashMap();

			SetMap<Integer, Integer> wordToClusters = Generics.newSetMap();

			for (Entry<Integer, Set<Integer>> e : clstToKwds.getEntrySet()) {
				int cid = e.getKey();
				Set<Integer> kwdids = e.getValue();

				Counter<Integer> c = Generics.newCounter();

				for (int kwdid : kwdids) {
					StrPair kwdp = kwdIdx.get(kwdid);
					SparseVector sv = kwdToWordCnts.get(kwdid);

					if (sv == null) {
						continue;
					}

					for (int w : sv.indexes()) {
						wordToClusters.put(w, cid);
					}
					VectorMath.add(sv, c);
				}

				cents.put(cid, VectorUtils.toSparseVector(c));
			}

			TermWeighting.tfidf(cents.values());

			Timer timer = Timer.newTimer();

			CounterMap<Integer, Integer> cm = Generics.newCounterMap();

			List<Integer> cids = Generics.newArrayList(cents.keySet());

			for (int i = 0; i < cids.size(); i++) {
				if ((i + 1) % 10000 == 0) {
					System.out.printf("\r[%d/%d, %s]", i + 1, cids.size(), timer.stop());
				}

				int qcid = cids.get(i);
				SparseVector qCent = cents.get(qcid);
				qCent.sortValues();

				// String qKwdStr = kwdIdx.get(qcid);

				Counter<Integer> toCompare = Generics.newCounter();

				for (int j = 0; j < qCent.size() && j < search_word_size; j++) {
					int w = qCent.indexAt(j);
					double weight = qCent.valueAt(j);

					Set<Integer> set = wordToClusters.get(w, false);

					if (set != null) {
						for (int tcid : set) {
							if (qcid == tcid || cm.containKey(tcid, qcid)) {

							} else {
								toCompare.incrementCount(tcid, weight);
							}
						}
					}
				}

				qCent.sortIndexes();

				if (toCompare.size() < 2) {
					continue;
				}

				List<Integer> keys = toCompare.getSortedKeys();

				for (int j = 0; j < keys.size() && j < candidate_size; j++) {
					int tcid = keys.get(j);
					double cosine = VectorMath.dotProduct(qCent, cents.get(tcid));

					if (cosine >= 0.9) {
						cm.incrementCount(tcid, qcid, cosine);
					}
				}
			}

			System.out.printf("\r[%d/%d, %s]\n", cids.size(), cids.size(), timer.stop());

			if (cm.size() == 0) {
				break;
			}

			merge(cm);

			int new_size = clstToKwds.size();

			System.out.printf("[%d -> %d clusters]\n", old_size, new_size);

			selectClusterLabels();

			if (write_temp_result) {
				writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + String.format("temp-title-loop-%d.txt.gz", iter));
			}

			// cents = null;
			// wordToClusters = null;
			// cm = null;
			// cids = null;
		}
	}

	// public class SearchWorker implements Callable<Counter<Integer>> {
	//
	// private List<Integer> query;
	//
	// private AtomicInteger q_cnt;
	//
	// private SetMap<Integer, Integer> iindex;
	//
	// @Override
	// public Counter<Integer> call() throws Exception {
	// int q_loc = 0;
	// Counter<Integer> ret = Generics.newCounter();
	//
	// while ((q_loc = q_cnt.getAndIncrement()) < query.size()) {
	// int qw = query.get(q_loc);
	//
	// Set<Integer> set = iindex.get(qw, false);
	//
	// if (set != null) {
	// for (int cid : set) {
	// if (qcid == cid || cm.containKey(cid, qcid)) {
	//
	// } else {
	// toCompare.incrementCount(cid, 1);
	// }
	// }
	// }
	// }
	//
	// return null;
	// }
	//
	// }

	private void matchEnglishGrams() {
		System.out.println("match English grams.");

		for (int iter = 0; iter < 10; iter++) {

			int old_size = clstToKwds.size();

			Map<Integer, SparseVector> engCents = Generics.newHashMap();

			SetMap<Integer, Integer> wordToClst = Generics.newSetMap();
			SetMap<Integer, Integer> clstToWord = Generics.newSetMap();

			Indexer<String> gramIndexer = Generics.newIndexer();

			GramGenerator gg = new GramGenerator(3);

			for (Entry<Integer, Set<Integer>> e : clstToKwds.getEntrySet()) {
				int cid = e.getKey();
				Set<Integer> kwdids = e.getValue();
				Counter<String> engWordCnts = Generics.newCounter();
				Counter<String> engGramCnts = Generics.newCounter();

				for (int kwdid : kwdids) {
					StrPair kwdp = kwdIdx.get(kwdid);
					String engKey = normalizeEnglish(kwdp.getSecond());

					int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

					for (String word : engKey.split(" ")) {
						engWordCnts.incrementCount(word, kwd_freq);
					}

					for (Gram g : gg.generateQGrams(engKey)) {
						engGramCnts.incrementCount(g.getString(), kwd_freq);
					}
				}

				if (engWordCnts.size() > 0 && engGramCnts.size() > 0) {
					Set<Integer> ws = Generics.newHashSet();

					for (int w : gramIndexer.getIndexes(engWordCnts.keySet())) {
						wordToClst.put(w, cid);
						ws.add(w);
					}

					clstToWord.put(cid, ws);
					engCents.put(cid, VectorUtils.toSparseVector(engGramCnts, gramIndexer, true));
				}
			}

			TermWeighting.tfidf(engCents.values());

			Timer timer = Timer.newTimer();

			CounterMap<Integer, Integer> TtoQ = Generics.newCounterMap();

			List<Integer> cids = Generics.newArrayList(engCents.keySet());

			for (int i = 0; i < cids.size(); i++) {
				if ((i + 1) % 10000 == 0) {
					System.out.printf("\r[%d/%d, %s]", i + 1, cids.size(), timer.stop());
				}

				int qid = cids.get(i);
				SparseVector qEngCent = engCents.get(qid);

				Counter<Integer> toCompare = Generics.newCounter();

				for (int w : clstToWord.get(qid)) {
					Set<Integer> tids = wordToClst.get(w, false);

					if (tids != null) {
						for (int tid : tids) {
							if (qid == tid || TtoQ.containKey(tid, qid)) {

							} else {
								toCompare.incrementCount(tid, 1);
							}
						}
					}
				}

				if (toCompare.size() < 2) {
					continue;
				}

				List<Integer> keys = toCompare.getSortedKeys();

				for (int j = 0; j < keys.size() && j < 100; j++) {
					int tid = keys.get(j);

					SparseVector tEngCent = engCents.get(tid);

					double eng_cosine = VectorMath.dotProduct(qEngCent, tEngCent);

					if (eng_cosine >= 0.9) {
						TtoQ.incrementCount(tid, qid, eng_cosine);
					}
				}
			}

			System.out.printf("\r[%d/%d, %s]\n", cids.size(), cids.size(), timer.stop());

			if (TtoQ.size() == 0) {
				break;
			}

			merge(TtoQ);

			int new_size = clstToKwds.size();

			System.out.printf("[%d -> %d clusters]\n", old_size, new_size);

			if (write_temp_result) {
				selectClusterLabels();
				writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + String.format("temp-eng-loop-%d.txt.gz", iter));
			}

			engCents.clear();
			wordToClst.clear();
			clstToWord.clear();

			gramIndexer.clear();

			TtoQ.clear();
			cids.clear();
		}
	}

	private void matchExactEnglish() {
		System.out.println("match exact English language.");

		int old_size = clstToKwds.size();

		SetMap<String, Integer> keyToClsts = Generics.newSetMap();

		for (Entry<Integer, Set<Integer>> e : clstToKwds.getEntrySet()) {
			int cid = e.getKey();
			Set<Integer> kwdids = e.getValue();

			boolean is_candidate = true;

			for (int kwdid : kwdids) {
				StrPair kwdp = kwdIdx.getObject(kwdid);
				String korKey = normalize(kwdp.getFirst());
				String engKey = normalize(kwdp.getSecond());

				if (korKey.length() == 0 && engKey.length() > 0) {

				} else {
					is_candidate = false;
					break;
				}
			}

			if (is_candidate) {
				StrPair kwdp = kwdIdx.getObject(cid);
				String korKey = normalize(kwdp.getFirst());
				String engKey = normalize(kwdp.getSecond());
				keyToClsts.put(engKey, cid);
			}
		}

		for (String key : keyToClsts.keySet()) {
			Set<Integer> cids = keyToClsts.get(key);
			if (cids.size() > 1) {

				for (int cid : cids) {
					for (int kwdid : clstToKwds.get(cid)) {
						StrPair kwdp = kwdIdx.getObject(kwdid);
						System.out.printf("%s, %s\n", key, kwdp);
					}
				}

				merge(cids);
			}
		}

		int new_size = clstToKwds.size();

		System.out.printf("[%d -> %d clusters]\n", old_size, new_size);
	}

	private void matchExactKorean() {
		System.out.println("match exact Korean language.");

		int old_size = clstToKwds.size();

		SetMap<String, Integer> keyToClsts = Generics.newSetMap();

		for (Entry<Integer, Set<Integer>> e : clstToKwds.getEntrySet()) {
			int cid = e.getKey();
			Set<Integer> kwdids = e.getValue();

			StrPair kwdp = kwdIdx.getObject(cid);
			String korKey = normalize(kwdp.getFirst());

			if (korKey.length() > 0) {
				keyToClsts.put(korKey, cid);
			}
		}

		for (String key : keyToClsts.keySet()) {
			Set<Integer> cids = keyToClsts.get(key);

			if (cids.size() > 1) {
				merge(cids);
			}
		}

		int new_size = clstToKwds.size();

		System.out.printf("[%d -> %d clusters]\n", old_size, new_size);
	}

	private void matchExactTwoLanguages() {
		System.out.println("match exact two languages.");

		int old_size = clstToKwds.size();

		SetMap<String, Integer> keyToClsts = Generics.newSetMap();

		for (int cid : clstToKwds.keySet()) {
			for (int kwdid : clstToKwds.get(cid)) {
				StrPair kwdp = kwdIdx.getObject(kwdid);
				String korKey = normalize(kwdp.getFirst());
				String engKey = normalizeEnglish(kwdp.getSecond()).replace(" ", "");
				String key = korKey + "\t" + engKey;
				keyToClsts.put(key, cid);
			}
		}

		for (String key : keyToClsts.keySet()) {
			Set<Integer> cids = keyToClsts.get(key);

			if (cids.size() > 1) {
				merge(cids);
			}
		}

		int new_size = clstToKwds.size();

		System.out.printf("[%d -> %d clusters]\n", old_size, new_size);
	}

	private void matchKoreanGrams() {
		System.out.println("match korean inputGrams.");

		for (int iter = 0; iter < 10; iter++) {

			int old_size = clstToKwds.size();

			Map<Integer, SparseVector> korCents = Generics.newHashMap();
			Map<Integer, SparseVector> engCents = Generics.newHashMap();

			SetMap<Integer, Integer> gramToClsts = Generics.newSetMap();
			SetMap<Integer, Integer> clstToGrams = Generics.newSetMap();

			Indexer<String> gramIdx = Generics.newIndexer();

			GramGenerator gg = new GramGenerator(3);

			for (Entry<Integer, Set<Integer>> e : clstToKwds.getEntrySet()) {
				int cid = e.getKey();
				Set<Integer> kwdids = e.getValue();
				Counter<String> korGramCnts = Generics.newCounter();
				Counter<String> korCharCnts = Generics.newCounter();
				Counter<String> engGramCnts = Generics.newCounter();

				for (int kwdid : kwdids) {
					StrPair kwdp = kwdIdx.get(kwdid);
					String korKey = normalize(kwdp.getFirst());
					String engKey = normalizeEnglish(kwdp.getSecond());
					int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

					if (!UnicodeUtils.isKorean(korKey)) {
						continue;
					}

					// String s = UnicodeUtils.decomposeToJamo(korKwd);

					for (char c : korKey.toCharArray()) {
						korCharCnts.incrementCount(c + "", kwd_freq);
					}

					for (Gram g : gg.generateQGrams(korKey)) {
						korGramCnts.incrementCount(g.getString(), kwd_freq);
					}

					for (Gram g : gg.generateQGrams(engKey)) {
						engGramCnts.incrementCount(g.getString(), kwd_freq);
					}
				}

				if (korGramCnts.size() > 0 && engGramCnts.size() > 0) {
					Set<Integer> gids = Generics.newHashSet();

					for (int gid : gramIdx.getIndexes(korGramCnts.keySet())) {
						gramToClsts.put(gid, cid);
						gids.add(gid);
					}

					clstToGrams.put(cid, gids);
					korCents.put(cid, VectorUtils.toSparseVector(korCharCnts, gramIdx, true));
					engCents.put(cid, VectorUtils.toSparseVector(engGramCnts, gramIdx, true));
				}
			}

			TermWeighting.tfidf(korCents.values());

			TermWeighting.tfidf(engCents.values());

			Timer timer = Timer.newTimer();

			CounterMap<Integer, Integer> TtoQ = Generics.newCounterMap();

			List<Integer> cids = Generics.newArrayList(korCents.keySet());

			for (int i = 0; i < cids.size(); i++) {
				if ((i + 1) % 10000 == 0) {
					System.out.printf("\r[%d/%d, %s]", i + 1, cids.size(), timer.stop());
				}

				int qid = cids.get(i);
				SparseVector qKorCent = korCents.get(qid);
				SparseVector qEngCent = engCents.get(qid);

				// String qKwdStr = kwdIdx.get(qcid);

				Counter<Integer> toCompare = Generics.newCounter();

				for (int gid : clstToGrams.get(qid)) {
					Set<Integer> set = gramToClsts.get(gid, false);

					if (set != null) {
						for (int cid : set) {
							if (qid == cid || TtoQ.containKey(cid, qid)) {

							} else {
								toCompare.incrementCount(cid, 1);
							}
						}
					}
				}

				if (toCompare.size() < 2) {
					continue;
				}

				List<Integer> keys = toCompare.getSortedKeys();

				for (int j = 0; j < keys.size(); j++) {
					int tid = keys.get(j);
					// String tKwdStr = kwdIdx.get(tcid);

					// if (queryToTargets.containKey(qcid, tcid) || queryToTargets.containKey(tcid, qcid)) {
					// continue;
					// }

					SparseVector tKorCent = korCents.get(tid);
					SparseVector tEngCent = engCents.get(tid);

					double kor_cosine = VectorMath.dotProduct(qKorCent, tKorCent);
					double eng_cosine = VectorMath.dotProduct(qEngCent, tEngCent);
					double cosine1 = ArrayMath.addAfterMultiply(kor_cosine, 0.5, eng_cosine);

					if (cosine1 >= 0.9) {
						TtoQ.incrementCount(tid, qid, cosine1);
						// cm.incrementCount(tcid, qcid, cosine1);
					} else if (cosine1 >= 0.75) {
						double cosine2 = computeLetterCosine(qid, tid);

						if (cosine2 >= 0.9) {
							// double tmp = ArrayMath.addAfterScale(cosine1, 0.5, cosine2);
							TtoQ.incrementCount(tid, qid, cosine2);
						}
					}
				}
			}

			System.out.printf("\r[%d/%d, %s]\n", cids.size(), cids.size(), timer.stop());

			if (TtoQ.size() == 0) {
				break;
			}

			merge(TtoQ);

			int new_size = clstToKwds.size();

			System.out.printf("[%d -> %d clusters]\n", old_size, new_size);

			if (write_temp_result) {
				selectClusterLabels();
				writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + String.format("temp-kor-loop-%d.txt.gz", iter));
			}

			korCents.clear();
			engCents.clear();

			gramToClsts.clear();
			clstToGrams.clear();

			gramIdx.clear();

			TtoQ.clear();
			cids.clear();
		}

	}

	private void merge(Collection<Integer> cids) {
		Set<Integer> kwds = Generics.newHashSet();

		for (int cid : cids) {
			Set<Integer> temp = clstToKwds.removeKey(cid);
			if (temp != null) {
				kwds.addAll(temp);
			}
		}

		int new_cid = min(cids);

		for (int cid : cids) {
			clstToParent.put(cid, new_cid);
		}

		clstToKwds.put(new_cid, kwds);

		for (int kwdid : kwds) {
			kwdToClst[kwdid] = new_cid;
		}
	}

	private void merge(CounterMap<Integer, Integer> TtoQ) {
		for (int tid : TtoQ.keySet()) {
			Counter<Integer> qids = TtoQ.getCounter(tid);
			int qid = qids.argMin();
			double score = qids.getCount(qid);
			qids.clear();
			qids.setCount(qid, score);
		}

		TtoQ = TtoQ.invert();

		for (int qid : TtoQ.keySet()) {
			Counter<Integer> tids = TtoQ.getCounter(qid);

			if (tids.size() > 1) {
				merge(tids.keySet());
			}
		}

	}

	private int min(Collection<Integer> cids) {
		int ret = Integer.MAX_VALUE;
		for (int cid : cids) {
			if (cid < ret) {
				ret = cid;
			}
		}
		return ret;
	}

	private void selectClusterLabels() {
		System.out.println("select cluster labels.");

		clstToLabel = Generics.newHashMap();

		for (int cid : clstToKwds.keySet()) {
			Counter<Integer> kwdScores = computeKeywordScores(clstToKwds.get(cid));

			List<Integer> kwdids = kwdScores.getSortedKeys();

			int label = -1;

			for (int i = 0; i < kwdids.size(); i++) {
				int kwdid = kwdids.get(i);
				int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

				StrPair kwdp = kwdIdx.get(kwdid);

				if (kwd_freq < 2) {
					continue;
				}

				String engKwd = kwdp.getSecond().replaceAll("[\\s\\p{Punct}]+", "");

				if (!StrUtils.isUppercase(engKwd)) {
					label = kwdid;
					break;
				}
			}

			if (label == -1) {
				clstToLabel.put(cid, kwdScores.argMax());
			} else {
				clstToLabel.put(cid, label);
			}

		}
	}

	public void setTitleData(CounterMap<String, String> cm) {
		kwdToWordCnts = Generics.newHashMap();
		wordIdx = Generics.newIndexer();

		for (int kwdid : kwdData.getKeywordToDocs().keySet()) {
			Counter<String> c = Generics.newCounter();
			for (int docid : kwdData.getKeywordToDocs().get(kwdid)) {
				String cn = kwdData.getDocumentIndxer().getObject(docid);
				c.incrementAll(cm.getCounter(cn));
			}
			if (c.size() > 0) {
				kwdToWordCnts.put(kwdid, VectorUtils.toSparseVector(c, wordIdx, true));
			}
		}
	}

	public void writeClusters(String fileName) {
		TextFileWriter writer = new TextFileWriter(fileName);

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

		for (int i = 0, n = 1; i < cids.size(); i++) {
			int cid = cids.get(i);
			int label = clstToLabel.get(cid);
			StrPair kwdp = kwdIdx.getObject(label);

			StringBuffer sb = new StringBuffer();
			sb.append(String.format("No:\t%d", n));
			sb.append(String.format("\nID:\t%d", cid));

			sb.append(String.format("\nLabel:\n%d\t%s", label, kwdp.join("\t")));
			sb.append(String.format("\nKeywords:\t%d", clstToKwds.get(cid).size()));

			Counter<Integer> c = Generics.newCounter();

			for (int kwdid : clstToKwds.get(cid)) {
				c.setCount(kwdid, kwdData.getKeywordFreqs()[kwdid]);
			}

			n++;

			List<Integer> kwids = c.getSortedKeys();
			for (int j = 0; j < kwids.size(); j++) {
				int kwid = kwids.get(j);
				int kw_freq = kwdData.getKeywordFreqs()[kwid];
				sb.append(String.format("\n%d:\t%d\t%s\t%d", j + 1, kwid, kwdIdx.getObject(kwid), kw_freq));
			}
			writer.write("\n\n" + sb.toString());
		}
		writer.close();

		System.out.printf("write [%d] clusters at [%s]\n", clstToKwds.size(), fileName);
	}

}
