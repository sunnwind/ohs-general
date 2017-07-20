package ohs.eden.keyphrase.cluster;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

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

public class NewKeywordClusterer {

	private class EnglishGramMatchWorker implements Callable<CounterMap<Integer, Integer>> {

		private SetMap<Integer, Integer> wordToClsts;

		private SetMap<Integer, Integer> clstToWords;

		private Map<Integer, SparseVector> engCents;

		private List<Integer> cids;

		private AtomicInteger c_cnt;

		public EnglishGramMatchWorker(SetMap<Integer, Integer> wordToClst, SetMap<Integer, Integer> clstToWord,
				Map<Integer, SparseVector> engCents, List<Integer> cids, AtomicInteger c_cnt) {
			super();
			this.wordToClsts = wordToClst;
			this.clstToWords = clstToWord;
			this.engCents = engCents;
			this.cids = cids;
			this.c_cnt = c_cnt;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {

			int c_loc = 0;
			CounterMap<Integer, Integer> QtoT = Generics.newCounterMap();
			Counter<Integer> candis = Generics.newCounter();
			int num_clusters = clstToKwds.size();

			while ((c_loc = c_cnt.getAndIncrement()) < cids.size()) {
				int qid = cids.get(c_loc);
				StrPair p1 = kwdIdx.getObject(qid);
				SparseVector qEngCent = engCents.get(qid);

				if (c_loc % 50000 == 0) {
					System.out.printf("[%d/%d]\n", c_loc, cids.size());
				}

				candis.clear();

				SparseVector q = getWeightedQuery(qid, wordToClsts, clstToWords, num_clusters);
				q.sortValues();

				for (int i = 0; i < q.size() && i < query_size; i++) {
					int w = q.indexAt(i);
					Set<Integer> cids = wordToClsts.get(w, false);

					if (cids == null) {
						continue;
					}

					for (int cid : cids) {
						if (qid != cid) {
							double idf = TermWeighting.idf(num_clusters, cids.size());
							candis.incrementCount(cid, idf);
						}
					}
				}

				if (candis.size() < 2) {
					continue;
				}

				List<Integer> keys = candis.getSortedKeys();

				for (int i = 0; i < keys.size() && i < candidate_size; i++) {
					int tid = keys.get(i);
					SparseVector tEngCent = engCents.get(tid);
					StrPair p2 = kwdIdx.getObject(tid);
					double eng_cosine = VectorMath.dotProduct(qEngCent, tEngCent);

					if (eng_cosine >= 0.9) {
						QtoT.incrementCount(qid, tid, eng_cosine);
						// cm.incrementCount(tcid, qcid, cosine1);
					}
				}
			}
			return QtoT;
		}
	}

	private class KoranGramMatchWorker implements Callable<CounterMap<Integer, Integer>> {

		private SetMap<Integer, Integer> featToClsts;

		private SetMap<Integer, Integer> clstToFeats;

		private Map<Integer, SparseVector> korCents;

		private Map<Integer, SparseVector> engCents;

		private List<Integer> cids;

		private AtomicInteger c_cnt;

		public KoranGramMatchWorker(SetMap<Integer, Integer> featToClst, SetMap<Integer, Integer> clstToFeat,
				Map<Integer, SparseVector> korCents, Map<Integer, SparseVector> engCents, List<Integer> cids, AtomicInteger c_cnt) {
			super();
			this.featToClsts = featToClst;
			this.clstToFeats = clstToFeat;
			this.korCents = korCents;
			this.engCents = engCents;
			this.cids = cids;
			this.c_cnt = c_cnt;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {

			int c_loc = 0;

			CounterMap<Integer, Integer> QtoT = Generics.newCounterMap();
			int num_clusters = clstToKwds.size();

			while ((c_loc = c_cnt.getAndIncrement()) < cids.size()) {
				int qid = cids.get(c_loc);
				StrPair p1 = kwdIdx.getObject(qid);
				SparseVector qKorCent = korCents.get(qid);
				SparseVector qEngCent = engCents.get(qid);

				SparseVector q = getWeightedQuery(qid, featToClsts, clstToFeats, num_clusters);
				q.sortValues();

				Counter<Integer> candis = Generics.newCounter();

				for (int i = 0; i < q.size() && i < query_size; i++) {
					int f = q.indexAt(i);
					Set<Integer> cids = featToClsts.get(f, false);

					if (cids == null) {
						continue;
					}

					for (int cid : cids) {
						if (qid != cid) {
							double idf = TermWeighting.idf(num_clusters, cids.size());
							candis.incrementCount(cid, idf);
						}
					}
				}

				if (candis.size() < 2) {
					continue;
				}

				List<Integer> keys = candis.getSortedKeys();

				for (int i = 0; i < keys.size() && i < candidate_size; i++) {
					int tid = keys.get(i);
					SparseVector tKorCent = korCents.get(tid);
					SparseVector tEngCent = engCents.get(tid);

					StrPair p2 = kwdIdx.getObject(tid);

					double kor_cosine = VectorMath.dotProduct(qKorCent, tKorCent);
					double eng_cosine = VectorMath.dotProduct(qEngCent, tEngCent);
					double cosine1 = ArrayMath.addAfterMultiply(kor_cosine, 0.5, eng_cosine);

					if (cosine1 >= 0.9) {
						QtoT.incrementCount(qid, tid, cosine1);
						// cm.incrementCount(tcid, qcid, cosine1);
					} else if (cosine1 >= 0.75) {
						double cosine2 = computeSyllableScore(qid, tid);

						if (cosine2 >= 0.9) {
							// double tmp = ArrayMath.addAfterScale(cosine1, 0.5, cosine2);
							QtoT.incrementCount(qid, tid, cosine2);
						}
					}
				}
			}
			return QtoT;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.printf("[%s] begins.\n", NewKeywordClusterer.class.getName());

		KeywordData kwdData = new KeywordData();

		kwdData.add(KPPath.KYP_2P_FILE);
		// kwdData.add(KPPath.KYP_WOS_FILE);

		// kwdData.add(KPPath);
		// kwdData.writeObject(KPPath.KYP_DATA_SER_FILE);

		NewKeywordClusterer kc = new NewKeywordClusterer(kwdData);
		// kc.setTitleData(FileUtils.readStrCounterMap(KPPath.KYP_TITLE_DATA_FILE));
		kc.cluster();
		kc.writeClusters(KPPath.KP_DIR + "kwd_2p-wos_clusters.txt.gz");

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

	private int query_size = 20;

	private int candidate_size = 1000;

	private int thread_size = 10;

	private Map<Integer, SparseVector> kwdToWordCnts;

	private Indexer<String> wordIdx;

	private KeywordData kwdData;

	private Indexer<StrPair> kwdIdx;

	private SetMap<Integer, Integer> clstToKwds;

	private Map<Integer, Integer> clstToParent;

	private int[] kwdToClst;

	private Map<Integer, Integer> clstToLabel;

	private boolean write_temp_result = false;

	public NewKeywordClusterer(KeywordData kwdData) {
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

	private double computeSyllableScore(int cid1, int cid2) {
		int[] cids = new int[] { cid1, cid2 };
		SparseVector[][] svss = new SparseVector[2][];

		for (int i = 0; i < cids.length; i++) {
			SparseVector[] svs = new SparseVector[2];

			Counter<Integer> c1 = Generics.newCounter();
			Counter<Integer> c2 = Generics.newCounter();

			for (int kwdid : clstToKwds.get(cids[i])) {
				StrPair kwdp = kwdIdx.getObject(kwdid);
				String korKey = normalize(kwdp.getFirst());
				String engKey = normalizeEnglish(kwdp.getSecond());

				korKey = UnicodeUtils.decomposeToJamoStr(korKey);

				int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

				for (int j = 0; j < korKey.length(); j++) {
					c1.incrementCount((int) korKey.charAt(j), 1);
				}

				for (int j = 0; j < engKey.length(); j++) {
					c2.incrementCount((int) engKey.charAt(j), 1);
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

	private SetMap<Integer, Integer> filterFrequentWords(SetMap<Integer, Integer> wordToClsts, int num_clusters) {
		int old_size = wordToClsts.size();
		SetMap<Integer, Integer> ret = Generics.newSetMap();
		for (int w : wordToClsts.keySet()) {
			Set<Integer> clsts = wordToClsts.get(w);
			double ratio = 1f * clsts.size() / num_clusters;
			if (ratio < 0.5) {
				ret.put(w, clsts);
			}
		}
		int new_size = ret.size();
		// System.out.printf("[%d -> %d]\n", old_size, new_size);
		return ret;

	}

	public SparseVector getWeightedQuery(int qid, SetMap<Integer, Integer> featToClsts, SetMap<Integer, Integer> clstToFeats,
			int num_clusters) {
		Set<Integer> feats = clstToFeats.get(qid);
		SparseVector q = new SparseVector(feats.size());
		int loc = 0;
		for (int f : feats) {
			Set<Integer> clsts = featToClsts.get(f, false);
			if (clsts == null) {
				continue;
			}
			double idf = TermWeighting.idf(num_clusters, clsts.size());
			q.setAt(loc++, f, idf);
		}
		return q;
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

	private void matchEnglishGrams() throws Exception {
		System.out.println("match English grams.");

		GramGenerator gg = new GramGenerator(3);

		for (int iter = 0; iter < 100; iter++) {
			Timer timer = Timer.newTimer();
			int old_size = clstToKwds.size();

			Map<Integer, SparseVector> engCents = Generics.newHashMap();
			SetMap<Integer, Integer> wordToClsts = Generics.newSetMap();
			SetMap<Integer, Integer> clstToWords = Generics.newSetMap();
			Indexer<String> featIndexer = Generics.newIndexer();
			CounterMap<Integer, Integer> QtoT = Generics.newCounterMap();

			int num_clusters = 0;

			for (Entry<Integer, Set<Integer>> e : clstToKwds.getEntrySet()) {
				int cid = e.getKey();
				Set<Integer> kwdids = e.getValue();
				Counter<String> engWordCnts = Generics.newCounter();
				Counter<String> engFeatCnts = Generics.newCounter();

				for (int kwdid : kwdids) {
					StrPair kwdp = kwdIdx.get(kwdid);
					String engKey = normalizeEnglish(kwdp.getSecond());

					int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

					for (String word : engKey.split(" ")) {
						engWordCnts.incrementCount(word, 1);
					}

					engFeatCnts.incrementAll(engWordCnts);

					for (Gram g : gg.generateQGrams(engKey)) {
						engFeatCnts.incrementCount(g.getString(), 1);
					}

				}

				if (engWordCnts.size() > 0 && engFeatCnts.size() > 0) {
					num_clusters++;
					for (int w : featIndexer.getIndexes(engWordCnts.keySet())) {
						wordToClsts.put(w, cid);
					}
					engCents.put(cid, VectorUtils.toSparseVector(engFeatCnts, featIndexer, true));
				}
			}

			wordToClsts = filterFrequentWords(wordToClsts, num_clusters);
			clstToWords = wordToClsts.invert();

			TermWeighting.tfidf(engCents.values());

			ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);
			List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList();
			List<Integer> cids = Generics.newArrayList(clstToWords.keySet());
			AtomicInteger c_cnt = new AtomicInteger(0);

			for (int i = 0; i < thread_size; i++) {
				fs.add(tpe.submit(new EnglishGramMatchWorker(wordToClsts, clstToWords, engCents, cids, c_cnt)));
			}

			QtoT.clear();

			for (int i = 0; i < thread_size; i++) {
				QtoT.incrementAll(fs.get(i).get());
			}

			tpe.shutdown();

			if (QtoT.size() == 0) {
				break;
			}

			merge(QtoT);

			int new_size = clstToKwds.size();

			System.out.printf("[%d -> %d clusters, %s]\n", old_size, new_size, timer.stop());

			if (write_temp_result) {
				selectClusterLabels();
				writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + String.format("temp-eng-loop-%d.txt.gz", iter));
			}

			if (old_size == new_size) {
				break;
			}
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

	private void matchKoreanGrams() throws Exception {
		System.out.println("match korean grams.");

		GramGenerator gg = new GramGenerator(3);

		for (int iter = 0; iter < 100; iter++) {
			Timer timer = Timer.newTimer();

			int old_size = clstToKwds.size();
			Map<Integer, SparseVector> korCents = Generics.newHashMap();
			Map<Integer, SparseVector> engCents = Generics.newHashMap();
			SetMap<Integer, Integer> gramToClst = Generics.newSetMap();
			SetMap<Integer, Integer> clstToFeat = Generics.newSetMap();
			Indexer<String> featIndexer = Generics.newIndexer();
			CounterMap<Integer, Integer> QtoT = Generics.newCounterMap();

			int num_clusters = 0;

			for (Entry<Integer, Set<Integer>> e : clstToKwds.getEntrySet()) {
				int cid = e.getKey();
				Set<Integer> kwdids = e.getValue();
				Counter<String> korFeatCnts = Generics.newCounter();
				Counter<String> engFeatCnts = Generics.newCounter();
				Set<Integer> grams = Generics.newHashSet();

				for (int kwdid : kwdids) {
					StrPair kwdp = kwdIdx.get(kwdid);
					String korKey = normalize(kwdp.getFirst());
					String engKey = normalizeEnglish(kwdp.getSecond());
					int kwd_freq = kwdData.getKeywordFreqs()[kwdid];

					if (!UnicodeUtils.isKorean(korKey)) {
						continue;
					}

					for (Character c : korKey.toCharArray()) {
						String s = c.toString().trim();
						if (!s.equals(" ")) {
							korFeatCnts.incrementCount(c + "", 1);
						}
					}

					for (String word : engKey.split(" ")) {
						engFeatCnts.incrementCount(word, 1);
					}

					for (String tok : korKey.split(" ")) {
						for (Gram g : gg.generateQGrams(tok)) {
							grams.add(featIndexer.getIndex(g.getString()));
						}
					}
				}

				if (grams.size() > 0 && korFeatCnts.size() > 0 && engFeatCnts.size() > 0) {
					korCents.put(cid, VectorUtils.toSparseVector(korFeatCnts, featIndexer, true));
					engCents.put(cid, VectorUtils.toSparseVector(engFeatCnts, featIndexer, true));
					num_clusters++;

					for (int gid : grams) {
						gramToClst.put(gid, cid);
					}
				}
			}

			gramToClst = filterFrequentWords(gramToClst, num_clusters);
			clstToFeat = gramToClst.invert();

			TermWeighting.tfidf(korCents.values());
			TermWeighting.tfidf(engCents.values());

			ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);
			List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList();
			List<Integer> cids = Generics.newArrayList(clstToFeat.keySet());
			AtomicInteger c_cnt = new AtomicInteger(0);

			for (int i = 0; i < thread_size; i++) {
				fs.add(tpe.submit(new KoranGramMatchWorker(gramToClst, clstToFeat, korCents, engCents, cids, c_cnt)));
			}

			QtoT.clear();

			for (int i = 0; i < thread_size; i++) {
				QtoT.incrementAll(fs.get(i).get());
			}

			tpe.shutdown();

			if (QtoT.size() == 0) {
				break;
			}

			merge(QtoT);

			int new_size = clstToKwds.size();

			System.out.printf("[%d -> %d clusters, %s]\n", old_size, new_size, timer.stop());

			if (write_temp_result) {
				selectClusterLabels();
				writeClusters(KPPath.KYP_CLUSTER_TEMP_DIR + String.format("temp-kor-loop-%d.txt.gz", iter));
			}

			korCents.clear();
			engCents.clear();

			gramToClst.clear();
			clstToFeat.clear();

			featIndexer.clear();

			if (old_size == new_size) {
				break;
			}
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

	private void merge(CounterMap<Integer, Integer> QtoT) {
		CounterMap<Integer, Integer> TtoQ = QtoT.invert();

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
