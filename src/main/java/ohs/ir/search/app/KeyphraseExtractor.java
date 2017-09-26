package ohs.ir.search.app;

import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.search.model.ParsimoniousLanguageModelEstimator;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class KeyphraseExtractor {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsWeights = Generics.newCounter();

		{
			List<String> lines = FileUtils.readLinesFromText(MIRPath.DATA_DIR + "phrs/phrs_cnt.txt");
			phrsWeights = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				double weight = Double.parseDouble(ps[1]);
				phrsWeights.setCount(ps[0], weight);
			}
		}

		// DocumentSearcher ds = new DocumentSearcher(MIRPath.DATA_DIR +
		// "merged/col/dc/", MIRPath.STOPWORD_INQUERY_FILE);
		// DocumentCollection dc = new DocumentCollection(MIRPath.DATA_DIR +
		// "merged/col/dc/");
		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);

		int dseq = 1000;

		Pair<String, IntegerArray> p = dc.get(dseq);

		IntegerMatrix d = DocumentCollection.toMultiSentences(p.getSecond());

		List<List<String>> doc = Generics.newArrayList();

		for (IntegerArray s : d) {
			List<String> words = Generics.newArrayList(s.size());
			for (int w : s) {
				words.add(dc.getVocab().getObject(w));
			}
			doc.add(words);
		}

		String text = DocumentCollection.toText(dc.getVocab(), d);
		System.out.println(text);

		KeyphraseExtractor ke = new KeyphraseExtractor(dc, null, phrsWeights);
		// ke.extract(StrUtils.split(text));
		// ke.extractTFIDF(StrUtils.split(text));
		ke.extractLM(StrUtils.split(text));

		// biases.set(dc.getVocab().indexOf("cancer"), 1);
		// biases.set(dc.getVocab().indexOf("breast"), 1);
		// biases.set(dc.getVocab().indexOf("risk"), 1);
		// biases.set(dc.getVocab().indexOf("cohort"), 1);

		System.out.println("process ends.");
	}

	private DocumentCollection dc;

	private Counter<String> dv;

	private Counter<String> phrsBiases;

	private Indexer<String> phrsIdxer;

	private PhraseMapper<String> pm;

	private Vocab vocab;

	private WordFilter wf;

	private int window_size = 5;

	public KeyphraseExtractor(DocumentCollection dc, WordFilter wf, Counter<String> phrsBiases) {
		this.dc = dc;
		this.wf = wf;
		this.vocab = dc.getVocab();
		this.phrsBiases = phrsBiases;

		phrsIdxer = Generics.newIndexer(phrsBiases.size());

		{
			Trie<String> trie = new Trie<String>();

			for (String phrs : phrsBiases.keySet()) {
				List<String> words = StrUtils.split(phrs);

				Node<String> node = trie.insert(words);
				node.setFlag(true);
				phrsIdxer.add(phrs);
			}
			trie.trimToSize();
			pm = new PhraseMapper<>(trie);
		}

	}

	private double computeKLDScore(DenseVector q, DenseVector psg, DenseVector d, DenseVector c) {
		double ret = 0;
		double prior_dir = 1000;
		double mixture_jm = 0.5;

		// for (int w = 0; w < q.size(); w++) {
		// double pr_w_in_q = q.prob(w);
		// if (pr_w_in_q == 0) {
		// continue;
		// }
		// double cnt_w_in_p = psg.value(w);
		// double len_p = psg.sum();
		// double pr_w_in_p = cnt_w_in_p / len_p;
		// double pr_w_in_c = c.prob(w);
		// double cnt_w_in_d = d.value(w);
		// double len_d = d.sum();
		// // double pr_w_in_d = cnt_w_in_d / len_d;
		//
		// double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d,
		// pr_w_in_c, prior_dir, pr_w_in_p,
		// mixture_jm);
		//
		// ret += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d);
		// }

		for (int w = 0; w < q.size(); w++) {
			double pr_w_in_q = q.prob(w);
			if (pr_w_in_q == 0) {
				continue;
			}
			double cnt_w_in_p = psg.value(w);
			double len_p = psg.sum();
			double pr_w_in_p = cnt_w_in_p / len_p;

			double pr_w_in_c = c.prob(w);
			double cnt_w_in_d = d.value(w);
			double len_d = d.sum();

			pr_w_in_p = TermWeighting.jelinekMercerSmoothing(pr_w_in_p, pr_w_in_c, mixture_jm);

			ret += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_p);
		}

		ret = Math.exp(-ret);
		return ret;
	}

	public Counter<String> extract(String d) {

		return null;
	}

	public SparseVector extractLM(List<String> d) {
		// System.out.println(candCnts.toStringSortedByValues(true, true,
		// candCnts.size(), "\t"));

		List<IntPair> ps = pm.map(d);

		CounterMap<String, String> phrsSims = Generics.newCounterMap();
		CounterMap<String, String> wordSims = Generics.newCounterMap();

		Counter<String> phrsCnts = Generics.newCounter();
		Counter<String> wordCnts = Generics.newCounter();

		{
			for (IntPair p : ps) {
				String phrs = StrUtils.join("_", d.subList(p.getFirst(), p.getSecond()));
				phrsCnts.incrementCount(phrs, 1);

				int lower_bound = Math.max(0, p.getFirst() - window_size);
				int upper_bound = Math.min(p.getSecond() + window_size, d.size());

				for (int i = lower_bound; i < p.getFirst(); i++) {
					double dist = p.getFirst() - i;
					double score = 1d / dist;
					String word = d.get(i);
					phrsSims.incrementCount(phrs, word, score);
				}

				for (int i = p.getSecond(); i < upper_bound; i++) {
					double dist = i - p.getSecond() + 1;
					double score = 1d / dist;
					String word = d.get(i);
					phrsSims.incrementCount(phrs, word, score);
				}
			}

			for (int i = 0; i < d.size(); i++) {
				String word1 = d.get(i);
				wordCnts.incrementCount(word1, 1);

				double score1 = 1d / (i + 1);
				phrsBiases.incrementCount(word1, score1);

				int m = Math.min(i + window_size, d.size());

				for (int j = i + 1; j < m; j++) {
					String word2 = d.get(j);
					double dist = j - i;
					double score2 = 1d / dist;
					wordSims.incrementCount(word1, word2, score2);
				}
			}
		}

		Indexer<String> phrsIdxer = Generics.newIndexer(phrsSims.keySet());
		Indexer<String> wordIdxer = Generics.newIndexer(wordCnts.keySet());

		DenseVector dv = new DenseVector(wordIdxer.size());
		DenseVector cv = new DenseVector(wordIdxer.size());

		for (int w = 0; w < wordIdxer.size(); w++) {
			String word = wordIdxer.getObject(w);
			dv.add(w, wordCnts.getCount(word));
			cv.add(w, vocab.getProb(w));
		}

		{
			DenseVector dv2 = dv.copy();
			double prior_dir = 2000;
			double mixture_jm = 0.5;

			for (int i = 0; i < dv2.size(); i++) {
				double cnt_w_in_d = dv2.value(i);
				double len_d = dv2.sum();
				double pr_w_in_c = cv.value(i);
				double pr_w_in_d = TermWeighting.twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c,
						mixture_jm);
				dv2.set(i, pr_w_in_d);
			}

			DenseMatrix T2 = VectorUtils.toDenseMatrix(wordSims, wordIdxer, wordIdxer, false);
			T2.normalizeColumns();

			DenseVector dv3 = new DenseVector(dv2.size());

			for (int w = 0; w < dv2.size(); w++) {
				DenseVector trm = T2.row(w);
				double pr = VectorMath.dotProduct(trm, dv2);
				dv3.add(w, pr);
			}

			SparseVector dv4 = VectorUtils.toSparseVector(wordCnts, vocab);

			ParsimoniousLanguageModelEstimator plme = new ParsimoniousLanguageModelEstimator(dc);

			SparseVector dv5 = plme.estimate(dv4);

			System.out.println(VectorUtils.toCounter(dv, wordIdxer));
			System.out.println(VectorUtils.toCounter(dv2, wordIdxer));
			System.out.println(VectorUtils.toCounter(dv3, wordIdxer));
			System.out.println(VectorUtils.toCounter(dv5, vocab));
			System.out.println();

			System.exit(0);
		}

		DenseMatrix T1 = VectorUtils.toDenseMatrix(phrsSims, phrsIdxer, wordIdxer, false);
		DenseMatrix T2 = VectorUtils.toDenseMatrix(wordSims, wordIdxer, wordIdxer, false);

		double mixture_jm = 0.5;

		for (int i = 0; i < T1.rowSize(); i++) {
			DenseVector t = T1.row(i);
			for (int j = 0; j < t.size(); j++) {
				double pr_w_in_p = t.value(j);
				double pr_w_in_c = cv.prob(j);
				double pr = TermWeighting.jelinekMercerSmoothing(pr_w_in_p, pr_w_in_c, mixture_jm);
				t.set(j, pr);
			}
			t.summation();
		}

		DenseMatrix T3 = new DenseMatrix(T1.rowSize(), T2.colSize());

		for (int i = 0; i < T1.rowSize(); i++) {
			DenseVector lm_d = T1.row(i);
			DenseVector lm_d2 = T3.row(i);

			for (int w = 0; w < lm_d.size(); w++) {
				DenseVector trm = T2.row(w);
				double pr = VectorMath.dotProduct(trm, lm_d);
				lm_d2.add(w, pr);
			}
		}

		DenseMatrix T4 = new DenseMatrix(phrsIdxer.size());

		for (int i = 0; i < T3.rowSize(); i++) {
			DenseVector t1 = T3.row(i);

			for (int j = i + 1; j < T3.rowSize(); j++) {
				DenseVector t2 = T3.row(j);

				double sim = VectorMath.cosine(t1, t2);

				sim = Math.exp(sim);

				T4.add(i, j, sim);
				T4.add(j, i, sim);
			}
		}

		// CounterMap<String, String> cm = VectorUtils.toCounterMap(T2, phrsIdxer,
		// phrsIdxer);

		T4.normalizeColumns();

		DenseVector cents = new DenseVector(phrsIdxer.size());
		DenseVector biases = cents.copy();
		DenseVector biases2 = cents.copy();

		Set<String> mWords = Generics.newHashSet();
		mWords.add("cancer");
		mWords.add("cancers");
		mWords.add("tumor");
		mWords.add("tumors");
		mWords.add("breast");

		for (int i = 0; i < biases.size(); i++) {
			int pid = biases.indexAt(i);
			String phrs = phrsIdxer.getObject(pid);
			double bias = phrsBiases.getCount(phrs);
			double bias2 = 0;

			// if (!mWords.contains(phrs)) {
			// bias = 0;
			// }

			biases.add(i, bias);
			biases2.add(i, bias2);
		}

		// biases.normalize();
		// biases2.normalize();

		biases2.setAll(0);

		for (IntPair p : ps) {
			String phrs = StrUtils.join("_", d.subList(p.getFirst(), p.getSecond()));
			int pos = p.getFirst();
			int pid = phrsIdxer.indexOf(phrs);
			biases2.add(pid, 1d / (pos + 1));
		}

		// ArrayMath.addAfterMultiply(biases.values(), 0.5, biases2.values(), 0.5,
		// biases2.values());
		// ArrayMath.multiply(biases.values(), biases2.values(), biases2.values());

		biases2.normalizeAfterSummation();

		ArrayMath.randomWalk(T4.values(), cents.values(), null, 500, 10);
		// ArrayMath.randomWalk(T2.values(), cents.values(), null, 500, 0.0000001, 0);

		// Set<Integer> toKeep = Generics.newHashSet();
		//
		// for (int i = 0; i < cents.size(); i++) {
		// int pid = cents.indexAt(i);
		// double score = cents.valueAt(i);
		// String phrs = phrsIdxer.getObject(i);
		//
		// if (phrsCnts.containsKey(phrs)) {
		// toKeep.add(pid);
		// }
		// }

		// cents.pruneExcept(toKeep);
		// System.out.println(phrsWeights2.toStringSortedByValues(true, true, 20,
		// "\t"));
		// System.out.println(phrsCnts.toStringSortedByValues(true, true,
		// phrsCnts.size(), "\t"));
		// System.out.println(VectorUtils.toCounter(biases,
		// dc.getVocab()).toStringSortedByValues(true, true, 20, "\t"));
		System.out.println(VectorUtils.toCounter(cents, phrsIdxer).toStringSortedByValues(true, true, 50, "\t"));

		return null;
	}

	private CounterMap<String, String> getPhraseToWords(List<String> d, List<IntPair> ps) {
		CounterMap<String, String> ret = Generics.newCounterMap();

		for (IntPair p : ps) {
			String phrs = StrUtils.join("_", d.subList(p.getFirst(), p.getSecond()));

			int lower_boud = Math.max(0, p.getFirst() - window_size);
			int upper_bound = Math.min(p.getSecond() + window_size, d.size());

			for (int i = lower_boud; i < p.getFirst(); i++) {
				double dist = p.getFirst() - i;
				double score = 1d / dist;
				String word = d.get(i);
				ret.incrementCount(phrs, word, score);
			}

			for (int i = p.getSecond(); i < upper_bound; i++) {
				double dist = i - p.getSecond() + 1;
				double score = 1d / dist;
				String word = d.get(i);
				ret.incrementCount(phrs, word, score);
			}
		}
		return ret;
	}

	private CounterMap<String, String> getWordToWords(List<String> d) {
		CounterMap<String, String> ret = Generics.newCounterMap();
		for (int i = 0; i < d.size(); i++) {
			String word1 = d.get(i);

			double score1 = 1d / (i + 1);
			phrsBiases.incrementCount(word1, score1);

			int m = Math.min(i + window_size, d.size());

			for (int j = i + 1; j < m; j++) {
				String word2 = d.get(j);
				double dist = j - i;
				double score2 = 1d / dist;
				ret.incrementCount(word1, word2, score2);
			}
		}
		return ret;
	}

	private Counter<String> getPhraseCounts(List<String> d, List<IntPair> ps) {
		Counter<String> ret = Generics.newCounter();
		for (IntPair p : ps) {
			String phrs = StrUtils.join("_", d.subList(p.getFirst(), p.getSecond()));
			ret.incrementCount(phrs, 1);
		}
		return ret;
	}

	private DenseVector getWordWeights(Indexer<String> wordIdxer, Counter<String> wordCnts) {
		DenseVector ret = new DenseVector(wordIdxer.size());

		for (int w = 0; w < wordIdxer.size(); w++) {
			String word = wordIdxer.getObject(w);
			double doc_freq = vocab.getDocFreq(word);
			double cnt = wordCnts.getCount(word);
			double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), doc_freq);
			ret.add(w, tfidf);
		}
		return ret;
	}

	private DenseVector getPhraseWeights(Indexer<String> phrsIdxer, Indexer<String> wordIdxer,
			DenseVector wordWeights) {
		DenseVector ret = new DenseVector(phrsIdxer.size());
		for (int p = 0; p < phrsIdxer.size(); p++) {
			String phrs = phrsIdxer.getObject(p);
			double weight = 0;
			List<String> words = StrUtils.split("_", phrs);
			for (String word : words) {
				int w = wordIdxer.indexOf(word);
				weight += wordWeights.value(w);
			}
			ret.add(p, weight);
		}
		return ret;
	}

	public SparseVector extractTFIDF(List<String> d) {
		List<IntPair> ps = pm.map(d);
		CounterMap<String, String> phrsSims = getPhraseToWords(d, ps);
		CounterMap<String, String> wordSims = getWordToWords(d);
		Counter<String> phrsCnts = getPhraseCounts(d, ps);
		Counter<String> wordCnts = Generics.newCounter(d);

		Counter<String> phrsBiases = Generics.newCounter();

		Indexer<String> phrsIdxer = Generics.newIndexer(phrsCnts.keySet());
		Indexer<String> wordIdxer = Generics.newIndexer(wordCnts.keySet());

		DenseVector wordWeights = getWordWeights(wordIdxer, wordCnts);
		DenseVector phrsWeights = getPhraseWeights(phrsIdxer, wordIdxer, wordWeights);
		DenseVector pBiases = new DenseVector(phrsIdxer.size());

		// System.out.println(cm1.toString());

		DenseMatrix T1 = new DenseMatrix(phrsIdxer.size(), wordIdxer.size());

		for (String phrs : phrsSims.keySet()) {
			int p = phrsIdxer.indexOf(phrs);
			double p_weight = phrsWeights.value(p);
			Counter<String> c = phrsSims.getCounter(phrs);

			for (Entry<String, Double> e : c.entrySet()) {
				String word = e.getKey();
				double sim = e.getValue();
				int w = wordIdxer.indexOf(word);
				double w_weight = wordWeights.value(w);
				double new_sim = sim * p_weight * w_weight;
				T1.add(p, w, new_sim);
			}
		}
		// T1.normalizeColumns();

		// VectorMath.unitVector(T1, T1);

		DenseMatrix T2 = new DenseMatrix(wordIdxer.size());

		for (int i = 0; i < T2.rowSize(); i++) {
			String word1 = wordIdxer.getObject(i);
			double weight1 = wordWeights.value(i);
			for (int j = i + 1; j < T2.rowSize(); j++) {
				String word2 = wordIdxer.getObject(j);
				double weight2 = wordWeights.value(j);
				double ed = StrUtils.editDistance(word1, word2, true);
				double sim = 1 - ed;

				if (sim < 0.9) {
					continue;
				}

				if (T2.value(i, j) == 0) {
					T2.add(i, j, 1);
					T2.add(j, i, 1);
				}

				// System.out.printf("[%s, %s, %f]\n", word1, word2, ed);
				// tmp.add(i, j, ed);
				// tmp.add(j, i, ed);
			}
		}

		System.out.println(VectorUtils.toCounterMap(T2, wordIdxer, wordIdxer));
		System.out.println();

		for (String word1 : wordSims.keySet()) {
			int w1 = wordIdxer.indexOf(word1);
			double weight1 = wordWeights.value(w1);
			Counter<String> c = wordSims.getCounter(word1);

			for (Entry<String, Double> e : c.entrySet()) {
				String word2 = e.getKey();
				double sim = e.getValue();
				int w2 = wordIdxer.indexOf(word2);
				double weight2 = wordWeights.value(w2);
				double new_sim = sim * weight1 * weight2;

				T2.add(w1, w2, new_sim);
				T2.add(w2, w1, new_sim);
			}
		}

		// T2.normalizeRows();

		DenseMatrix T3 = VectorMath.product(T1, T2);

		DenseMatrix T4 = new DenseMatrix(T3.rowSize());

		VectorMath.productRows(T3, T3, T4, false);

		T4.normalizeColumns();

		DenseVector cents = new DenseVector(phrsIdxer.size());
		DenseVector biases = pBiases.copy();
		DenseVector biases2 = cents.copy();

		biases.normalize();

		Set<String> mWords = Generics.newHashSet();
		mWords.add("cancer");
		mWords.add("cancers");
		mWords.add("tumor");
		mWords.add("tumors");
		mWords.add("breast");

		for (IntPair p : ps) {
			String phrs = StrUtils.join("_", d.subList(p.getFirst(), p.getSecond()));
			int pos = p.getFirst();
			int pid = phrsIdxer.indexOf(phrs);
			biases2.add(pid, 1d / (pos + 1));
		}

		// ArrayMath.addAfterMultiply(biases.values(), 0.5, biases2.values(), 0.5,
		// biases2.values());
		// ArrayMath.multiply(biases.values(), biases2.values(), biases2.values());

		biases2.normalizeAfterSummation();

		ArrayMath.randomWalk(T4.values(), cents.values(), null, 500, 10);

		System.out.println(VectorUtils.toCounter(cents, phrsIdxer).toStringSortedByValues(true, true, 50, "\t"));

		return null;
	}

	private DenseVector getSubDocumentVector(DenseVector t, DenseVector dv) {
		DenseVector ret = new DenseVector(dv.size());
		for (int i = 0; i < t.size(); i++) {
			if (t.value(i) != 0) {
				ret.add(i, dv.value(i));
			}
		}
		return ret;
	}

}
