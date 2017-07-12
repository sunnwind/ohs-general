package ohs.ir.search.app;

import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
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
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class KeyphraseExtractorOld {

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

		IntegerArrayMatrix d = DocumentCollection.toMultiSentences(p.getSecond());

		String text = DocumentCollection.toText(dc.getVocab(), d);
		System.out.println(text);

		KeyphraseExtractorOld ke = new KeyphraseExtractorOld(dc, null, phrsWeights);
		ke.extract(StrUtils.split(text));

		// biases.set(dc.getVocab().indexOf("cancer"), 1);
		// biases.set(dc.getVocab().indexOf("breast"), 1);
		// biases.set(dc.getVocab().indexOf("risk"), 1);
		// biases.set(dc.getVocab().indexOf("cohort"), 1);

		System.out.println("process ends.");
	}

	private Indexer<String> phrsIdxer;

	private PhraseMapper<String> pm;

	private Counter<String> dv;

	private int window_size = 5;

	private DocumentCollection dc;

	private Vocab vocab;

	private WordFilter wf;

	private Counter<String> phrsBiases;

	public KeyphraseExtractorOld(DocumentCollection dc, WordFilter wf, Counter<String> phrsBiases) {
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

	public SparseVector extract(List<String> d) {
		// System.out.println(candCnts.toStringSortedByValues(true, true,
		// candCnts.size(), "\t"));

		List<IntPair> ps = pm.map(d);
		CounterMap<String, String> phrsSims = Generics.newCounterMap();
		CounterMap<String, String> wordSims = Generics.newCounterMap();
		Counter<String> phrsCnts = Generics.newCounter();

		{
			for (IntPair p : ps) {
				String phrs = StrUtils.join("_", d.subList(p.getFirst(), p.getSecond()));
				phrsCnts.incrementCount(phrs, 1);

				int lower_boud = Math.max(0, p.getFirst() - window_size);
				int upper_bound = Math.min(p.getSecond() + window_size, d.size());

				for (int i = lower_boud; i < p.getFirst(); i++) {
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
				int m = Math.min(i + window_size, d.size());

				for (int j = i + 1; j < m; j++) {
					String word2 = d.get(j);
					double dist = j - i;
					double score = 1d / dist;
					wordSims.incrementCount(word1, word2, score);
				}
			}
		}

		Counter<String> phrsWeights = Generics.newCounter(phrsSims.size());
		Counter<String> wordWeights = Generics.newCounter(phrsSims.size());

		for (String phrs : phrsSims.keySet()) {
			double weight = 0;
			for (String word : phrs.split("_")) {
				double tfidf = wordWeights.getCount(word);
				if (tfidf == 0) {
					double doc_freq = vocab.getDocFreq(word);
					double cnt = vocab.getCount(word);
					tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), doc_freq);
					wordWeights.setCount(word, tfidf);
				}
				weight += tfidf;
			}
			phrsWeights.setCount(phrs, weight);
		}

		for (String phrs1 : phrsSims.keySet()) {
			Counter<String> c = phrsSims.getCounter(phrs1);
			double weight1 = phrsWeights.getCount(phrs1);

			for (Entry<String, Double> e : c.entrySet()) {
				String phrs2 = e.getKey();
				double sim = e.getValue();
				double weight2 = phrsWeights.getCount(phrs2);
				double new_sim = sim * weight1 * weight2;
				c.setCount(phrs2, new_sim);
			}
		}

		// System.out.println(cm1.toString());

		Indexer<String> phrsIdxer = Generics.newIndexer(phrsSims.keySet());
		Indexer<String> wordIdxer = Generics.newIndexer(wordSims.keySet());

		DenseMatrix T1 = new DenseMatrix(phrsIdxer.size(), wordIdxer.size());

		for (String phrs : phrsSims.keySet()) {
			int p = phrsIdxer.indexOf(phrs);
			double weight1 = phrsWeights.getCount(phrs);
			Counter<String> c = phrsSims.getCounter(phrs);

			for (Entry<String, Double> e : c.entrySet()) {
				String word = e.getKey();
				double sim = e.getValue();
				double weight2 = wordWeights.getCount(word);
				int w = wordIdxer.indexOf(word);
				double new_sim = sim * weight1 * weight2;
				T1.add(p, w, new_sim);
			}
		}

		T1.normalizeColumns();

		DenseMatrix T2 = new DenseMatrix(wordIdxer.size(), wordIdxer.size());

		for (String word1 : wordSims.keySet()) {
			int w1 = wordIdxer.indexOf(word1);
			double weight1 = wordWeights.getCount(word1);
			Counter<String> c = wordSims.getCounter(word1);

			for (Entry<String, Double> e : c.entrySet()) {
				String word2 = e.getKey();
				double sim = e.getValue();
				double weight2 = wordWeights.getCount(word2);
				int w2 = wordIdxer.indexOf(word2);
				double new_sim = sim * weight1 * weight2;

				T2.add(w1, w2, new_sim);
				T2.add(w2, w1, new_sim);
			}
		}

		T2.normalizeRows();

		DenseMatrix T3 = VectorMath.product(T1, T2);

		DenseMatrix T4 = new DenseMatrix(T3.rowSize());

		VectorMath.productRows(T3, T3, T4, false);

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

			biases.addAt(i, bias);
			biases2.addAt(i, bias2);
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

		ArrayMath.randomWalk(T4.values(), cents.values(), null, 500);

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
		//
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

	public Counter<String> extract(String d) {

		return null;
	}

}
