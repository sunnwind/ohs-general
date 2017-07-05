package ohs.ir.search.app;

import java.util.List;
import java.util.Map.Entry;

import ohs.corpus.type.DocumentCollection;
import ohs.eden.keyphrase.mine.PhraseMapper;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
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

		// DocumentSearcher ds = new DocumentSearcher(MIRPath.DATA_DIR + "merged/col/dc/", MIRPath.STOPWORD_INQUERY_FILE);
		// DocumentCollection dc = new DocumentCollection(MIRPath.DATA_DIR + "merged/col/dc/");
		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);

		int dseq = 1000;

		Pair<String, IntegerArray> p = dc.get(dseq);

		IntegerArrayMatrix d = DocumentCollection.toMultiSentences(p.getSecond());

		String text = DocumentCollection.toText(dc.getVocab(), d);
		System.out.println(text);

		KeyphraseExtractor ke = new KeyphraseExtractor(dc, null, phrsWeights);
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

	public SparseVector extract(List<String> d) {
		// System.out.println(candCnts.toStringSortedByValues(true, true, candCnts.size(), "\t"));

		List<IntPair> ps = pm.map(d);
		CounterMap<String, String> cm1 = Generics.newCounterMap();

		{
			for (IntPair p : ps) {
				String phrs = StrUtils.join("_", d.subList(p.getFirst(), p.getSecond()));
				int lower_boud = Math.max(0, p.getFirst() - window_size);
				int upper_bound = Math.min(p.getSecond() + window_size, d.size());

				for (int i = lower_boud; i < p.getFirst(); i++) {
					double dist = p.getFirst() - i;
					double score = 1d / dist;
					String word = d.get(i);
					cm1.incrementCount(phrs, word, score);
				}

				for (int i = p.getSecond(); i < upper_bound; i++) {
					double dist = i - p.getSecond() + 1;
					double score = 1d / dist;
					String word = d.get(i);
					cm1.incrementCount(phrs, word, score);
				}
			}

			CounterMap<String, String> cm2 = Generics.newCounterMap();

			for (int i = 0; i < d.size(); i++) {
				String word1 = d.get(i);
				int m = Math.min(i + window_size, d.size());

				for (int j = i + 1; j < m; j++) {
					String word2 = d.get(j);
					double dist = j - i;
					double score = 1d / dist;
					cm2.incrementCount(word1, word2, score);
				}
			}

			CounterMap<String, String> cm3 = Generics.newCounterMap();
			cm3.incrementAll(cm1, 0.5);
			cm3.incrementAll(cm2, 0.5);
		}

		CounterMap<String, String> cm2 = Generics.newCounterMap();

		for (String phrs1 : cm1.keySet()) {
			Counter<String> c = cm1.getCounter(phrs1);

			for (Entry<String, Double> e : c.entrySet()) {
				String phrs2 = e.getKey();
				if (cm1.containsKey(phrs2)) {
					cm2.incrementCount(phrs1, phrs2, e.getValue());
				}
			}
		}

		Counter<String> wordWeights = Generics.newCounter();
		Counter<String> phrsWeights = Generics.newCounter();

		for (String phrs : cm2.keySet()) {
			double weight = 0;
			for (String word : phrs.split("_")) {
				if (!wordWeights.containsKey(word)) {
					double doc_freq = vocab.getDocFreq(word);
					double cnt = vocab.getCount(word);
					double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), doc_freq);
					wordWeights.setCount(word, tfidf);
					weight += tfidf;
				}
			}
			phrsWeights.setCount(phrs, weight);
		}

		for (String phrs1 : cm2.keySet()) {
			Counter<String> c = cm2.getCounter(phrs1);
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

		Indexer<String> phrsIdxer = Generics.newIndexer(cm2.keySet());

		DenseMatrix T = new DenseMatrix(phrsIdxer.size());

		for (String phrs1 : cm1.keySet()) {
			int pid1 = phrsIdxer.indexOf(phrs1);
			Counter<String> c = cm2.getCounter(phrs1);

			for (Entry<String, Double> e : c.entrySet()) {
				String phrs2 = e.getKey();
				double sim = e.getValue();
				int pid2 = phrsIdxer.indexOf(phrs2);
				T.add(pid1, pid2, sim);
				T.add(pid2, pid1, sim);
			}
		}

		T.normalizeColumns();

		SparseVector cents = VectorUtils.toSparseVector(phrsWeights, phrsIdxer);
		cents.setAll(0);

		SparseVector biases = cents.copy();
		biases.setAll(0);

		for (int i = 0; i < biases.size(); i++) {
			int pid = biases.indexAt(i);
			String phrs = phrsIdxer.getObject(pid);
			double bias = phrsBiases.getCount(phrs);
			biases.addAt(i, bias);
		}
		biases.normalize();

		ArrayMath.randomWalk(T.values(), cents.values(), null, 100);
		//
		// // System.out.println(VectorUtils.toCounter(biases, dc.getVocab()).toStringSortedByValues(true, true, 20, "\t"));
		System.out.println(VectorUtils.toCounter(cents, phrsIdxer).toStringSortedByValues(true, true, 50, "\t"));

		return null;
	}

	public Counter<String> extract(String d) {

		return null;
	}

}
