package ohs.eden.keyphrase.mine;

import java.io.File;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import de.bwaldvogel.liblinear.Linear;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.RawDocumentCollection;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.ir.search.index.WordFilter;
import ohs.ir.weight.TermWeighting;
import ohs.math.ArrayMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListList;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * @author Heung-Seon Oh
 * 
 */
public class PhraseRanker {

	private static String LONG_UNDER_BAR = "__";

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> phrsBiases = Generics.newCounter();

		{
			List<String> lines = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/kphrs.txt");
			phrsBiases = Generics.newCounter(lines.size());

			for (String line : lines) {
				String[] ps = line.split("\t");
				String phrs = ps[0];
				// MSentence sent = MSentence.newSentence(phrs);
				//
				// List<String> l = Generics.newArrayList();
				//
				// for (MultiToken mt : sent) {
				// for (Token t : mt) {
				// l.add(String.format("%s_/_%s", t.get(TokenAttr.WORD), t.get(TokenAttr.POS)));
				// }
				// }
				//
				// phrs = StrUtils.join(" ", l);

				double weight = Double.parseDouble(ps[1]);
				phrsBiases.setCount(phrs, weight);
			}
		}

		DocumentCollection dc = new DocumentCollection(KPPath.COL_DC_DIR);
		RawDocumentCollection rdc = new RawDocumentCollection(KPPath.COL_DC_DIR);

		List<String> ins = FileUtils.readLinesFromText(KPPath.KP_DIR + "ext/label_data.txt");

		PhraseRanker pr = new PhraseRanker(dc, null, phrsBiases);
		PhraseNumberPredictor pnp = new PhraseNumberPredictor(
				DocumentCollection.readVocab(KPPath.KP_DIR + "ext/vocab_num_pred.ser"),
				Linear.loadModel(new File(KPPath.KP_DIR + "ext/model_num_pred.txt")));

		for (String line : ins) {
			String[] ps = line.split("\t");
			ps = StrUtils.unwrap(ps);
			List<String> ansPhrss = Generics.newArrayList();

			for (String phrs : ps[0].split(StrUtils.LINE_REP)) {
				ansPhrss.add(phrs);
			}

			String title = ps[1].replace(StrUtils.LINE_REP, "\n");
			String body = ps[2].replace(StrUtils.LINE_REP, "\n");
			String content = title + "\n" + body;
			MDocument doc = MDocument.newDocument(content);
			Counter<MultiToken> phrss = pr.rank(doc);

			System.out.println(phrss.toStringSortedByValues(true, true, phrss.size(), "\t"));

			int pred_phrs_size = pnp.predict(doc);

			System.out.println();

			// ke.extractTFIDF(content);
		}

		int dseq = 1000;

		// ke.extract(StrUtils.split(text));

		// ke.extractLM(StrUtils.split(text));

		System.out.println("process ends.");
	}

	private DocumentCollection dc;

	private Counter<String> dv;

	private Counter<String> phrsBiases;

	private Indexer<String> phrsIdxer;

	private PhrasePatternMapper pm;

	private Vocab vocab;

	private WordFilter wf;

	private int window_size = 5;

	public PhraseRanker(DocumentCollection dc, WordFilter wf, Counter<String> phrsBiases) {
		this.dc = dc;
		this.wf = wf;
		this.vocab = dc.getVocab();
		this.phrsBiases = phrsBiases;

		phrsIdxer = Generics.newIndexer(phrsBiases.size());

		pm = PhrasePatternMapper.newPhrasePatternMapper(phrsBiases.keySet());

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

	private Counter<MultiToken> getPhraseCounts(MDocument doc, ListList<IntPair> psData) {
		Counter<MultiToken> ret = Generics.newCounter();
		for (int i = 0; i < doc.size(); i++) {
			List<Token> ts = doc.get(i).getTokens();
			List<IntPair> ps = psData.get(i);

			for (IntPair p : ps) {
				MultiToken phrs = new MultiToken(ts.subList(p.getFirst(), p.getSecond()));
				ret.incrementCount(phrs, 1);
			}
		}

		return ret;
	}

	private CounterMap<MultiToken, Token> getPhraseToWords(MDocument d, ListList<IntPair> psData) {
		CounterMap<MultiToken, Token> ret = Generics.newCounterMap();

		for (int i = 0; i < d.size(); i++) {
			MSentence s = d.get(i);
			List<IntPair> ps = psData.get(i);

			if (ps.size() == 0) {
				continue;
			}

			List<Token> ts = s.getTokens();

			for (IntPair p : ps) {
				MultiToken phrs = new MultiToken(ts.subList(p.getFirst(), p.getSecond()));

				int lower_boud = Math.max(0, p.getFirst() - window_size);
				int upper_bound = Math.min(p.getSecond() + window_size, d.size());

				for (int j = lower_boud; j < p.getFirst(); j++) {
					double dist = p.getFirst() - j;
					double score = 1d / dist;
					Token t = ts.get(j);
					ret.incrementCount(phrs, t, score);
				}

				for (int j = p.getSecond(); j < upper_bound; j++) {
					double dist = j - p.getSecond() + 1;
					double score = 1d / dist;
					Token t = ts.get(j);
					ret.incrementCount(phrs, t, score);
				}
			}

		}

		return ret;
	}

	private DenseVector getPhraseWeights(Indexer<MultiToken> phrsIdxer, Indexer<Token> wordIdxer,
			DenseVector wordWeights) {
		DenseVector ret = new DenseVector(phrsIdxer.size());
		for (int p = 0; p < phrsIdxer.size(); p++) {
			MultiToken phrs = phrsIdxer.getObject(p);
			double weight = 0;
			for (Token t : phrs) {
				int w = wordIdxer.indexOf(t);
				weight += wordWeights.value(w);
			}
			ret.add(p, weight);
		}
		return ret;
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

	public List<String> getWords(String content) {
		MDocument doc = MDocument.newDocument(content);
		List<String> ret = Generics.newArrayList();
		for (MSentence sent : doc) {
			for (MultiToken mt : sent) {
				for (Token t : mt) {
					ret.add(String.format("%s_/_%s", t.get(TokenAttr.WORD), t.get(TokenAttr.POS)));
				}
			}
		}
		return ret;
	}

	private CounterMap<Token, Token> getWordToWords(MDocument doc) {
		CounterMap<Token, Token> ret = Generics.newCounterMap();
		for (int i = 0; i < doc.size(); i++) {
			MSentence s = doc.get(i);
			List<Token> ts = s.getTokens();

			for (int j = 0; j < ts.size(); j++) {
				Token t1 = ts.get(j);

				double score1 = 1d / (j + 1);
				// phrsBiases.incrementCount(t1, score1);

				int m = Math.min(i + window_size, ts.size());

				for (int k = j + 1; k < m; k++) {
					Token t2 = ts.get(k);
					double dist = k - j;
					double score2 = 1d / dist;
					ret.incrementCount(t1, t2, score2);
				}
			}
		}
		return ret;
	}

	private DenseVector getWordWeights(Indexer<Token> wordIdxer, Counter<Token> wordCnts) {
		DenseVector ret = new DenseVector(wordIdxer.size());

		for (int w = 0; w < wordIdxer.size(); w++) {
			Token t = wordIdxer.getObject(w);
			String s = String.format("%s_/_%s", t.get(TokenAttr.WORD), t.get(TokenAttr.POS));
			double doc_freq = vocab.getDocFreq(s);
			double cnt = wordCnts.getCount(t);
			double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), doc_freq);
			ret.add(w, tfidf);
		}
		return ret;
	}

	public Counter<MultiToken> rank(MDocument doc) {
		ListList<IntPair> ps = pm.map(doc);
		CounterMap<MultiToken, Token> phrsSims = getPhraseToWords(doc, ps);
		CounterMap<Token, Token> wordSims = getWordToWords(doc);
		Counter<MultiToken> phrsCnts = getPhraseCounts(doc, ps);
		Counter<Token> wordCnts = Generics.newCounter();

		for (List<Token> ts : doc.getTokens()) {
			for (Token t : ts) {
				wordCnts.incrementCount(t, 1);
			}
		}

		Counter<String> phrsBiases = Generics.newCounter();

		Indexer<MultiToken> phrsIdxer = Generics.newIndexer(phrsCnts.keySet());
		Indexer<Token> wordIdxer = Generics.newIndexer(wordCnts.keySet());

		DenseVector wordWeights = getWordWeights(wordIdxer, wordCnts);
		DenseVector phrsWeights = getPhraseWeights(phrsIdxer, wordIdxer, wordWeights);
		DenseVector pBiases = new DenseVector(phrsIdxer.size());

		// System.out.println(cm1.toString());

		DenseMatrix T1 = new DenseMatrix(phrsIdxer.size(), wordIdxer.size());

		for (MultiToken phrs : phrsSims.keySet()) {
			int p = phrsIdxer.indexOf(phrs);
			double p_weight = phrsWeights.value(p);
			Counter<Token> c = phrsSims.getCounter(phrs);

			for (Entry<Token, Double> e : c.entrySet()) {
				Token word = e.getKey();
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

		// for (int i = 0; i < T2.rowSize(); i++) {
		// Token t1 = wordIdxer.getObject(i);
		// String word1 = t1.get(TokenAttr.WORD);
		// double weight1 = wordWeights.value(i);
		// for (int j = i + 1; j < T2.rowSize(); j++) {
		// Token t2 = wordIdxer.getObject(j);
		// String word2 = t2.get(TokenAttr.WORD);
		// double weight2 = wordWeights.value(j);
		// double ed = StrUtils.editDistance(word1, word2, true);
		// double sim = 1 - ed;
		//
		// if (sim < 0.9) {
		// continue;
		// }
		//
		// if (T2.value(i, j) == 0) {
		// T2.add(i, j, 1);
		// T2.add(j, i, 1);
		// }
		//
		// // System.out.printf("[%s, %s, %f]\n", word1, word2, ed);
		// // tmp.add(i, j, ed);
		// // tmp.add(j, i, ed);
		// }
		// }

		// System.out.println(VectorUtils.toCounterMap(T2, wordIdxer, wordIdxer));
		System.out.println();

		for (Token t1 : wordSims.keySet()) {
			int w1 = wordIdxer.indexOf(t1);
			double weight1 = wordWeights.value(w1);
			Counter<Token> c = wordSims.getCounter(t1);

			for (Entry<Token, Double> e : c.entrySet()) {
				Token t2 = e.getKey();
				double sim = e.getValue();
				int w2 = wordIdxer.indexOf(t2);
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

		// for (IntPair p : ps) {
		// String phrs = StrUtils.join(LONG_UNDER_BAR, d.subList(p.getFirst(),
		// p.getSecond()));
		// int pos = p.getFirst();
		// int pid = phrsIdxer.indexOf(phrs);
		// biases2.add(pid, 1d / (pos + 1));
		// }

		// ArrayMath.addAfterMultiply(biases.values(), 0.5, biases2.values(), 0.5,
		// biases2.values());
		// ArrayMath.multiply(biases.values(), biases2.values(), biases2.values());

		biases2.normalizeAfterSummation();

		ArrayMath.randomWalk(T4.values(), cents.values(), null, 500);

		Counter<MultiToken> ret = Generics.newCounter();

		for (int w = 0; w < phrsIdxer.size(); w++) {
			MultiToken phrs = phrsIdxer.getObject(w);
			ret.incrementCount(phrs, cents.value(w));
		}

		// System.out.println(VectorUtils.toCounter(cents,
		// phrsIdxer).toStringSortedByValues(true, true, 50, "\t"));

		return ret;
	}

}
