package ohs.corpus.search.model;

import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.search.index.PostingList;
import ohs.corpus.type.DocumentCollection;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class WeightedMRFScorer extends MRFScorer {

	private IntegerArray srcCnts;

	private List<String> phrss;

	private InvertedIndex pii;

	public WeightedMRFScorer(DocumentSearcher ds, Counter<String> phrss) {
		this(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex(), phrss);
	}

	public WeightedMRFScorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii, Counter<String> phrsCnts) {
		super(vocab, dc, ii);
		createPhraseInvertedIndex(phrsCnts);
	}

	public void createPhraseInvertedIndex(Counter<String> phrsCnts) {
		ListMapMap<Integer, Integer, Integer> lmm = Generics.newListMapMap();

		phrss = Generics.newArrayList(phrsCnts.size());
		srcCnts = new IntegerArray(phrsCnts.size());

		for (Entry<String, Double> e : phrsCnts.entrySet()) {
			phrss.add(e.getKey());
			srcCnts.add(e.getValue().intValue());
		}

		for (int i = 0; i < phrss.size(); i++) {
			String phrs = phrss.get(i);
			String[] words = phrs.split(" ");
			IntegerArray ws = new IntegerArray(words.length);

			for (int j = 0; j < words.length; j++) {
				String word = words[j];
				int w = vocab.indexOf(word);
				if (w == -1) {
					continue;
				}
				ws.add(w);
			}

			if (words.length == ws.size()) {
				for (int j = 0; j < ws.size(); j++) {
					int w = ws.get(j);
					lmm.put(w, i, j);
				}
			}
		}

		IntegerArray ws = new IntegerArray(lmm.keySet());
		ws.sort(false);

		HashMap<Integer, PostingList> plm = Generics.newHashMap(ws.size());

		for (int i = 0; i < ws.size(); i++) {
			int w = ws.get(i);
			ListMap<Integer, Integer> lm = lmm.get(w);

			IntegerArray dseqs = new IntegerArray(lm.keySet());
			dseqs.sort(false);

			IntegerArrayMatrix posData = new IntegerArrayMatrix(dseqs.size());

			for (int dseq : dseqs) {
				IntegerArray poss = new IntegerArray(lm.get(dseq));
				poss.sort(false);
				posData.add(poss);
			}

			PostingList pl = new PostingList(w, dseqs, posData);
			plm.put(w, pl);
		}

		pii = new InvertedIndex(plm, phrsCnts.size(), vocab);
	}

	public SparseVector score(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector s1 = scoreUnigrams(Q, docs);
		SparseVector s2 = scoreOrderedPhrases(Q, docs);
		SparseVector s3 = scoreUnorderedPhrases(Q, docs);

		if (mixtures.sum() != 1) {
			mixtures.normalize();
		}

		for (int i = 0; i < docs.size(); i++) {
			int dseq = docs.indexAt(i);
			DenseVector scores = new DenseVector(new double[] { s1.valueAt(i), s2.valueAt(i), s3.valueAt(i) });
			double score = VectorMath.dotProduct(mixtures, scores);
			s1.setAt(i, score);
		}
		s1.summation();
		return s1;
	}

	protected SparseVector scoreOrderedPhrases(SparseVector Q, SparseVector docs) throws Exception {
		return scorePhrases(new IntegerArray(Q.indexes()), docs, true, 1);
	}

	protected SparseVector scorePhrases(IntegerArray Q, SparseVector docs, boolean keep_order, int window_size) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));
		SparseVector tmp = ret.copy();

		String Qstr = StrUtils.join(" ", vocab.getObjects(Q.values()));

		for (int s = 0; s < Q.size() - 1; s++) {
			int r1 = s + 2;
			int r2 = Math.min(r1 + phrs_size, Q.size() + 1);

			for (int e = r1; e < r2; e++) {
				IntegerArray Qsub = Q.subArray(s, e);
				String Qsubstr = StrUtils.join(" ", vocab.getObjects(Qsub));

				PostingList pl = ii.getPostingList(Qsub, keep_order, window_size);

				if (pl == null) {
					continue;
				}

				PostingList ppl = pii.getPostingList(Qsub, keep_order, window_size);
				double phrs_weight = 1;

				if (ppl != null) {
					// System.out.printf("phrs=[%s], %s\n", phrs, ppl);
					//
					IntegerArray dseqs = ppl.getDocSeqs();
					double avg_src_cnt = 0;
					for (int k = 0; k < dseqs.size(); k++) {
						int dseq = dseqs.get(k);
						String phrs = phrss.get(dseq);
						int src_cnt = srcCnts.get(dseq);
						avg_src_cnt += src_cnt;
					}
					avg_src_cnt /= dseqs.size();
					double ratio = 1f * ppl.size() / pii.getDocCnt();
					phrs_weight = Math.exp(ratio * avg_src_cnt);
				}

				tmp.setAll(0);

				score(pl.getWord(), 1, pl, tmp);

				for (int j = 0; j < ret.size(); j++) {
					ret.addAt(j, tmp.valueAt(j) * phrs_weight);
				}
			}
		}
		return ret;
	}

	private SparseVector scoreUnigrams(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));

		SparseVector tmp = ret.copy();

		for (int i = 0; i < Q.size(); i++) {
			int w = Q.indexAt(i);
			double pr_w_in_q = Q.probAt(i);
			String word = vocab.getObject(w);

			PostingList pl = ii.getPostingList(w);

			if (pl == null) {
				continue;
			}
			// System.out.printf("word=[%s], %s\n", word, pl.toString());

			tmp.setAll(0);

			score(w, pr_w_in_q, pl, tmp);

			PostingList ppl = pii.getPostingList(w);

			double phrs_weight = 1;

			if (ppl != null) {
				IntegerArray dseqs = ppl.getDocSeqs();
				double avg_src_cnt = 0;
				for (int k = 0; k < dseqs.size(); k++) {
					int dseq = dseqs.get(k);
					String phrs = phrss.get(dseq);
					int src_cnt = srcCnts.get(dseq);
					avg_src_cnt += src_cnt;
				}
				avg_src_cnt /= dseqs.size();
				double ratio = 1f * ppl.size() / pii.getDocCnt();
				phrs_weight = Math.exp(ratio * avg_src_cnt);
			}

			for (int j = 0; j < ret.size(); j++) {
				ret.addAt(j, tmp.valueAt(j) * phrs_weight);
			}
		}
		return ret;
	}

	protected SparseVector scoreUnorderedPhrases(SparseVector Q, SparseVector docs) throws Exception {
		return scorePhrases(new IntegerArray(Q.indexes()), docs, false, window_size);
	}

}
