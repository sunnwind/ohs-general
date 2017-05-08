package ohs.corpus.search.model;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.search.index.PostingList;
import ohs.corpus.type.DocumentCollection;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.StrUtils;

public class MRFScorer extends LMScorer {

	protected DenseVector mixtures = new DenseVector(new double[] { 0.8, 0.1, 0.1 });

	protected int window_size = 5;

	protected int phrase_size = 3;

	public MRFScorer(DocumentSearcher ds) {
		this(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex());
	}

	public MRFScorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii) {
		super(vocab, dc, ii);
		setType(Type.QL);
	}

	@Override
	public void postprocess(SparseVector scores) {
		VectorMath.softmax(scores);
		scores.sortValues();
	}

	public SparseVector score(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector s1 = super.score(Q, docs);
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

	private SparseVector scoreOrderedPhrases(SparseVector Q, SparseVector docs) throws Exception {
		return scorePhrases(new IntegerArray(Q.indexes()), docs, true, 1);
	}

	private SparseVector scorePhrases(IntegerArray Q, SparseVector docs, boolean keep_order, int window_size) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));

		String str = StrUtils.join(" ", vocab.getObjects(Q.values()));

		for (int s = 0; s < Q.size() - 1; s++) {
			int r1 = s + 2;
			int r2 = Math.min(r1 + phrase_size, Q.size() + 1);

			for (int e = r1; e < r2; e++) {
				String phrs = StrUtils.join(" ", vocab.getObjects(Q.subArray(s, e)));
				// System.out.println(phrs);

				PostingList pl = ii.getPostingList(Q.subArray(s, e), keep_order, window_size);

				if (pl == null) {
					continue;
				}

				super.score(pl.getWord(), 1, pl, ret);
			}
		}
		return ret;
	}

	private SparseVector scoreUnorderedPhrases(SparseVector Q, SparseVector docs) throws Exception {
		return scorePhrases(new IntegerArray(Q.indexes()), docs, false, window_size);
	}

	public void setCliqueTypeMixtures(double[] mixtures) {
		this.mixtures.setValues(mixtures);
	}

	public void setGramSize(int gram_size) {
		this.phrase_size = gram_size;
	}

	public void setWindowSize(int window_size) {
		this.window_size = window_size;
	}

}
