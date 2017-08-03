package ohs.ir.search.model;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.search.app.DocumentSearcher;
import ohs.ir.search.index.InvertedIndex;
import ohs.ir.search.index.PostingList;
import ohs.math.ArrayUtils;
import ohs.math.VectorMath;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.utils.StrUtils;

public class MRFScorer extends LMScorer {

	protected DenseVector mixtures = new DenseVector(new double[] { 0.85, 0.1, 0.05 });

	protected int phrs_size = 3;

	protected int window_size = 5;

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

		if (docPriors != null) {
			for (int i = 0; i < scores.size(); i++) {
				int w = scores.indexAt(i);
				double score = scores.valueAt(i);
				double doc_prior = docPriors.value(w);
				scores.setAt(i, score * doc_prior);
			}
		}

		scores.summation();
		scores.sortValues();

	}

	@Override
	public SparseVector scoreFromCollection(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector s1 = mixtures.value(0) > 0 ? super.scoreFromCollection(Q, docs) : new SparseVector(ArrayUtils.copy(docs.indexes()));
		SparseVector s2 = mixtures.value(1) > 0 ? scoreOrderedPhrases(Q, docs) : new SparseVector(ArrayUtils.copy(docs.indexes()));
		SparseVector s3 = mixtures.value(2) > 0 ? scoreUnorderedPhrases(Q, docs) : new SparseVector(ArrayUtils.copy(docs.indexes()));

		for (int i = 0; i < docs.size(); i++) {
			int dseq = docs.indexAt(i);
			DenseVector scores = new DenseVector(new double[] { s1.valueAt(i), s2.valueAt(i), s3.valueAt(i) });
			double score = VectorMath.dotProduct(mixtures, scores);
			s1.setAt(i, score);
		}
		s1.summation();
		return s1;
	}

	@Override
	public SparseVector scoreFromIndex(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector s1 = mixtures.value(0) > 0 ? super.scoreFromIndex(Q, docs) : new SparseVector(ArrayUtils.copy(docs.indexes()));
		SparseVector s2 = mixtures.value(1) > 0 ? scoreOrderedPhrases(Q, docs) : new SparseVector(ArrayUtils.copy(docs.indexes()));
		SparseVector s3 = mixtures.value(2) > 0 ? scoreUnorderedPhrases(Q, docs) : new SparseVector(ArrayUtils.copy(docs.indexes()));

		for (int i = 0; i < docs.size(); i++) {
			int dseq = docs.indexAt(i);
			DenseVector scores = new DenseVector(new double[] { s1.valueAt(i), s2.valueAt(i), s3.valueAt(i) });
			double score = VectorMath.dotProduct(mixtures, scores);
			s1.setAt(i, score);
		}
		s1.summation();
		return s1;
	}

	public SparseVector scoreOrderedPhrases(SparseVector Q, SparseVector docs) throws Exception {
		return scorePhrases(Q, docs, true, 1);
	}

	public SparseVector scorePhrases(SparseVector Q, SparseVector docs, boolean keep_order, int window_size) throws Exception {
		SparseVector ret = new SparseVector(ArrayUtils.copy(docs.indexes()));

		String str = StrUtils.join(" ", vocab.getObjects(Q.indexes()));

		for (int i = 0; i < Q.size() - 1; i++) {
			for (int j = 2; j <= phrs_size; j++) {
				if (i + j > Q.size()) {
					break;
				}

				IntegerArray subQ = new IntegerArray(Q.subVector(i, j).indexes());
				String phrs = StrUtils.join(" ", vocab.getObjects(subQ));
				PostingList pl = ii.getPostingList(subQ, keep_order, window_size);

				if (pl == null) {
					continue;
				}
				
				super.score(pl.getWord(), 1, pl, ret);
			}
		}
		return ret;
	}

	public SparseVector scoreUnorderedPhrases(SparseVector Q, SparseVector docs) throws Exception {
		return scorePhrases(Q, docs, false, window_size);
	}

	public void setCliqueTypeMixtures(double[] mixtures) {
		this.mixtures.setValues(mixtures);
		this.mixtures.normalizeAfterSummation();
	}

	public void setPhraseSize(int phrs_size) {
		this.phrs_size = phrs_size;
	}

	public void setWindowSize(int window_size) {
		this.window_size = window_size;
	}

}
