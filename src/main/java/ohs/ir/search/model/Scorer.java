package ohs.ir.search.model;

import ohs.corpus.type.DocumentCollection;
import ohs.ir.search.app.DocumentSearcher;
import ohs.ir.search.index.InvertedIndex;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Vocab;

public abstract class Scorer {

	protected DocumentCollection dc;

	protected DenseVector docPriors;

	protected InvertedIndex ii;

	protected Vocab vocab;

	public Scorer(DocumentSearcher ds) {
		this(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex());
	}

	public Scorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii) {
		this.vocab = vocab;
		this.dc = dc;
		this.ii = ii;
	}

	public void postprocess(SparseVector scores) {
		scores.sortValues();
	}

	public SparseVector score(SparseVector Q, SparseVector docs, boolean use_index) throws Exception {
		return use_index ? scoreFromIndex(Q, docs) : scoreFromCollection(Q, docs);
	}

	public abstract SparseVector scoreFromCollection(SparseVector Q, SparseVector docs) throws Exception;

	public abstract SparseVector scoreFromIndex(SparseVector Q, SparseVector docs) throws Exception;

	public void setDocumentPriors(DenseVector docPriors) {
		this.docPriors = docPriors;
	}

}
