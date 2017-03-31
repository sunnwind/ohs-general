package ohs.corpus.search.model;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.type.DocumentCollection;
import ohs.matrix.SparseVector;
import ohs.types.generic.Vocab;

public abstract class Scorer {

	protected Vocab vocab;

	protected DocumentCollection dc;

	protected InvertedIndex ii;

	public Scorer(DocumentSearcher ds) {
		this(ds.getVocab(), ds.getDocumentCollection(), ds.getInvertedIndex());
	}

	public Scorer(Vocab vocab, DocumentCollection dc, InvertedIndex ii) {
		this.vocab = vocab;
		this.dc = dc;
		this.ii = ii;
	}

	public void postprocess(SparseVector scores) {

	}

	public abstract SparseVector score(SparseVector Q, SparseVector docs) throws Exception;

}
