package ohs.corpus.search.app;

import ohs.corpus.search.index.InvertedIndex;
import ohs.corpus.search.index.PostingList;
import ohs.math.CommonMath;

public class PmiEstimator {

	private InvertedIndex ii;

	public PmiEstimator(InvertedIndex ii) {
		this.ii = ii;
	}

	public double pmi(int w1, int w2) throws Exception {
		PostingList pl1 = ii.getPostingList(w1);
		PostingList pl2 = ii.getPostingList(w2);
		int cnt_x = pl1.size();
		int cnt_y = pl2.size();
		int cnt_xy = InvertedIndex.intersection(pl1, pl2).size();
		int doc_cnt = ii.getDocCnt();
		double pmi = CommonMath.pmi(cnt_x, cnt_y, cnt_xy, doc_cnt, false);
		return pmi;
	}

}
