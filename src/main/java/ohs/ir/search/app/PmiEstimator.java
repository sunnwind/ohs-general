package ohs.ir.search.app;

import ohs.ir.search.index.InvertedIndex;
import ohs.ir.search.index.PostingList;
import ohs.math.CommonMath;
import ohs.types.number.IntegerArray;

public class PmiEstimator {

	private InvertedIndex ii;

	private boolean normalize = false;

	public PmiEstimator(InvertedIndex ii) {
		this.ii = ii;
	}

	public double pmi(int w1, int w2) throws Exception {
		PostingList pl1 = ii.getPostingList(w1);
		PostingList pl2 = ii.getPostingList(w2);
		return pmi(pl1, pl2);
	}

	public double pmi(IntegerArray x1, IntegerArray x2) throws Exception {
		PostingList pl1 = ii.getPostingList(x1, true, 1);
		PostingList pl2 = ii.getPostingList(x2, true, 1);
		return pmi(pl1, pl2);
	}

	public double pmi(PostingList pl1, PostingList pl2) throws Exception {
		double pmi = Double.NEGATIVE_INFINITY;
		if (pl1 == null || pl2 == null) {

		} else {
			int cnt_x = pl1.size();
			int cnt_y = pl2.size();
			int cnt_xy = InvertedIndex.intersection(pl1, pl2).size();
			int doc_cnt = ii.getDocCnt();
			pmi = CommonMath.pmi(cnt_x, cnt_y, cnt_xy, doc_cnt, normalize);
		}
		return pmi;
	}

}
