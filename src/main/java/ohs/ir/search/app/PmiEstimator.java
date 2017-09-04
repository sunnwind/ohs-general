package ohs.ir.search.app;

import java.util.List;

import ohs.ir.search.index.InvertedIndex;
import ohs.ir.search.index.PostingList;
import ohs.math.CommonMath;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

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

	public List<Pair<Integer, Double>> pmi(IntegerArray Q) throws Exception {
		List<Pair<Integer, Double>> ret = Generics.newArrayList(Q.size() + 1);
		ret.add(Generics.newPair(0, (double) ii.getPostingList(Q, true, 1).size()));

		for (int i = 1; i < Q.size(); i++) {
			IntegerArray QL = Q.subArray(0, i);
			IntegerArray QR = Q.subArray(i, Q.size());

			PostingList pl1 = ii.getPostingList(QL, true, 1);
			PostingList pl2 = ii.getPostingList(QR, true, 1);

			pl1.setEndPosData(null);
			pl2.setEndPosData(null);

			double pmi = Double.NEGATIVE_INFINITY;
			int cnt_x = pl1.size();
			int cnt_y = pl2.size();
			int cnt_xy1 = 0;
			int cnt_xy2 = 0;
			int doc_cnt = ii.getDocCnt();

			if (pl1.size() == 0 || pl2.size() == 0) {

			} else {
				cnt_x = pl1.size();
				cnt_y = pl2.size();
				cnt_xy1 = InvertedIndex.intersection(pl1, pl2).size();
				cnt_xy2 = InvertedIndex.findCollocations(pl1, pl2, true, QL.size()).size();
				pmi = CommonMath.pmi(cnt_x, cnt_y, cnt_xy2, doc_cnt, true, normalize);
			}
			ret.add(Generics.newPair(i, pmi));
		}

		return ret;
	}

	public double pmi(IntegerArray Q, int sep_idx) throws Exception {
		IntegerArray x = Q.subArray(0, sep_idx);
		IntegerArray y = Q.subArray(sep_idx, Q.size());

		PostingList pl1 = ii.getPostingList(x, true, 1);
		PostingList pl2 = ii.getPostingList(y, true, 1);

		pl1.setEndPosData(null);
		pl2.setEndPosData(null);

		int cnt_x = pl1.size();
		int cnt_y = pl2.size();
		int cnt_xy = InvertedIndex.findCollocations(pl1, pl2, true, x.size()).size();
		int doc_cnt = ii.getDocCnt();
		double pmi = CommonMath.pmi(cnt_x, cnt_y, cnt_xy, doc_cnt, true, normalize);
		return pmi;
	}

	public double pmi(IntegerArray x1, IntegerArray x2) throws Exception {
		PostingList pl1 = ii.getPostingList(x1, true, 1);
		PostingList pl2 = ii.getPostingList(x2, true, 1);
		return pmi(pl1, pl2);
	}

	public double pmi(PostingList pl1, PostingList pl2) throws Exception {
		int cnt_x = pl1.size();
		int cnt_y = pl2.size();
		int cnt_xy = InvertedIndex.intersection(pl1, pl2).size();
		int doc_cnt = ii.getDocCnt();
		double pmi = CommonMath.pmi(cnt_x, cnt_y, cnt_xy, doc_cnt, true, normalize);
		return pmi;
	}

	public void setNormalize(boolean normalize) {
		this.normalize = normalize;
	}

}
