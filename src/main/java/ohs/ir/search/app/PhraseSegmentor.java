package ohs.ir.search.app;

import java.util.List;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.search.index.InvertedIndex;
import ohs.ir.search.index.PostingList;
import ohs.math.CommonMath;
import ohs.types.generic.Counter;
import ohs.types.number.IntegerArray;
import ohs.utils.StrUtils;

public class PhraseSegmentor {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DocumentSearcher ds = new DocumentSearcher(MIRPath.TREC_CDS_2016_COL_DC_DIR, MIRPath.STOPWORD_INQUERY_FILE);

		PhraseSegmentor ps = new PhraseSegmentor(ds);

		Counter<String> phrsWeights = FileUtils.readStringCounterFromText(MIRPath.PHRS_DIR + "phrs_weight.txt");
		int cnt = 0;
		for (String phrs : phrsWeights.getSortedKeys()) {
			ps.segment(phrs);

			if (cnt++ > 200) {
				break;
			}
		}

		System.out.println("process ends.");
	}

	private DocumentSearcher ds;

	public PhraseSegmentor(DocumentSearcher ds) {
		this.ds = ds;
	}

	public void segment(String phrs) throws Exception {
		List<String> words = StrUtils.split(phrs);
		IntegerArray Q = new IntegerArray(ds.getVocab().indexesOf(words, -1));

		PmiEstimator pe = new PmiEstimator(ds.getInvertedIndex());

		InvertedIndex ii = ds.getInvertedIndex();

		PostingList pl0 = ii.getPostingList(Q, true, 1);

		int cnt_xy0 = pl0.size();

		for (int i = 1; i < Q.size(); i++) {
			IntegerArray QL = Q.subArray(0, i);
			IntegerArray QR = Q.subArray(i, Q.size());

			String pL = StrUtils.join(" ", words, 0, i);
			String pR = StrUtils.join(" ", words, i, words.size());

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
				pmi = CommonMath.pmi(cnt_x, cnt_y, cnt_xy2, doc_cnt, true, false);
				// double pr_x = 1d * cnt_x / doc_cnt;
				// double pr_y = 1d * cnt_y / doc_cnt;
				// double pr_xy = 1d * cnt_xy2 / cnt_xy1;
//				pmi = CommonMath.pmi(pr_x, pr_y, pr_xy, true, false);
			}

			// double pmi = pe.pmi(QL, QR);

			System.out.printf("[%s # %s] => %d, %d, %d, %d, %d => %f\n", StrUtils.join(" ", words, 0, i),
					StrUtils.join(" ", words, i, words.size()), cnt_x, cnt_y, cnt_xy0, cnt_xy1, cnt_xy2, pmi);
		}

		System.out.println();
	}

}
