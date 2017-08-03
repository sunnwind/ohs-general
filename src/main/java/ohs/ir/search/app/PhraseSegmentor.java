package ohs.ir.search.app;

import java.util.List;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
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

			if (cnt++ > 10) {
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

		for (int i = 1; i < Q.size(); i++) {
			IntegerArray QL = Q.subArray(0, i);
			IntegerArray QR = Q.subArray(i, Q.size());

			double pmi = pe.pmi(QL, QR);
			
			System.out.printf("[%s|%s] = %f\n", StrUtils.join(" ", words, 0, i), StrUtils.join(" ", words, i, words.size()), pmi);
		}
		System.out.println();
	}

}
