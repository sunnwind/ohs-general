package ohs.ml.feat.select;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import ohs.io.FileUtils;
import ohs.ir.search.index.MemoryInvertedIndex;
import ohs.ir.search.index.PostingList;
import ohs.math.CommonMath;
import ohs.math.VectorUtils;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public class ChisquareComputer {

	private SparseMatrix X;

	private DenseVector Y;

	private Vocab vocab;

	private Indexer<String> labelIdxer;

	private MemoryInvertedIndex ii;

	private DenseVector labelCnts;

	public ChisquareComputer(Indexer<String> labelIdxer, Vocab vocab, SparseMatrix X, DenseVector Y) {
		this.labelIdxer = labelIdxer;
		this.vocab = vocab;
		this.X = X;
		this.Y = Y;

		createInvertedIndex();

		labelCnts = new DenseVector(labelIdxer.size());
	}

	public DenseVector compute(int target) throws Exception {
		int size = (int) labelCnts.value(target);
		IntegerArray dseqs1 = new IntegerArray(size);

		for (int i = 0; i < X.size(); i++) {
			int l = (int) Y.value(i);
			if (l == target) {
				dseqs1.add(i);
			}
		}

		int N = X.size();

		dseqs1.sort(false);

		DenseVector ret = new DenseVector(vocab.size());

		Set<Integer> W = Generics.newHashSet(vocab.size());

		for (int dseq : dseqs1) {
			SparseVector x = X.rowAt(dseq);
			for (int w : x.indexes()) {
				W.add(w);
			}
		}

		for (int w : W) {
			String word = vocab.getObject(w);
			PostingList pl = ii.getPostingList(w);
			IntegerArray dseqs2 = pl.getDocSeqs();

			if (dseqs2.size() < 10) {
				continue;
			}

			double N11 = 0;
			double N01 = 0;
			double N10 = 0;
			double N00 = 0;

			int i = 0, j = 0;

			while (i < dseqs1.size() && j < dseqs2.size()) {
				int dseq1 = dseqs1.get(i);
				int dseq2 = dseqs2.get(j);

				if (dseq1 == dseq2) {
					N11++;
					i++;
					j++;
				} else if (dseq1 > dseq2) {
					N01++;
					j++;
				} else if (dseq1 < dseq2) {
					N10++;
					i++;
				}
			}

			while (i < dseqs1.size()) {
				N10++;
				i++;
			}

			while (j < dseqs2.size()) {
				N01++;
				j++;
			}

			N00 = N - (N11 + N01 + N10);

			double chisquare = CommonMath.chisquare(N11, N10, N01, N00);
			ret.add(w, chisquare);
		}

		return ret;
	}

	private void createInvertedIndex() {
		ListMap<Integer, Integer> lm = Generics.newListMap();

		for (int i = 0; i < X.size(); i++) {
			SparseVector x = X.rowAt(i);
			for (int w : x.indexes()) {
				lm.put(w, i);
			}
		}

		IntegerArray ws = new IntegerArray(lm.keySet());
		ws.sort(false);

		HashMap<Integer, PostingList> plm = Generics.newHashMap(ws.size());

		for (int i = 0; i < ws.size(); i++) {
			int w = ws.get(i);

			IntegerArray dseqs = new IntegerArray(lm.get(w));
			dseqs.sort(false);

			PostingList pl = new PostingList(w, dseqs, new IntegerMatrix());
			plm.put(w, pl);
		}

		ii = new MemoryInvertedIndex(plm, X.rowSize(), vocab);
	}

}
