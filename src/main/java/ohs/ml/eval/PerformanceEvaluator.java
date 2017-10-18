package ohs.ml.eval;

import java.util.List;
import java.util.Set;

import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public class PerformanceEvaluator {

	private Counter<Integer> countCommonLabelSeqs(List<String> anss, List<String> preds, Indexer<String> labelIdxer) {
		Counter<Integer> c = Generics.newCounter(labelIdxer.size());

		int size = preds.size();

		for (int i = 0; i < size;) {
			String pred = preds.get(i);
			String ans = anss.get(i);

			String[] p1 = splitBioLabel(pred);
			String[] a1 = splitBioLabel(ans);

			if (p1[1].equals("B") && a1[1].equals("B")) {
				int e1 = i + 1;
				int e2 = i + 1;

				while (e1 < size) {
					String[] p2 = splitBioLabel(preds.get(e1));
					if (!p2[1].equals("I")) {
						break;
					}
					e1++;
				}

				while (e2 < size) {
					String[] a2 = splitBioLabel(anss.get(e2));
					if (!a2[1].equals("I")) {
						break;
					}
					e2++;
				}

				if (e1 == e2) {
					String label = p1[0];
					c.incrementCount(labelIdxer.indexOf(label), 1);
				}
				i = e1;
			} else {
				i++;
			}
		}
		return c;
	}

	private Counter<Integer> countLabelSeqs(List<String> bioLabels, Indexer<String> labelIdxer) {
		Counter<Integer> ret = Generics.newCounter(labelIdxer.size());

		for (int i = 0; i < bioLabels.size();) {
			String s1 = bioLabels.get(i);
			String[] ps1 = splitBioLabel(s1);
			String l1 = ps1[0];
			String bio1 = ps1[1];

			if (bio1.equals("B")) {
				int j = i + 1;
				while (j < bioLabels.size()) {
					String s2 = bioLabels.get(j);
					String[] ps2 = splitBioLabel(s2);
					String l2 = ps2[0];
					String bio2 = ps2[1];

					if (!bio2.equals("I")) {
						break;
					}
					j++;
				}
				ret.incrementCount(labelIdxer.indexOf(l1), 1);
				i = j;
			} else {
				i++;
			}
		}
		return ret;
	}

	public Performance evalute(IntegerArray Y, IntegerArray Yh, Indexer<String> labelIdxer) {
		Counter<Integer> corCnts = Generics.newCounter();
		Counter<Integer> predCnts = Generics.newCounter();
		Counter<Integer> ansCnts = Generics.newCounter();

		for (int i = 0; i < Y.size(); i++) {
			int ans = Y.get(i);
			int pred = Yh.get(i);

			ansCnts.incrementCount(ans, 1);
			predCnts.incrementCount(pred, 1);

			if (ans == pred) {
				corCnts.incrementCount(ans, 1);
			}
		}

		Set<Integer> labels = Generics.newTreeSet();

		if (labelIdxer == null) {
			labels.addAll(predCnts.keySet());
			labels.addAll(ansCnts.keySet());
		} else {
			for (int i = 0; i < labelIdxer.size(); i++) {
				labels.add(i);
			}
		}

		IntegerArray ans_cnts = new IntegerArray(labels.size());
		IntegerArray pred_cnts = new IntegerArray(labels.size());
		IntegerArray cor_cnts = new IntegerArray(labels.size());

		for (int i = 0; i < labels.size(); i++) {
			ans_cnts.add((int) ansCnts.getCount(i));
			pred_cnts.add((int) predCnts.getCount(i));
			cor_cnts.add((int) corCnts.getCount(i));

			if (labelIdxer != null) {
				String label = labelIdxer.getObject(i);
				if (label.equals("O")) {
					ans_cnts.set(i, 0);
					cor_cnts.set(i, 0);
					pred_cnts.set(i, 0);
				}
			}
		}

		Performance p = new Performance(ans_cnts, pred_cnts, cor_cnts, labelIdxer);
		return p;
	}

	public Performance evalute(IntegerMatrix Y, IntegerMatrix Yh, Indexer<String> labelIdxer) {
		int size = Y.sizeOfEntries();
		IntegerArray Y_ = new IntegerArray(size);
		IntegerArray Yh_ = new IntegerArray(size);
		for (int i = 0; i < Y.size(); i++) {
			Y_.addAll(Y.get(i));
			Yh_.addAll(Yh.get(i));
		}
		return evalute(Y_, Yh_, labelIdxer);
	}

	public Performance evaluteSequences(IntegerMatrix Y, IntegerMatrix Yh, Indexer<String> bioIdxer) {
		Indexer<String> labelIdxer = Generics.newIndexer();

		{
			Set<String> L = Generics.newTreeSet();

			for (String bio : bioIdxer) {
				String[] ps = bio.split("-");
				if (ps.length == 2) {
					String l = ps[0];
					if (l.equals("B") || l.equals("I")) {
						l = ps[1];
					}
					L.add(l);
				}
			}
			labelIdxer.addAll(L);
			labelIdxer.add("O");
		}

		Counter<Integer> corCnts = Generics.newCounter(labelIdxer.size());
		Counter<Integer> predCnts = Generics.newCounter(labelIdxer.size());
		Counter<Integer> ansCnts = Generics.newCounter(labelIdxer.size());

		for (int u = 0; u < Y.size(); u++) {
			IntegerArray y = Y.get(u);
			IntegerArray yh = Yh.get(u);

			List<String> anss = Generics.newArrayList(bioIdxer.getObjects(y));
			List<String> preds = Generics.newArrayList(bioIdxer.getObjects(yh));

			ansCnts.incrementAll(countLabelSeqs(anss, labelIdxer));
			predCnts.incrementAll(countLabelSeqs(preds, labelIdxer));
			corCnts.incrementAll(countCommonLabelSeqs(anss, preds, labelIdxer));
		}

		IntegerArray ans_cnts = new IntegerArray(labelIdxer.size());
		IntegerArray pred_cnts = new IntegerArray(labelIdxer.size());
		IntegerArray cor_cnts = new IntegerArray(labelIdxer.size());

		for (int i = 0; i < labelIdxer.size(); i++) {
			ans_cnts.add((int) ansCnts.getCount(i));
			pred_cnts.add((int) predCnts.getCount(i));
			cor_cnts.add((int) corCnts.getCount(i));
		}

		return new Performance(ans_cnts, pred_cnts, cor_cnts, labelIdxer);
	}

	private String[] splitBioLabel(String bioLabel) {
		String[] ps = bioLabel.split("-");
		String label = "";
		String bio = "O";
		if (ps.length == 2) {
			label = ps[0];
			bio = ps[1];
			if (label.equals("B") || label.equals("I")) {
				label = ps[1];
				bio = ps[0];
			}
		}
		return new String[] { label, bio };
	}

}