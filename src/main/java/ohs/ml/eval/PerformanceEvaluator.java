package ohs.ml.eval;

import java.util.List;
import java.util.Set;

import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public class PerformanceEvaluator {

	private void countLabelSeqs(List<String> bios, Indexer<String> labelIndexer, Counter<Integer> c) {
		for (int i = 0; i < bios.size();) {
			String s1 = bios.get(i);
			if (s1.startsWith("B-")) {
				int j = i + 1;
				while (j < bios.size()) {
					String s2 = bios.get(j);
					if (!s2.startsWith("I-")) {
						break;
					}
					j++;
				}

				String label = s1.substring(3);
				c.incrementCount(labelIndexer.indexOf(label), 1);
				i = j;
			} else {
				i++;
			}
		}
	}

	private void countLabelSeqs(List<String> anss, List<String> preds, Indexer<String> labelIndexer, Counter<Integer> c) {
		int size = preds.size();

		for (int i = 0; i < size;) {
			String pred = preds.get(i);
			String ans = anss.get(i);

			if (pred.startsWith("B-") && ans.startsWith("B-")) {
				int e1 = i + 1;
				int e2 = i + 1;

				while (e1 < size) {
					String s = preds.get(e1);
					if (!s.startsWith("I-")) {
						break;
					}
					e1++;
				}

				while (e2 < size) {
					String s = anss.get(e2);
					if (!s.startsWith("I-")) {
						break;
					}
					e2++;
				}

				if (e1 == e2) {
					String label = ans.substring(3);
					c.incrementCount(labelIndexer.indexOf(label), 1);
				}
				i = e1;
			} else {
				i++;
			}
		}
	}

	public Performance evalute(IntegerArray y, IntegerArray yh, Indexer<String> labelIndexer) {
		Counter<Integer> corCnts = Generics.newCounter();
		Counter<Integer> predCnts = Generics.newCounter();
		Counter<Integer> ansCnts = Generics.newCounter();
		Set<Integer> labels = Generics.newHashSet();

		for (int i = 0; i < y.size(); i++) {
			int ans = y.get(i);
			int pred = yh.get(i);

			labels.add(ans);
			labels.add(pred);

			ansCnts.incrementCount(ans, 1);
			predCnts.incrementCount(pred, 1);
			if (ans == pred) {
				corCnts.incrementCount(ans, 1);
			}
		}

		IntegerArray ans_cnts = new IntegerArray(labels.size());
		IntegerArray pred_cnts = new IntegerArray(labels.size());
		IntegerArray cor_cnts = new IntegerArray(labels.size());

		for (int i = 0; i < labels.size(); i++) {
			ans_cnts.add((int) ansCnts.getCount(i));
			pred_cnts.add((int) predCnts.getCount(i));
			cor_cnts.add((int) corCnts.getCount(i));

			if (labelIndexer != null) {
				String label = labelIndexer.getObject(i);
				if (label.equals("O")) {
					ans_cnts.set(i, 0);
					cor_cnts.set(i, 0);
					pred_cnts.set(i, 0);
				}
			}
		}

		Performance p = new Performance(ans_cnts, pred_cnts, cor_cnts);
		p.SetLabelIndexer(labelIndexer);
		return p;
	}

	public Performance evalute(IntegerMatrix Y, IntegerMatrix Yh, int label_size) {
		Counter<Integer> corCnts = Generics.newCounter();
		Counter<Integer> predCnts = Generics.newCounter();
		Counter<Integer> ansCnts = Generics.newCounter();
		Set<Integer> labels = Generics.newHashSet();

		for (int i = 0; i < Y.size(); i++) {
			IntegerArray y = Y.get(i);
			IntegerArray yh = Yh.get(i);
			for (int j = 0; j < y.size(); j++) {
				int ans = y.get(j);
				int pred = yh.get(j);

				labels.add(ans);
				labels.add(pred);

				if (ans == pred) {
					corCnts.incrementCount(y.get(j), 1);
				}
			}
		}

		IntegerArray ans_cnts = new IntegerArray(label_size);
		IntegerArray pred_cnts = new IntegerArray(label_size);
		IntegerArray cor_cnts = new IntegerArray(label_size);

		for (int i = 0; i < ansCnts.size(); i++) {
			ans_cnts.add((int) ansCnts.getCount(i));
			pred_cnts.add((int) predCnts.getCount(i));
			cor_cnts.add((int) corCnts.getCount(i));
		}

		return new Performance(ans_cnts, pred_cnts, cor_cnts);
	}

	public Performance evaluteSequences(IntegerMatrix Y, IntegerMatrix Yh, Indexer<String> bioIndexer) {
		Counter<Integer> corCnts = Generics.newCounter();
		Counter<Integer> predCnts = Generics.newCounter();
		Counter<Integer> ansCnts = Generics.newCounter();
		Set<Integer> labels = Generics.newHashSet();

		Indexer<String> labelIndexer = Generics.newIndexer();
		Set<String> labelSet = Generics.newTreeSet();

		for (String bio : bioIndexer) {
			if (!bio.startsWith("O")) {
				labelSet.add(bio.substring(3));
			}
		}
		for (String label : labelSet) {
			labelIndexer.add(label);
		}

		for (int u = 0; u < Y.size(); u++) {
			IntegerArray y = Y.get(u);
			IntegerArray yh = Yh.get(u);

			List<String> anss = Generics.newArrayList(bioIndexer.getObjects(y));
			List<String> preds = Generics.newArrayList(bioIndexer.getObjects(yh));

			countLabelSeqs(anss, labelIndexer, ansCnts);
			countLabelSeqs(preds, labelIndexer, predCnts);
			countLabelSeqs(anss, preds, labelIndexer, corCnts);
		}

		IntegerArray ans_cnts = new IntegerArray(labelIndexer.size());
		IntegerArray pred_cnts = new IntegerArray(labelIndexer.size());
		IntegerArray cor_cnts = new IntegerArray(labelIndexer.size());

		for (int i = 0; i < labels.size(); i++) {
			ans_cnts.add((int) ansCnts.getCount(i));
			pred_cnts.add((int) predCnts.getCount(i));
			cor_cnts.add((int) corCnts.getCount(i));
		}

		return new Performance(ans_cnts, pred_cnts, cor_cnts);
	}

}
