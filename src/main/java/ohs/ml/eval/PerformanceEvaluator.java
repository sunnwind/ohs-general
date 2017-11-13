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

			if (p1[0].equals(a1[0]) && p1[1].equals("B") && a1[1].equals("B")) {
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

	private Set<String> countLabelSeqs(List<String> tags, Indexer<String> labelIdxer) {
		Set<String> ret = Generics.newHashSet();

		for (int i = 0; i < tags.size();) {
			String tag1 = tags.get(i);

			if (tag1.startsWith("B-") || tag1.startsWith("S-")) {
				String prefix1 = tag1.split("-")[0];
				String cat1 = tag1.split("-")[1];

				int match_len = 1;

				if (prefix1.equals("S")) {

				} else if (prefix1.equals("B")) {
					for (int j = i + 1; j < tags.size(); j++) {
						String tag2 = tags.get(j);
						if (tag2.equals("O")) {
							break;
						} else if (tag2.startsWith("B-") || tag2.startsWith("S-")) {
							break;
						}
						match_len++;
					}
				} else {
					System.err.println("wrong tag sequence");
					System.exit(0);
				}
				ret.add(String.format("%d_%d_%s", i, match_len, cat1));
				i += match_len;
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

	public Performance evaluteSequences(IntegerMatrix Y, IntegerMatrix Yh, Indexer<String> tagIdxer) {

		int size = Y.sizeOfEntries();

		Indexer<String> catIdxer = Generics.newIndexer();

		{
			Counter<String> c = Generics.newCounter();

			for (String tag : tagIdxer) {
				if (!tag.equals("O")) {
					String[] ps = tag.split("-");
					String cat = ps[1];
					c.incrementCount(cat, 1);
				}
			}

			if (c.size() > 0) {
				catIdxer.addAll(Generics.newTreeSet(c.keySet()));
				catIdxer.add("O");
			} else {
				catIdxer = Generics.newIndexer(tagIdxer);
			}
		}

		Counter<Integer> corCnts = Generics.newCounter(catIdxer.size());
		Counter<Integer> predCnts = Generics.newCounter(catIdxer.size());
		Counter<Integer> ansCnts = Generics.newCounter(catIdxer.size());

		for (int u = 0; u < Y.size(); u++) {
			IntegerArray y = Y.get(u);
			IntegerArray yh = Yh.get(u);

			List<String> anss = Generics.newArrayList(tagIdxer.getObjects(y));
			List<String> preds = Generics.newArrayList(tagIdxer.getObjects(yh));

			Set<String> s1 = countLabelSeqs(anss, catIdxer);
			Set<String> s2 = countLabelSeqs(preds, catIdxer);
			Set<String> s3 = Generics.newHashSet(s1.size());

			for (String ans : s2) {
				if (s1.contains(ans)) {
					s3.add(ans);
				}
			}

			for (String tag : s1) {
				String[] ps = tag.split("_");
				ansCnts.incrementCount(catIdxer.indexOf(ps[2]), 1);
			}

			for (String tag : s2) {
				String[] ps = tag.split("_");
				predCnts.incrementCount(catIdxer.indexOf(ps[2]), 1);
			}

			for (String tag : s3) {
				String[] ps = tag.split("_");
				corCnts.incrementCount(catIdxer.indexOf(ps[2]), 1);
			}

			// ansCnts.incrementAll(countLabelSeqs(anss, labelIdxer));
			// predCnts.incrementAll(countLabelSeqs(preds, labelIdxer));
			// corCnts.incrementAll(countCommonLabelSeqs(anss, preds, labelIdxer));
		}

		IntegerArray ans_cnts = new IntegerArray(catIdxer.size());
		IntegerArray pred_cnts = new IntegerArray(catIdxer.size());
		IntegerArray cor_cnts = new IntegerArray(catIdxer.size());

		for (int i = 0; i < catIdxer.size(); i++) {
			ans_cnts.add((int) ansCnts.getCount(i));
			pred_cnts.add((int) predCnts.getCount(i));
			cor_cnts.add((int) corCnts.getCount(i));
		}

		return new Performance(ans_cnts, pred_cnts, cor_cnts, catIdxer);
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